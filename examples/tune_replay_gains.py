#!/usr/bin/env python3
"""
Auto-tune PD gains for replay by sweeping parameters and measuring control error.
Finds optimal gains that minimize tracking error while maintaining stability.

Example usage:
    python tune_replay_gains.py --hand orca --obj box --coarse
    python tune_replay_gains.py --hand orca --obj box --fine --kp_range 60 100 --kv_range 4 8
    python tune_replay_gains.py --hand orca --obj box --best_only  # Just test current best
"""

import os
import sys
import argparse
import torch
import numpy as np
import genesis as gs
import re
from tqdm import tqdm
import json
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg, get_hand_specific_cfg
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.demo_data import get_demo_data

RETARGET_DIR = get_asset_path("retargeted")


def run_replay_with_gains(args, finger_kp, finger_kv, wrist_rot_kp, wrist_rot_kv, 
                          wrist_trans_kp, wrist_trans_kv, force_range,
                          scene, hands, obj, hand_qposes, num_steps, verbose=False):
    """Run replay with specified gains and return average control error."""
    
    device = torch.device("cuda")
    hand_name = args.hand if "hand" in args.hand else f"{args.hand}_hand"
    
    # Set PD gains with the test parameters (scene already built)
    hand_cfg = get_hand_specific_cfg(name=hand_name)
    
    for side in ['left', 'right']:
        hand = hands[side]
        actuator_cfgs = hand_cfg[side]['actuators']
        
        for joint_group, act_cfg in actuator_cfgs.items():
            joint_exprs = act_cfg['joint_exprs']
            
            # Use test gains
            if joint_group == 'finger':
                kp, kv = finger_kp, finger_kv
            elif joint_group == 'wrist_rot':
                kp, kv = wrist_rot_kp, wrist_rot_kv
            elif joint_group == 'wrist_trans':
                kp, kv = wrist_trans_kp, wrist_trans_kv
            else:
                continue
            
            fr = force_range
            
            # Find joints matching the patterns
            joint_names, joint_idxs = [], []
            for joint in hand.entity.joints:
                if joint.type not in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]:
                    continue
                jname = joint.name
                for expr in joint_exprs:
                    if bool(re.match(expr, jname)):
                        joint_names.append(jname)
                        joint_idxs.append(joint.dof_idx_local)
                        break
            
            if len(joint_idxs) > 0:
                num_joints = len(joint_idxs)
                kp_tensor = torch.tensor([kp] * num_joints, dtype=torch.float32)
                kv_tensor = torch.tensor([kv] * num_joints, dtype=torch.float32)
                fr_tensor = torch.tensor([fr] * num_joints, dtype=torch.float32)
                
                hand.entity.set_dofs_kp(kp_tensor, dofs_idx_local=joint_idxs)
                hand.entity.set_dofs_kv(kv_tensor, dofs_idx_local=joint_idxs)
                hand.entity.set_dofs_force_range(-fr_tensor, fr_tensor, dofs_idx_local=joint_idxs)
    
    # Reset scene (set initial states)
    
    for side in ['left', 'right']:
        hand = hands[side]
        init_qpos = hand_qposes[side][0:1]
        hand.set_joint_position(init_qpos, env_idxs=[0])
        hand.entity.zero_all_dofs_velocity(envs_idx=[0])
    
    # Run replay
    control_errors = {'left': [], 'right': []}
    max_forces = {'left': [], 'right': []}
    
    # Track object state for task success
    obj_positions = []
    obj_quaternions = []
    obj_articulations = []
    
    for step in range(num_steps):
        # Control hand positions using PD controller
        for side in ['left', 'right']:
            hand = hands[side]
            target_qpos = hand_qposes[side][step:step+1]
            hand.control_joint_position(target_qpos, env_idxs=[0])
        
        scene.step()
        
        # Track object state
        obj_positions.append(obj.root_pos[0].cpu().numpy())
        obj_quaternions.append(obj.root_quat[0].cpu().numpy())
        obj_articulations.append(obj.dof_pos[0].cpu().numpy())
        
        # Track control errors and forces
        for side in ['left', 'right']:
            hand = hands[side]
            hand.update_value_buffers()
            err = hand.get_control_errors()
            control_errors[side].append(err.mean().item())
            
            # Get max control force
            control_force = hand.entity.get_dofs_control_force(dofs_idx_local=hand.actuated_dof_idxs)[0]
            max_forces[side].append(torch.abs(control_force).max().item())
    
    # Calculate control metrics
    avg_error_left = np.mean(control_errors['left'])
    avg_error_right = np.mean(control_errors['right'])
    avg_error = (avg_error_left + avg_error_right) / 2
    max_error = max(avg_error_left, avg_error_right)
    
    max_force_left = np.max(max_forces['left'])
    max_force_right = np.max(max_forces['right'])
    max_force = max(max_force_left, max_force_right)
    
    # Check for force saturation (convert to native bool)
    force_saturated = bool(max_force >= force_range * 0.95)
    
    # Calculate task success metrics
    obj_positions = np.array(obj_positions)
    obj_articulations = np.array(obj_articulations)
    
    # Task metric 1: Max articulation reached (how much the box opened)
    max_articulation = np.max(np.abs(obj_articulations))
    final_articulation = np.abs(obj_articulations[-1])
    
    # Task metric 2: Did object fall? (position too low)
    min_height = np.min(obj_positions[:, 2])
    object_fell = min_height < 0.85  # Below table height
    
    # Task metric 3: Articulation progress (normalized to demo)
    # Load demo articulation for comparison
    hand_name = args.hand if "hand" in args.hand else f"{args.hand}_hand"
    fname = f"{RETARGET_DIR}/{hand_name}/{args.subject}/{args.obj}_use_{args.use_clip}_vector_para.pt"
    data = torch.load(fname, weights_only=False)
    demo_data = data['demo_data']
    demo_arti = demo_data['obj_arti'][args.start_frame:args.start_frame+num_steps]
    demo_max_arti = np.max(np.abs(demo_arti))
    
    articulation_ratio = max_articulation / (demo_max_arti + 1e-6)  # How much of demo articulation achieved
    task_success = articulation_ratio > 0.7 and not object_fell  # Success if >70% articulation and didn't fall
    
    if verbose:
        print(f"  Error L:{avg_error_left:.4f} R:{avg_error_right:.4f} | "
              f"Max Force L:{max_force_left:.1f} R:{max_force_right:.1f} | "
              f"Arti: {max_articulation:.3f}/{demo_max_arti:.3f} ({articulation_ratio*100:.1f}%) | "
              f"Success: {task_success}")
    
    return {
        'avg_error': float(avg_error),
        'max_error': float(max_error),
        'error_left': float(avg_error_left),
        'error_right': float(avg_error_right),
        'max_force': float(max_force),
        'max_force_left': float(max_force_left),
        'max_force_right': float(max_force_right),
        'force_saturated': force_saturated,
        'max_articulation': float(max_articulation),
        'final_articulation': float(final_articulation),
        'demo_max_articulation': float(demo_max_arti),
        'articulation_ratio': float(articulation_ratio),
        'task_success': bool(task_success),
        'object_fell': bool(object_fell),
        'min_height': float(min_height),
    }


def setup_scene(args):
    """Setup scene once, return scene, hands, obj, and trajectories."""
    hand_name = args.hand if "hand" in args.hand else f"{args.hand}_hand"
    device = torch.device("cuda")
    
    # Load retargeted data
    fname = f"{RETARGET_DIR}/{hand_name}/{args.subject}/{args.obj}_use_{args.use_clip}_vector_para.pt"
    data = torch.load(fname, weights_only=False)
    retarget_data = data['retargeter_results']
    demo_data = data['demo_data']
    
    # Calculate number of steps
    total_frames = len(retarget_data['left']['hand_qpos'])
    available_steps = total_frames - args.start_frame
    num_steps = min(available_steps, args.max_steps) if args.max_steps > 0 else available_steps
    
    # Initialize Genesis ONCE
    gs.init(backend=gs.gpu, logging_level='warning')
    
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1/60, substeps=2, gravity=(0, 0, -9.81)),
        show_viewer=False,
        use_visualizer=False,
        show_FPS=False,
    )
    
    # Add ground and platform
    plane = scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))
    CARDBOARD_POS = (0, -0.08, 0.90)
    cardbox_size = (0.2, 0.2, 0.1)
    cardbox_surface = gs.surfaces.Rough(roughness=0.1, color=(167/255, 134/255, 103/255, 1.0))
    cardboard_box = scene.add_entity(
        gs.morphs.Box(pos=CARDBOARD_POS, size=cardbox_size, fixed=True, visualization=True),
        surface=cardbox_surface,
    )
    
    # Add hands
    hands = {}
    for side in ['left', 'right']:
        cfg = get_default_robot_cfg(name=hand_name, side=side)
        cfg['action_mode'] = 'absolute'
        hands[side] = BaseRobot(cfg, scene=scene, num_envs=1, device=device, 
                                retarget_data=dict(), is_eval=False, visualize_contact=False)
    
    # Add object
    obj_cfg = get_arctic_object_cfg(name=args.obj, convexify=False)
    obj_cfg['fixed'] = False
    obj_cfg['collect_data'] = False
    
    demo_data_obj = get_demo_data(
        obj_name=args.obj, hand_name=hand_name, frame_start=0, frame_end=num_steps,
        use_clip=args.use_clip, subject_name=args.subject,
    )
    
    obj = ArticulatedObject(obj_cfg, device=device, scene=scene, num_envs=1, 
                           demo_data=demo_data_obj, visualize_contact=False)
    
    # Build scene
    scene.build(n_envs=1)
    
    # Post-build setup
    for hand in hands.values():
        hand.post_scene_build_setup()
    obj.post_scene_build_setup()
    
    # Prepare trajectories
    obj_pos_init = torch.tensor(demo_data['obj_pos'][args.start_frame:args.start_frame+1], device=device)
    obj_quat_init = torch.tensor(demo_data['obj_quat'][args.start_frame:args.start_frame+1], device=device)
    obj_arti_init = torch.tensor(demo_data['obj_arti'][args.start_frame:args.start_frame+1], device=device)[:, None]
    
    hand_qposes = {
        side: torch.tensor(retarget_data[side]['hand_qpos'][args.start_frame:args.start_frame+num_steps], device=device)
        for side in ['left', 'right']
    }
    
    return scene, hands, obj, hand_qposes, num_steps, obj_pos_init, obj_quat_init, obj_arti_init


def main(args):
    print("="*80)
    print("PD Gain Auto-Tuner for Replay")
    print("="*80)
    print(f"Hand: {args.hand}")
    print(f"Object: {args.obj}")
    print(f"Subject: {args.subject}, Clip: {args.use_clip}")
    print(f"Steps: {args.max_steps}, Start frame: {args.start_frame}")
    print()
    
    # Setup scene once
    print("Setting up scene...")
    scene, hands, obj, hand_qposes, num_steps, obj_pos_init, obj_quat_init, obj_arti_init = setup_scene(args)
    print("Scene ready!")
    print()
    
    # Define parameter search space
    if args.coarse:
        print("Running COARSE sweep (small to large gains)...")
        finger_kps = [40, 80, 120, 180, 240]
        wrist_rot_kps = [60, 100, 140, 180, 240]
        wrist_trans_kps = [200, 350, 500, 700, 1000]
        force_ranges = [100]  # Fixed - never saturates
    elif args.ultra:
        print("Running ULTRA HIGH GAIN sweep (testing if stiffness causes drops)...")
        finger_kps = [200, 300, 400, 500]
        wrist_rot_kps = [200, 280, 360]
        wrist_trans_kps = [800, 1100, 1400]
        force_ranges = [150]  # Fixed - higher for safety with high gains
    elif args.fine:
        print("Running FINE sweep...")
        finger_kps = np.linspace(args.kp_range[0], args.kp_range[1], 8)
        wrist_rot_kps = np.linspace(args.wrist_rot_kp_range[0], args.wrist_rot_kp_range[1], 6)
        wrist_trans_kps = np.linspace(args.wrist_trans_kp_range[0], args.wrist_trans_kp_range[1], 6)
        force_ranges = [args.force_range]  # Use specified value
    else:
        # Just test current best or single point
        print("Testing single configuration...")
        finger_kps = [args.kp_range[0]]
        wrist_rot_kps = [args.wrist_rot_kp_range[0]]
        wrist_trans_kps = [args.wrist_trans_kp_range[0]]
        force_ranges = [args.force_range]
    
    results = []
    best_result = None
    best_score = float('inf')
    
    total_tests = len(finger_kps) * len(wrist_rot_kps) * len(wrist_trans_kps) * len(force_ranges)
    print(f"Total configurations to test: {total_tests}")
    print()
    
    pbar = tqdm(total=total_tests, desc="Tuning")
    
    for finger_kp in finger_kps:
        for wrist_rot_kp in wrist_rot_kps:
            for wrist_trans_kp in wrist_trans_kps:
                for force_range in force_ranges:
                    # Calculate kv using critical damping rule: kv ≈ sqrt(kp) * 0.6
                    finger_kv = np.sqrt(finger_kp) * 0.6
                    wrist_rot_kv = np.sqrt(wrist_rot_kp) * 0.6
                    wrist_trans_kv = np.sqrt(wrist_trans_kp) * 0.6
                    
                    pbar.set_description(
                        f"f_kp={finger_kp:.0f}, wr_kp={wrist_rot_kp:.0f}, wt_kp={wrist_trans_kp:.0f}, fr={force_range:.0f}"
                    )
                    
                    try:
                        # Reset object state before each test
                        obj.set_object_state(root_pos=obj_pos_init, root_quat=obj_quat_init, 
                                            joint_qpos=obj_arti_init, env_idxs=[0])
                        
                        result = run_replay_with_gains(
                            args, finger_kp, finger_kv, wrist_rot_kp, wrist_rot_kv,
                            wrist_trans_kp, wrist_trans_kv, force_range,
                            scene, hands, obj, hand_qposes, num_steps,
                            verbose=args.verbose
                        )
                        
                        result['finger_kp'] = float(finger_kp)
                        result['finger_kv'] = float(finger_kv)
                        result['wrist_rot_kp'] = float(wrist_rot_kp)
                        result['wrist_rot_kv'] = float(wrist_rot_kv)
                        result['wrist_trans_kp'] = float(wrist_trans_kp)
                        result['wrist_trans_kv'] = float(wrist_trans_kv)
                        result['force_range'] = float(force_range)
                        
                        # Compute score: PRIORITIZE TASK SUCCESS over control error!
                        # Good gains = Box opens (high articulation) + doesn't fall + tight tracking
                        # 
                        # Scoring formula:
                        #   Base: control_error (0.05-0.30)
                        #   +1.0 if task fails (weighted by how far from success)
                        #   +2.0 if object falls
                        #   +0.1 if force saturated
                        #
                        # This means:
                        #   - Task success matters 10x more than control error
                        #   - Better to have 0.20 error with success than 0.10 error with failure
                        score = result['avg_error']
                        
                        if not result['task_success']:
                            # Failed task: big penalty based on how far from success
                            score += 1.0 * (1.0 - result['articulation_ratio'])  # Penalty for incomplete articulation
                        
                        if result['object_fell']:
                            score += 2.0  # Huge penalty for dropping object
                        
                        if result['force_saturated']:
                            score += 0.1  # Small penalty for saturation
                        
                        result['score'] = float(score)
                        
                        results.append(result)
                        
                        if score < best_score:
                            best_score = score
                            best_result = result
                        
                    except Exception as e:
                        print(f"\nError with config: {e}")
                        continue
                    
                    pbar.update(1)
    
    pbar.close()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Sort results by score
    results.sort(key=lambda x: x['score'])
    
    # Show top 5
    print("\nTop 5 configurations:")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {r['score']:.4f} | Task Success: {r['task_success']} | Arti: {r['articulation_ratio']*100:.1f}%")
        print(f"   finger_kp={r['finger_kp']:.1f}, finger_kv={r['finger_kv']:.2f}")
        print(f"   wrist_rot_kp={r['wrist_rot_kp']:.1f}, wrist_rot_kv={r['wrist_rot_kv']:.2f}")
        print(f"   wrist_trans_kp={r['wrist_trans_kp']:.1f}, wrist_trans_kv={r['wrist_trans_kv']:.2f}")
        print(f"   force_range={r['force_range']:.1f}")
        print(f"   Avg Error: {r['avg_error']:.4f} (L:{r['error_left']:.4f}, R:{r['error_right']:.4f})")
        print(f"   Max Force: {r['max_force']:.1f}, Fell: {r['object_fell']}")
    
    # Best result
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    r = best_result
    print(f"Score: {r['score']:.4f}")
    print(f"Task Success: {r['task_success']}")
    print(f"Articulation: {r['max_articulation']:.3f} / {r['demo_max_articulation']:.3f} ({r['articulation_ratio']*100:.1f}%)")
    print(f"Object Fell: {r['object_fell']} (min height: {r['min_height']:.3f}m)")
    print(f"Average Error: {r['avg_error']:.4f} (Left: {r['error_left']:.4f}, Right: {r['error_right']:.4f})")
    print(f"Max Force: {r['max_force']:.1f} / {r['force_range']:.1f}")
    print()
    print("Recommended gain values for orca.py:")
    print(f"finger_kp = {r['finger_kp']:.1f}")
    print(f"finger_kv = {r['finger_kv']:.1f}")
    print(f"wrist_rot_kp = {r['wrist_rot_kp']:.1f}")
    print(f"wrist_rot_kv = {r['wrist_rot_kv']:.1f}")
    print(f"wrist_trans_kp = {r['wrist_trans_kp']:.1f}")
    print(f"wrist_trans_kv = {r['wrist_trans_kv']:.1f}")
    print(f"force_range = {r['force_range']:.1f}")
    
    # Save results
    output_dir = "outputs/tune_gains"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/tune_results_{args.hand}_{args.obj}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'best': best_result,
            'top_5': results[:5],
            'all_results': results,
            'args': vars(args),
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    if args.update_config:
        print("\nWARNING: --update_config not implemented yet.")
        print("Please manually update dexmachina/envs/hand_cfgs/orca.py with the values above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='box')
    parser.add_argument('--hand', type=str, default='orca_hand')
    parser.add_argument('--subject', type=str, default='s01')
    parser.add_argument('--use_clip', type=str, default='01')
    parser.add_argument('--max_steps', type=int, default=100, 
                       help='Number of steps to replay (100 is fast, 200+ is more accurate)')
    parser.add_argument('--start_frame', type=int, default=0)
    
    # Sweep modes
    parser.add_argument('--coarse', action='store_true',
                       help='Run coarse sweep over wide range')
    parser.add_argument('--ultra', action='store_true',
                       help='Run ultra-high gain sweep (for very stiff tracking)')
    parser.add_argument('--fine', action='store_true',
                       help='Run fine sweep over narrow range')
    parser.add_argument('--best_only', action='store_true',
                       help='Test only the current best configuration')
    
    # Parameter ranges
    parser.add_argument('--kp_range', type=float, nargs=2, default=[120, 180],
                       help='finger kp range for fine sweep [min, max]')
    parser.add_argument('--wrist_rot_kp_range', type=float, nargs=2, default=[100, 160],
                       help='wrist rotation kp range [min, max]')
    parser.add_argument('--wrist_trans_kp_range', type=float, nargs=2, default=[350, 500],
                       help='wrist translation kp range [min, max]')
    parser.add_argument('--kv_range', type=float, nargs=2, default=[4, 8],
                       help='kv range for fine sweep [min, max] (or auto from kp)')
    parser.add_argument('--force_range', type=float, default=100.0,
                       help='Force range (fixed, not swept - never saturates anyway)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--update_config', action='store_true',
                       help='Automatically update config file with best gains')
    
    args = parser.parse_args()
    main(args)
