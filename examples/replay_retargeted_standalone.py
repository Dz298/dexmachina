#!/usr/bin/env python3
"""
Replay retargeted motion to verify if it achieves the task.
Standalone script that doesn't depend on complex constructor functions.
Supports both live visualization and headless video recording.
Can add randomization to show that exact replay fails with different initial conditions.
"""

import os
import argparse
import re
import torch
import numpy as np
import genesis as gs
import cv2
from tqdm import tqdm
import scipy.spatial.transform

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg, get_hand_specific_cfg
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.demo_data import get_demo_data

RETARGET_DIR = get_asset_path("retargeted")


def main(args):
    hand_name = args.hand if "hand" in args.hand else f"{args.hand}_hand"
    
    # Load retargeted data
    fname = f"{RETARGET_DIR}/{hand_name}/{args.subject}/{args.obj}_use_{args.use_clip}_vector_para.pt"
    print(f"Loading: {fname}")
    data = torch.load(fname, weights_only=False)
    retarget_data = data['retargeter_results']
    demo_data = data['demo_data']
    
    # Calculate number of steps from start_frame
    total_frames = len(retarget_data['left']['hand_qpos'])
    available_steps = total_frames - args.start_frame
    if args.max_steps > 0:
        num_steps = min(available_steps, args.max_steps)
    else:
        num_steps = available_steps
    
    print(f"\nReplay configuration:")
    print(f"  Frame range: {args.start_frame} to {args.start_frame + num_steps - 1}")
    print(f"  Number of steps: {num_steps}")
    if args.randomize:
        print(f"  Randomization: ENABLED")
        print(f"    Position noise: ±{args.pos_noise}m")
        print(f"    Rotation noise: ±{args.rot_noise}rad")
    else:
        print(f"  Randomization: DISABLED")
    print()
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1/60, substeps=2, gravity=(0, 0, -9.81)),
        show_viewer=args.vis,
        use_visualizer=True,
        show_FPS=not args.record_video,  # Hide FPS when recording
    )
    
    # Add ground
    plane = scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))
    
    # Add platform (cardboard box) to catch the object if it falls
    CARDBOARD_POS = (0, -0.08, 0.90)
    cardbox_size = (0.2, 0.2, 0.1)
    cardbox_surface = gs.surfaces.Rough(roughness=0.1, color=(167/255, 134/255, 103/255, 1.0))
    cardboard_box = scene.add_entity(
        gs.morphs.Box(
            pos=CARDBOARD_POS,
            size=cardbox_size,
            fixed=True,
            visualization=True,
        ),
        surface=cardbox_surface,
    )
    
    # Add hands
    device = torch.device("cuda")
    hands = {}
    for side in ['left', 'right']:
        cfg = get_default_robot_cfg(name=hand_name, side=side)
        cfg['action_mode'] = 'absolute'
        hands[side] = BaseRobot(cfg, scene=scene, num_envs=1, device=device, retarget_data=dict(), is_eval=False, visualize_contact=False)
    
    # Add object
    obj_cfg = get_arctic_object_cfg(name=args.obj, convexify=False)
    obj_cfg['fixed'] = False  # Let it be dynamic
    obj_cfg['collect_data'] = False
    
    demo_data_obj = get_demo_data(
        obj_name=args.obj,
        hand_name=hand_name,
        frame_start=0,
        frame_end=num_steps,
        use_clip=args.use_clip,
        subject_name=args.subject,
    )
    
    obj = ArticulatedObject(obj_cfg, device=device, scene=scene, num_envs=1, demo_data=demo_data_obj, visualize_contact=False)
    
    # Add camera for recording (before building scene)
    if args.record_video:
        camera = scene.add_camera(
            pos=(1.5, -1.5, 1.3),
            lookat=(0, 0.4, 1.15),
            res=(1280, 720),
            fov=65,
            GUI=False
        )
    
    # Build scene
    scene.build(n_envs=1)
    
    # Post-build setup
    for hand in hands.values():
        hand.post_scene_build_setup()
    obj.post_scene_build_setup()
    
    # Set PD gains from hand config (AFTER scene is built)
    hand_cfg = get_hand_specific_cfg(name=hand_name)
    print("\nSetting PD gains from hand config:")
    for side in ['left', 'right']:
        hand = hands[side]
        actuator_cfgs = hand_cfg[side]['actuators']
        for joint_group, act_cfg in actuator_cfgs.items():
            joint_exprs = act_cfg['joint_exprs']
            kp = act_cfg['kp']
            kv = act_cfg['kv']
            fr = act_cfg['force_range']
            
            # Find joints matching the patterns
            joint_names = []
            joint_idxs = []
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
                
                print(f"  {side} {joint_group}: kp={kp}, kv={kv}, fr={fr} ({len(joint_idxs)} joints)")
    
    # Prepare data - extract from start_frame onwards
    obj_pos = torch.tensor(demo_data['obj_pos'][args.start_frame:args.start_frame+num_steps], device=device)
    obj_quat = torch.tensor(demo_data['obj_quat'][args.start_frame:args.start_frame+num_steps], device=device)
    obj_arti = torch.tensor(demo_data['obj_arti'][args.start_frame:args.start_frame+num_steps], device=device)[:, None]
    
    # Extract initial object state from start_frame
    obj_pos_init = obj_pos[0:1].clone()
    obj_quat_init = obj_quat[0:1].clone()
    obj_arti_init = obj_arti[0:1].clone()
    
    print(f"\nInitial object state at frame {args.start_frame}:")
    print(f"  Position: [{obj_pos_init[0,0]:.4f}, {obj_pos_init[0,1]:.4f}, {obj_pos_init[0,2]:.4f}]")
    print(f"  Quaternion: [{obj_quat_init[0,0]:.4f}, {obj_quat_init[0,1]:.4f}, {obj_quat_init[0,2]:.4f}, {obj_quat_init[0,3]:.4f}]")
    
    # Add randomization to object initial state if requested
    if args.randomize:
        print(f"\nApplying randomization to object state at frame {args.start_frame}...")
        # Random position offset
        pos_offset = torch.randn(3, device=device) * args.pos_noise
        obj_pos_init = obj_pos_init + pos_offset
        
        # Random rotation offset (convert to quaternion)
        import scipy.spatial.transform as st
        rot_offset = np.random.randn(3) * args.rot_noise
        quat_offset = st.Rotation.from_rotvec(rot_offset).as_quat()  # [x, y, z, w]
        obj_quat_init = torch.tensor(quat_offset, device=device, dtype=obj_quat.dtype).unsqueeze(0)
        
        print(f"  Position offset: [{pos_offset[0]:.4f}, {pos_offset[1]:.4f}, {pos_offset[2]:.4f}]")
        print(f"  Rotation offset: [{rot_offset[0]:.4f}, {rot_offset[1]:.4f}, {rot_offset[2]:.4f}] rad")
        print(f"  Object will now be SIMULATED (physics), not kinematically replayed")
    
    # Extract hand trajectories from start_frame onwards  
    hand_qposes = {
        side: torch.tensor(retarget_data[side]['hand_qpos'][args.start_frame:args.start_frame+num_steps], device=device)
        for side in ['left', 'right']
    }
    print()
    
    # Set initial object state (before replay loop)
    obj.set_object_state(
        root_pos=obj_pos_init,
        root_quat=obj_quat_init,
        joint_qpos=obj_arti_init,
        env_idxs=[0]
    )
    
    # Set initial hand positions to match start_frame of retargeted trajectory
    print("Setting initial hand positions to match retargeted trajectory at start frame...")
    for side in ['left', 'right']:
        hand = hands[side]
        init_qpos = hand_qposes[side][0:1]  # First frame of the trajectory
        hand.set_joint_position(init_qpos, env_idxs=[0])
        print(f"  {side} hand initialized to frame {args.start_frame}")
    
    print("Starting replay...")
    print("="*60)
    
    control_errors = {'left': [], 'right': []}
    render_frames = []
    
    step_iter = tqdm(range(num_steps), desc="Replaying") if args.record_video else range(num_steps)
    
    for step in step_iter:
        # Object is simulated (not set) - it responds to hand contacts
        # Only hands follow the exact trajectory
        
        # Control hand positions using PD controller (not kinematic setting)
        for side in ['left', 'right']:
            hand = hands[side]
            target_qpos = hand_qposes[side][step:step+1]
            hand.control_joint_position(target_qpos, env_idxs=[0])
        
        scene.step()
        
        # Render frame if recording
        if args.record_video:
            img = camera.render()[0]
            render_frames.append(img)
        
        # Track control errors
        for side in ['left', 'right']:
            hand = hands[side]
            hand.update_value_buffers()
            err = hand.get_control_errors()
            control_errors[side].append(err.mean().item())
        
        if not args.record_video and (step % 50 == 0 or step == num_steps - 1):
            avg_l = np.mean(control_errors['left'][-50:]) if control_errors['left'] else 0
            avg_r = np.mean(control_errors['right'][-50:]) if control_errors['right'] else 0
            print(f"Step {step:3d}/{num_steps} | Err L:{avg_l:.4f} R:{avg_r:.4f}")
    
    print("="*60)
    print(f"\n✓ Replay complete!")
    print(f"  Avg control error (left):  {np.mean(control_errors['left']):.4f}")
    print(f"  Avg control error (right): {np.mean(control_errors['right']):.4f}")
    
    # Save video
    if args.record_video and len(render_frames) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_video = os.path.join(args.output_dir, args.output_name)
        
        print(f"\nCreating video: {output_video}")
        height, width = render_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, args.fps, (width, height))
        
        for frame in tqdm(render_frames, desc="Writing video"):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        print(f"✓ Video saved to: {output_video}")
        print(f"  Duration: {len(render_frames) / args.fps:.2f}s")
        print(f"  Frames: {len(render_frames)}")
        print(f"  FPS: {args.fps}")
        print(f"  Resolution: {width}x{height}")
    
    print("\nThe visualization shows if retargeting achieves the task!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='box')
    parser.add_argument('--hand', type=str, default='orca_hand')
    parser.add_argument('--subject', type=str, default='s01')
    parser.add_argument('--use_clip', type=str, default='01')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Show live viewer (requires display)')
    parser.add_argument('--record_video', '-r', action='store_true',
                        help='Record video (headless mode)')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/replay',
                        help='Output directory for video')
    parser.add_argument('--output_name', type=str, default='replay_retargeted.mp4',
                        help='Output video filename')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    parser.add_argument('--randomize', action='store_true',
                        help='Add randomization to object initial state')
    parser.add_argument('--pos_noise', type=float, default=0.05,
                        help='Position noise magnitude (meters)')
    parser.add_argument('--rot_noise', type=float, default=0.3,
                        help='Rotation noise magnitude (radians)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start replay from this frame (matches training clip start)')
    args = parser.parse_args()
    main(args)
