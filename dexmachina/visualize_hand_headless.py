#!/usr/bin/env python3
"""
Headless visualizer for retargeted hand motion.
Renders to video/images without requiring X11 display.
Perfect for SSH/remote environments.
"""
import os
import argparse
import torch
import numpy as np
import genesis as gs
from pathlib import Path

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.demo_data import get_demo_data


RETARGET_DIR = get_asset_path("retargeted")


def create_headless_scene(num_envs, robot_cfgs, object_cfg, demo_data, device=torch.device("cuda")):
    """Create a headless scene that can render without display"""
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0, -9.81),
        ),
        vis_options=gs.options.VisOptions(
            n_rendered_envs=1,  # Only render one env for video
            show_world_frame=False,
            visualize_contact=True,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=1/60,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        use_visualizer=True,  # Need visualizer for rendering
        show_viewer=False,     # Don't show window (headless)
        show_FPS=False,
    )
    
    scene = gs.Scene(**scene_cfg)
    
    # Add ground plane
    plane = scene.add_entity(
        gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True)
    )
    
    # Create robots
    robots = dict()
    for side, cfg in robot_cfgs.items():
        robots[side] = BaseRobot(
            robot_cfg=cfg,
            scene=scene,
            num_envs=num_envs,
            device=device,
            retarget_data=dict(),
            visualize_contact=False,
            is_eval=False,
        )
    
    # Create object
    obj = ArticulatedObject(
        object_cfg,
        device=device,
        scene=scene,
        num_envs=num_envs,
        demo_data=demo_data,
        visualize_contact=True,
    )
    
    # Add camera BEFORE building scene
    # Position camera to see hands and object properly
    # Hands are around height z=1.0-1.3, x=±0.6
    camera = scene.add_camera(
        pos=(1.5, -1.5, 1.3),  # Further back and at hand height
        lookat=(0, 0.4, 1.15),  # Look at center between hands
        res=(1280, 720),
        fov=65,
        GUI=False
    )
    
    # Build scene
    scene.build(n_envs=num_envs, env_spacing=(1.5, 1.5), n_envs_per_row=8)
    
    # Post-build setup
    for robot in robots.values():
        robot.post_scene_build_setup()
    obj.post_scene_build_setup()
    
    return scene, robots, obj, camera


def main(args):
    hand_name = args.hand if "hand" in args.hand else f"{args.hand}_hand"
    
    # Load retargeted data
    fname = args.load_file
    if not fname:
        fname = f"{RETARGET_DIR}/{hand_name}/{args.subject}/{args.obj}_use_{args.use_clip}_vector_para.pt"
    
    if not os.path.exists(fname):
        print(f"Error: File not found: {fname}")
        print(f"\nAvailable files in {RETARGET_DIR}/{hand_name}/{args.subject}/:")
        if os.path.exists(f"{RETARGET_DIR}/{hand_name}/{args.subject}/"):
            for f in os.listdir(f"{RETARGET_DIR}/{hand_name}/{args.subject}/"):
                print(f"  - {f}")
        return
    
    print(f"Loading retargeted data from: {fname}")
    loaded_data = torch.load(fname, weights_only=False)
    retarget_data = loaded_data['retargeter_results']
    demo_data = loaded_data['demo_data']
    
    # Get number of steps
    num_steps = len(retarget_data['left']['hand_qpos'])
    print(f"Loaded {num_steps} steps of retargeted hand motion")
    
    # Print wrist position info for debugging
    print(f"\nHand positions (first frame):")
    for side in ['left', 'right']:
        wrist_pos = retarget_data[side]['wrist_qpos'][0][:3]
        print(f"  {side} wrist: x={wrist_pos[0]:.3f}, y={wrist_pos[1]:.3f}, z={wrist_pos[2]:.3f}")
    
    # Determine which steps to render
    if args.num_frames > 0:
        # Subsample to specified number of frames
        step_indices = np.linspace(0, num_steps - 1, args.num_frames, dtype=int)
        print(f"Rendering {args.num_frames} frames (subsampled from {num_steps})")
    else:
        # Render all frames
        step_indices = range(0, num_steps, args.frame_skip)
        print(f"Rendering {len(list(step_indices))} frames (every {args.frame_skip} frame)")
    
    # Setup robot configs
    num_envs = 1  # Only need 1 env for video rendering
    robot_cfgs = dict()
    sides = ['left', 'right'] if args.both_hands else [args.hand_side]
    
    for side in sides:
        cfg = get_default_robot_cfg(name=hand_name, side=side)
        cfg['action_mode'] = 'absolute'
        cfg['collect_data'] = False
        robot_cfgs[side] = cfg
    
    # Setup object config
    obj_cfg = get_arctic_object_cfg(name=args.obj, convexify=False)
    obj_cfg['fixed'] = True  # Keep object fixed in place
    obj_cfg['collect_data'] = False
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene
    device = torch.device("cuda")
    scene, hands, obj, camera = create_headless_scene(
        num_envs=num_envs,
        robot_cfgs=robot_cfgs,
        object_cfg=obj_cfg,
        demo_data=demo_data,
        device=device,
    )
    
    # Prepare object state tensors
    obj_pos = torch.tensor(demo_data['obj_pos'], device=device)
    obj_quat = torch.tensor(demo_data['obj_quat'], device=device)
    obj_arti = torch.tensor(demo_data['obj_arti'], device=device)[:, None]
    
    print("\nRendering frames...")
    render_frames = []
    
    import cv2
    from tqdm import tqdm
    
    # Set initial hand positions first
    for side in sides:
        hand = hands[side]
        hand_qpos = retarget_data[side]['hand_qpos'][0]
        hand_qpos_tensor = torch.tensor(hand_qpos, device=device).unsqueeze(0)
        hand.set_joint_position(hand_qpos_tensor, env_idxs=[0])
    
    # Set initial object state
    obj.set_object_state(
        root_pos=obj_pos[0:1],
        root_quat=obj_quat[0:1],
        joint_qpos=obj_arti[0:1],
        env_idxs=[0]
    )
    
    # Step once to initialize
    scene.step()
    
    for frame_idx, step_idx in enumerate(tqdm(step_indices, desc="Rendering")):
        # Update hand states
        for side in sides:
            hand = hands[side]
            hand_qpos = retarget_data[side]['hand_qpos'][step_idx]
            hand_qpos_tensor = torch.tensor(hand_qpos, device=device).unsqueeze(0)
            hand.set_joint_position(hand_qpos_tensor, env_idxs=[0])
        
        # Update object state
        obj.set_object_state(
            root_pos=obj_pos[step_idx:step_idx+1],
            root_quat=obj_quat[step_idx:step_idx+1],
            joint_qpos=obj_arti[step_idx:step_idx+1],
            env_idxs=[0]
        )
        
        # Step simulation to update visuals
        scene.step()
        
        # Render frame
        img = camera.render()[0]  # Get first element (image)
        render_frames.append(img)
        
        # Save individual frame if requested
        if args.save_frames:
            os.makedirs(args.output_dir, exist_ok=True)
            frame_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"\nRendered {len(render_frames)} frames")
    
    # Create video
    if args.create_video and len(render_frames) > 0:
        output_video = os.path.join(args.output_dir, args.output_name)
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nCreating video: {output_video}")
        
        # Use OpenCV to write video (more reliable than moviepy for headless)
        height, width = render_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, args.fps, (width, height))
        
        for frame in tqdm(render_frames, desc="Writing video"):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        print(f"✓ Video saved to: {output_video}")
        print(f"  Duration: {len(render_frames) / args.fps:.2f}s")
        print(f"  Frames: {len(render_frames)}")
        print(f"  FPS: {args.fps}")
        print(f"  Resolution: {width}x{height}")
    
    if args.save_frames:
        print(f"\n✓ Individual frames saved to: {args.output_dir}/")
        print(f"  Total frames: {len(render_frames)}")
    
    print("\n✓ Rendering complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Headless retargeted hand visualizer (for SSH/remote)')
    
    # Input options
    parser.add_argument('--load_file', '-f', type=str, default=None,
                        help='Path to retargeted data file')
    parser.add_argument('--obj', type=str, default='box',
                        help='Object name')
    parser.add_argument('--use_clip', type=str, default='01',
                        help='Clip number')
    parser.add_argument('--subject', type=str, default='s01',
                        help='Subject name')
    parser.add_argument('--hand', type=str, default='orca_hand',
                        help='Hand name')
    parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right'],
                        help='Which hand to show (if --both_hands is False)')
    parser.add_argument('--both_hands', '-b', action='store_true', default=True,
                        help='Show both hands')
    
    # Rendering options
    parser.add_argument('--num_frames', '-n', type=int, default=120,
                        help='Number of frames to render (0 = all frames)')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Frame skip (used when num_frames=0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    
    # Output options
    parser.add_argument('--output_dir', '-o', type=str, default='visualization_output',
                        help='Output directory for frames and video')
    parser.add_argument('--output_name', type=str, default='retargeted_hand.mp4',
                        help='Output video filename')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save individual frames as images')
    parser.add_argument('--create_video', action='store_true', default=True,
                        help='Create video from rendered frames')
    
    args = parser.parse_args()
    main(args)
