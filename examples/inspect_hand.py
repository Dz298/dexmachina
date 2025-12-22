import os
import argparse
import copy
import numpy as np
import torch 
from dexmachina.envs import BaseRobot, get_default_robot_cfg 
import genesis as gs
import argparse

"""
Creates a basic scene with a robot hand and a plane, and steps through random actions.
""" 

def main(args):
    num_envs = args.num_envs

    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0, -9.81) if not args.zero_gravity else (0, 0, 0), 
        ),
        show_viewer=args.vis,
        use_visualizer=True,
        show_FPS=False, 
    )
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')
    robot_cfg = get_default_robot_cfg(
        name=args.hand,
        side='left'
        )
        
    robot_cfg['action_mode'] = "absolute"
    robot = BaseRobot(
        robot_cfg, device=device, scene=scene, num_envs=num_envs
        )
    plane = scene.add_entity(
        gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))

    scene.build(
        n_envs=num_envs, 
        env_spacing=(2.0, 2.0)
        )
    robot.post_scene_build_setup()
    robot.reset_idx() 
    
    # Debug: print init_qpos to verify default_qpos is loaded
    print("\n=== DEBUG: robot.init_qpos (first 6 = forearm joints) ===")
    forearm_labels = ['tx', 'ty', 'tz', 'roll', 'pitch', 'yaw']
    for i, label in enumerate(forearm_labels):
        print(f"  {label}: {robot.init_qpos[0, i].item():.4f}")
    
    # IMPORTANT: Need to step the scene for physics to update positions!
    scene.step()
    
    # Now check actual joint positions after stepping
    print("\n=== DEBUG: Actual joint positions after scene.step() ===")
    actual_pos = robot.entity.get_dofs_position()[0]
    for i, label in enumerate(forearm_labels):
        print(f"  {label}: {actual_pos[i].item():.4f}")
    
    print("\nHand should now be at the correct position. Use breakpoint to inspect.")
    breakpoint()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--hand', type=str, default='allegro_hand')
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--zero_gravity', action='store_true')
    args = parser.parse_args()
    main(args)

