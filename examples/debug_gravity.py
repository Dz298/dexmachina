import genesis as gs
import torch
import numpy as np

from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import get_default_robot_cfg
from dexmachina.envs.object import get_arctic_object_cfg
from dexmachina.envs.rewards import get_reward_cfg
from dexmachina.envs.demo_data import load_genesis_retarget_data
from dexmachina.envs.curriculum import get_curriculum_cfg

def build_env():
    env_cfg = get_env_cfg()
    env_cfg.update(
        dict(
            num_envs=1,
            is_eval=True,
            record_video=True,
            episode_length=200,
        )
    )
    env_cfg['scene_kwargs']['use_visualizer'] = True  
    env_cfg['scene_kwargs']['show_viewer'] = False
    
    # Load demo data
    demo_data, retarget = load_genesis_retarget_data(
        obj_name="box", hand_name="orca_hand", frame_start=100, frame_end=230, save_name="para"
    )
    
    robot_cfgs = {
        "left": get_default_robot_cfg(side="left", name="orca_hand"),
        "right": get_default_robot_cfg(side="right", name="orca_hand"),
    }
    object_cfgs = {
        "box": get_arctic_object_cfg(name="box"),
    }
    object_cfgs["box"]["actuated"] = True

    
    # Fast curriculum to test gravity decay quickly
    curr_cfg = get_curriculum_cfg(
        dict(
            wait_epochs=0,
            interval=1,
            first_stop_iter=10,
            second_stop_iter=20,
            first_ratio=0.5,
            schedule="fixed",
            fixed_mode="lin",
            gain_mode="all"
        )
    )

    return BaseEnv(
        env_cfg=env_cfg,
        robot_cfgs=robot_cfgs,
        object_cfgs=object_cfgs,
        reward_cfg=get_reward_cfg(),
        demo_data=demo_data,
        retarget_data=retarget,
        curriculum_cfg=curr_cfg,
        render_figure=False,
    )

if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level='warning')
    env = build_env()
    device = env.device
    zero_actions = torch.zeros((env.num_envs, env.action_dim), device=device)
    
    # We will step for 30 epochs
    num_epochs = 30
    steps_per_epoch = 10 
    
    env.max_video_frames = num_epochs * steps_per_epoch
    env.start_recording()
    
    for epoch in range(num_epochs):
        env.reset()
        zero_gains, gains_decayed, reason = env.curriculum.set_curriculum(epoch)
        gains = env.curriculum.get_current_gains()
        print(f"Epoch {epoch}: gravity gain = {gains.get('gravity', 0.0):.4f}")
        
        # Step through epoch
        for _ in range(steps_per_epoch):
            # Object should fall faster at later epochs since the counter-force decays
            env.step(zero_actions)
    
    frames = env.get_recorded_frames(wait_for_max=False)
    if frames:
        from moviepy.editor import ImageSequenceClip
        output_path = f"/tmp/gravity_test.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(output_path)
        print(f"Saved video to {output_path}")

