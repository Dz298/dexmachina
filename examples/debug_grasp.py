import torch
import numpy as np
import random
from copy import deepcopy
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import get_default_robot_cfg
from dexmachina.envs.demo_data import load_genesis_retarget_data
from dexmachina.envs.base_env import OBJ_DEFAULT_POS
from dexmachina.envs.rewards import get_reward_cfg
from dexmachina.envs.object import get_arctic_object_cfg
import genesis as gs

def build_env(enforce=True):
    env_cfg = get_env_cfg()
    env_cfg.update(
        dict(
            num_envs=1,
            is_eval=True,
            record_video=True,  # so front camera is built
            rand_init_ratio=1.0,
            enforce_valid_rand_init_grasp=enforce,
            valid_grasp_contact_thresh=0.01,
            valid_grasp_opt_samples=16,
            episode_length=130,
        )
    )
    env_cfg['scene_kwargs']['use_visualizer'] = True  
    env_cfg['scene_kwargs']['show_viewer'] = False
    env_cfg['record_video'] = True 
    print(f"Setting render resolution to 512") 
    env_cfg['camera_kwargs']['front'] = dict(
        res=(512, 512),
        fov=40,
        pos=(0.5, -1.5, 1.2),
        lookat=(0.0, -1.58, 2.0),
    ) 
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
    return BaseEnv(
        env_cfg=env_cfg,
        robot_cfgs=robot_cfgs,
        object_cfgs=object_cfgs,
        reward_cfg=get_reward_cfg(),
        demo_data=demo_data,
        retarget_data=retarget,
        render_figure=False,
    )

def snapshot(env, tag, num_cycles=5, steps_per_cycle=60, seed=0):
    """Record a short rollout with repeated resets similar to eval_rl_games."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = env.device
    zero_actions = torch.zeros((env.num_envs, env.action_dim), device=device)
    env.max_video_frames = max(env.max_video_frames, num_cycles * steps_per_cycle * 2)
    env.start_recording()
    viewer = getattr(env.scene, "viewer", None)
    if viewer is not None and getattr(viewer, "camera", None) is not None:
        viewer.camera.follow(env.robots["left"].entity)

    for cycle in range(num_cycles):
        env.reset()
        for _ in range(steps_per_cycle):
            env.step(zero_actions)

    frames = env.get_recorded_frames(wait_for_max=False)
    if frames:
        from moviepy.editor import ImageSequenceClip
        output_path = f"/tmp/{tag}_reset.mp4"
        clip = ImageSequenceClip(frames, fps=int(1 / env.dt))
        clip.write_videofile(output_path)
        print(f"{tag}: saved video to {output_path}")

if __name__ == "__main__":
    seed = 42
    gs.init(backend=gs.gpu, logging_level='warning')
    raw_env = build_env(enforce=False)
    opt_env = build_env(enforce=True)
    snapshot(raw_env, "before_opt", seed=seed)
    snapshot(opt_env, "after_opt", seed=seed + 1)
