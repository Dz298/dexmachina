import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
from datetime import datetime 

from dexmachina.asset_utils import get_rl_config_path 
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
from dexmachina.rl.latent_world_model import load_world_model_for_eval, WorldModelTrainer

from collections import defaultdict
import shutil
import moviepy
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
 

def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()


def get_eval_hands(uenv):
    active_sides = list(getattr(uenv, "active_sides", []) or uenv.robots.keys())
    hands = {side: uenv.robots[side] for side in active_sides if side in uenv.robots}
    if not hands:
        raise KeyError("No active hands found in evaluation environment")
    return hands

def eval_one_episode(env, agent, obj_state_tensor, print_rew=False, record_video=False, show_reference=False):
    obs = env.reset() 
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    uenv = env.unwrapped
    ep_len = uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    if record_video:
        uenv.start_recording() 
        uenv.max_video_frames = int(uenv.max_episode_length)
        max_frames = uenv.max_video_frames   
    
    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if obj.actuated:
            print("Setting eval time obj gains to 0.0")
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    assert obj is not None, "No object found in the environment"
    eval_hands = get_eval_hands(uenv)
    joint_targets = {side: hand.residual_qpos for side, hand in eval_hands.items()}
    
    eval_data = defaultdict(list)
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True) 
            demo_state = obj_state_tensor[env_step]
                    
            if show_reference: # visualize the demo traj and set zero action
                ref_env_idx = 1 if num_envs > 1 else 0
                obj.set_object_state(
                    root_pos=demo_state[:3][None],
                    root_quat=demo_state[3:7][None],
                    joint_qpos=demo_state[7][None],
                    env_idxs=torch.tensor([ref_env_idx], dtype=torch.int32, device=device),
                )
                actions[ref_env_idx, :] = -1.0 
                for side, robot in eval_hands.items():
                    joints = joint_targets.get(side)
                    if joints is None:
                        continue
                    robot.set_joint_position(
                        joint_targets=joints[env_step][None],
                        env_idxs=[ref_env_idx],
                    ) 
            eval_data["action"].append(actions.detach().cpu().numpy())
            obs, rew, dones, infos = env.step(actions) 
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            # print(f"Step {env_step}: Obj pos: {obj_pos.cpu().numpy()}")
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())
            for side in sorted(eval_hands.keys()):
                eval_data[f"hand_dof_{side}"].append(
                    eval_hands[side].dof_pos.detach().cpu().numpy()
                )

            rew_dict = uenv.rew_dict
            for key in ['pos_dist', 'rot_dist', 'arti_dist']:
                eval_data[key].append(rew_dict[key].cpu().numpy())
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0 
        
        if print_rew:
            print(f"Step {env_step}: Reward: {rew.cpu().numpy()}")
            # perform operations for terminated episodes 
            rew_dict = uenv.rew_dict
            for k, v in rew_dict.items():
                if 'con' in k:
                    print(f"Step {env_step}: {k}: {v.cpu().numpy()}")
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    frames = uenv.get_recorded_frames()
    return frames, eval_data


def main():

    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--eval_episodes', '-ne', type=int, default=1)
    parser.add_argument('--print_rew', '-pr', action='store_true')
    parser.add_argument('--show_reference', '-ref', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--output_render', '-or', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--render_dir', '-out', type=str, default="rendered") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4") # if not provided, save in the same folder as the checkpoint
    
    args = parser.parse_args()

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_eval"))
    os.makedirs(ckpt_data_folder, exist_ok=True) 

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = os.path.join(args.render_dir, run_name)
        print('Saving video to a different folder')
        video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + args.video_fname)
        os.makedirs(render_dir, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    # load to pkl
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    
    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"

    # World model: if checkpoint was trained with WM, load it for eval
    nn_dir = os.path.join(ckpt_path, "nn")
    world_model_path = os.path.join(nn_dir, "world_model.pt")
    load_world_model = os.path.exists(world_model_path)
    wm_latent_dim = None
    if load_world_model:
        # Infer latent_dim from checkpoint before creating env
        _ckpt = torch.load(world_model_path, map_location="cpu")
        _state = _ckpt["model_state_dict"]
        _enc_w = [k for k in _state if k.startswith("encoder.") and k.endswith(".weight")]
        wm_latent_dim = int(_state[max(_enc_w, key=lambda x: int(x.split(".")[1]))].shape[0])
        env_kwargs["env_cfg"]["use_latent_world_model"] = True
        env_kwargs["env_cfg"]["wm_latent_dim"] = wm_latent_dim
        print(f"[INFO] Loading world model from {world_model_path} (latent_dim={wm_latent_dim})")
    
    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 

    if args.show_markers:
        hand_sides = list(env_kwargs["robot_cfgs"].keys())
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=hand_sides,
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to 512") 
        # env_kwargs['camera_res'] = (2048, 2048)
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        ) 
    for name, cfg in env_kwargs['object_cfgs'].items():
        print("Setting eval time obj gains to 0.0")
        cfg['actuated'] = False

    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    print("Removing any curriculum config during eval")
    env_kwargs.pop("curriculum_cfg") 
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    env = BaseEnv(
         **env_kwargs
    )
    demo_data = env_kwargs['demo_data']
    obj_state_tensor = gather_object_state_tensor(demo_data)

    # Load world model for eval if this run was trained with WM
    world_model_trainer = None
    if load_world_model and wm_latent_dim is not None:
        obs_dim_without_latent = env.obs_dim - wm_latent_dim
        world_model, _ = load_world_model_for_eval(
            world_model_path,
            obs_dim=obs_dim_without_latent,
            action_dim=env.action_dim,
            device=device,
        )
        world_model_trainer = WorldModelTrainer(
            world_model=world_model,
            num_envs=env.num_envs,
            device=device,
            eval_only=True,
        )
        world_model_trainer.world_model.eval()

    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, world_model_trainer=world_model_trainer)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    uenv = env.unwrapped 
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs

    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    resume_path = os.path.abspath(args.checkpoint) 
    agent.restore(resume_path)
    agent.reset()

    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env, agent, obj_state_tensor, args.print_rew, args.record_video, args.show_reference
            )
        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npy")
        np.save(ckpt_eval_fname, eval_data)
        print(f"Saved eval data to {ckpt_eval_fname}")
        # try loading the data
        # eval_data = np.load(ckpt_eval_fname, allow_pickle=True).item() 
        if args.record_video: 
            # save video with moviepy
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(frames, fps=int(1/uenv.dt/2))
            clip.write_videofile(video_fname)
            print(f"Saved video to {video_fname}")
    
    print("Done evaluating")


if __name__ == '__main__':
    main()
