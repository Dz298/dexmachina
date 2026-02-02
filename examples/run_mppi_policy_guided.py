#!/usr/bin/env python3
"""Run MPPI residual controller with Genesis (DexMachina), guided by a trained RL policy.

This script implements Model Predictive Path Integral (MPPI) control as a residual 
controller on top of a learned policy. MPPI is significantly faster than DIAL-MPC while
maintaining control quality through information-theoretic action weighting.

Key differences from DIAL-MPC:
1. Uses all samples (not just top 10%) with exponential reward weighting
2. Optimized for real-time: fewer samples (32 vs 1024), shorter horizon (6 vs 10)
3. Fast mode enabled by default (task reward only, skips contact/imitation)
4. Single iteration per step (no kernel annealing) for speed

Author: Based on run_dial_mpc_policy_guided.py
"""

from __future__ import annotations

import os
import math
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml
import genesis as gs

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string
from dexmachina.envs.reward_utils import position_distance, rotation_distance, chamfer_distance, transform_contact
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def load_policy_agent(checkpoint_path: str, env: BaseEnv, device: str = "cuda:0"):
    """Load a trained RL policy from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        env: The BaseEnv environment (used for observation/action dims)
        device: Device to load the agent on
        
    Returns:
        agent: The loaded rl_games agent ready for inference
        env_wrapper: The RlGamesVecEnvWrapper around env
    """
    # Load agent config
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    
    # Wrap environment for rl-games
    env_wrapper = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    # Register environment
    vecenv.register(
        "IsaacRlgWrapper", 
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register(
        "rlgpu", 
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env_wrapper}
    )
    
    # Configure agent
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs
    
    # Create runner and load agent
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(os.path.abspath(checkpoint_path))
    agent.reset()
    
    return agent, env_wrapper


def get_policy_action(agent, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    """Get action from the trained policy.
    
    Args:
        agent: The rl_games agent
        obs: Observation tensor of shape (num_envs, obs_dim) 
        deterministic: Whether to use deterministic (mean) action
        
    Returns:
        actions: Action tensor of shape (num_envs, action_dim)
    """
    _ = agent.get_batch_size(obs, 1)
    
    if agent.is_rnn:
        agent.init_rnn()
    
    with torch.inference_mode():
        actions = agent.get_action(obs, is_deterministic=deterministic)
    
    return actions


def sample_residual_actions(
    policy_action: torch.Tensor,
    num_samples: int,
    noise_scale: float,
    noise_decay_horizon: float = 0.9,
    horizon: int = 1,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Sample RESIDUAL actions centered around the policy prediction.
    
    Args:
        policy_action: Mean action from policy, shape (1, action_dim) or (action_dim,)
        num_samples: Number of samples to generate
        noise_scale: Scale of Gaussian noise for residuals
        noise_decay_horizon: Decay factor for noise over horizon
        horizon: Planning horizon
        device: Device for tensors
        
    Returns:
        samples: Action samples (policy + residual), shape (num_samples, horizon, action_dim)
    """
    if policy_action.dim() == 1:
        policy_action = policy_action.unsqueeze(0)
    
    if policy_action.shape[0] > 1:
        policy_action = policy_action[:1]
    
    action_dim = policy_action.shape[-1]
    
    # Base: policy action repeated
    samples = policy_action.unsqueeze(1).repeat(num_samples, horizon, 1)
    
    # Add residual perturbations that decay over horizon
    for t in range(horizon):
        decay = noise_decay_horizon ** t
        residual = torch.randn(num_samples, action_dim, device=device) * noise_scale * decay
        samples[:, t, :] += residual
    
    # First sample is pure policy (zero residual)
    samples[0] = policy_action.repeat(horizon, 1)
    
    # Clip to valid range
    samples = torch.clamp(samples, -1.0, 1.0)
    
    return samples


def setup_env_from_checkpoint(checkpoint_path: str, num_envs: int = 32, device: str = "cuda:0", 
                               vis: bool = False, record_video: bool = False, n_render: int = 1,
                               video_res: int = 720):
    """Setup environment using saved config from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        num_envs: Number of parallel environments
        device: Compute device
        vis: Whether to show visualization
        record_video: Whether to record video
        n_render: Number of environments to render
        video_res: Video resolution (width=height)
        
    Returns:
        env: The BaseEnv environment
        env_kwargs: The environment configuration dict
    """
    ckpt_path = "/".join(checkpoint_path.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    assert os.path.exists(saved_cfg_fname), f"Env config not found at {saved_cfg_fname}"
    
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    
    # Modify config for MPPI evaluation
    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    env_kwargs['env_cfg']['is_eval'] = True
    env_kwargs['env_cfg']['num_envs'] = num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    env_kwargs['rand_cfg']['randomize'] = False
    
    # Set env_spacing to 0 so all parallel envs overlap
    env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    
    env_kwargs['env_cfg']['use_contact_reward'] = env_kwargs['reward_cfg'].get('contact_rew_weight', 0.0) > 0.0
    
    # Disable object actuation for real-world deployment (object can't actuate itself)
    # Object will fall unless held by the robot - this matches real-world physics
    for name, cfg in env_kwargs['object_cfgs'].items():
        cfg['actuated'] = False
        cfg['visualization'] = True
    
    for name, cfg in env_kwargs['robot_cfgs'].items():
        cfg['visualization'] = True
    
    # Remove curriculum
    if 'curriculum_cfg' in env_kwargs:
        env_kwargs.pop('curriculum_cfg')
    
    # Visualization settings
    if vis or record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['n_rendered_envs'] = n_render
    
    if vis:
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    
    if record_video:
        env_kwargs['env_cfg']['record_video'] = True
        print(f"Setting render resolution to {video_res}")
        env_kwargs['env_cfg']['camera_kwargs'] = dict(
            front=dict(
                res=(video_res, video_res),
                fov=40,
                pos=(0.5, -1.5, 1.2),
                lookat=(0.0, -0.15, 1.0),
            )
        )
    
    gs.init(backend=gs.gpu, logging_level='warning')
    env = BaseEnv(**env_kwargs)
    
    return env, env_kwargs


def compute_mpc_reward(
    env: BaseEnv,
    num_samples: int,
    device: str = "cuda:0",
    fast_mode: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute reward for MPPI optimization.
    
    Args:
        env: The environment with reward_module
        num_samples: Number of samples (batch size)
        device: Device
        fast_mode: If True, skip contact and imitation reward (NOT recommended - degrades performance)
        
    Returns:
        reward: Total reward, shape (num_samples,)
        info: Dict with reward components
    """
    obj = env.objects[env.object_names[0]]
    obj_pos = obj.entity.get_pos()
    obj_quat = obj.entity.get_quat()
    obj_arti = obj.entity.get_dofs_position(obj.dof_idxs) if hasattr(obj, 'dof_idxs') and len(obj.dof_idxs) > 0 else None
    
    demo_pos = env.reward_module.match_demo_state("obj_pos", env.episode_length_buf)
    demo_quat = env.reward_module.match_demo_state("obj_quat", env.episode_length_buf)
    
    # Task reward - always computed
    task_rew, task_dict = env.reward_module.compute_task_reward(
        obj_pos, obj_quat, obj_arti, demo_pos, demo_quat, env.episode_length_buf
    )
    
    total_rew = task_rew.clone()
    
    def _mean_safe(v):
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bool:
                return v.float().mean().item()
            return v.mean().item()
        return v
    
    info = {k: _mean_safe(v) for k, v in task_dict.items()}
    
    # Skip expensive rewards in fast mode
    if fast_mode:
        info["total_rew"] = total_rew.mean().item()
        return total_rew, info
    
    # Imitation reward if enabled
    if env.reward_module.use_imi_rew:
        kpts_left = env.robots["left"].kpt_pos
        kpts_right = env.robots["right"].kpt_pos
        wrist_pose_left = env.robots["left"].wrist_pose
        wrist_pose_right = env.robots["right"].wrist_pose
        
        imi_rew, imi_dict = env.reward_module.compute_imitation_reward(
            wrist_pose_left, wrist_pose_right,
            kpts_left, kpts_right, env.episode_length_buf
        )
        total_rew += imi_rew
        info["imi_rew"] = imi_rew.mean().item()
    
    # Contact reward if enabled
    if env.reward_module.contact_rew_weight > 0.0 and hasattr(env, 'contact_link_pos'):
        obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
        demo_obj_pose = torch.cat([demo_pos, demo_quat], dim=1)
        wrist_pose_left = env.robots["left"].wrist_pose
        wrist_pose_right = env.robots["right"].wrist_pose
        
        num_left_links = env.num_left_contact_links
        contact_link_pos_left = env.contact_link_pos[:, :, :num_left_links]
        contact_link_valid_left = env.contact_link_valid[:, :, :num_left_links]
        contact_link_pos_right = env.contact_link_pos[:, :, num_left_links:]
        contact_link_valid_right = env.contact_link_valid[:, :, num_left_links:]
        
        if env.reward_module.use_retarget_contact:
            contact_rew, contact_dict = env.reward_module.compute_matched_contact_reward(
                contact_link_pos_left, contact_link_valid_left,
                contact_link_pos_right, contact_link_valid_right,
                obj_pose, demo_obj_pose, env.episode_length_buf
            )
        else:
            left_reshaped, left_valid = env.reward_module.reshape_contact_with_label(
                contact_link_pos_left, contact_link_valid_left
            )
            right_reshaped, right_valid = env.reward_module.reshape_contact_with_label(
                contact_link_pos_right, contact_link_valid_right
            )
            
            contact_rew, contact_dict = env.reward_module.compute_contact_reward(
                obj_pose, wrist_pose_left, wrist_pose_right,
                left_reshaped, left_valid, right_reshaped, right_valid,
                env.episode_length_buf
            )
        total_rew += contact_rew
        info["con_rew"] = contact_rew.mean().item()
    
    info["total_rew"] = total_rew.mean().item()
    
    return total_rew, info


def rollout_samples(
    env: BaseEnv,
    action_samples: torch.Tensor,
    horizon: int,
    device: str = "cuda:0",
    fast_mode: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Rollout action samples and compute rewards.
    
    Args:
        env: The environment
        action_samples: Actions of shape (num_samples, horizon, action_dim)
        horizon: Planning horizon
        device: Device
        fast_mode: If True, use simplified reward (NOT recommended)
        
    Returns:
        rewards: Cumulative rewards, shape (num_samples,)
        info: Dict with rollout info
    """
    num_samples = action_samples.shape[0]
    
    # Save initial state
    initial_state = env.scene.get_state()
    initial_ep_len = env.episode_length_buf.clone()
    initial_robot_ep_len = {}
    for k, robot in env.robots.items():
        initial_robot_ep_len[k] = robot.episode_length_buf.clone()
    initial_obj_ep_len = {}
    for k, obj in env.objects.items():
        initial_obj_ep_len[k] = obj.episode_length_buf.clone()
    
    # Rollout
    cumulative_reward = torch.zeros(num_samples, device=device)
    
    for t in range(horizon):
        actions = action_samples[:, t, :]
        
        # Step robots
        scaled_actions = actions * env.action_scale
        for k, robot in env.robots.items():
            idxs = env.action_idxs_to_robot[k]
            robot.step(scaled_actions[:, idxs], list(range(num_samples)))
        for obj in env.objects.values():
            obj.step()
        env.scene.step(update_visualizer=False)
        env.episode_length_buf += 1
        env._compute_intermediate_values()
        
        # Compute reward
        step_reward, step_info = compute_mpc_reward(env, num_samples, device, fast_mode=fast_mode)
        cumulative_reward += step_reward
    
    # Restore state
    env.scene.reset(state=initial_state)
    env.episode_length_buf = initial_ep_len
    for k, robot in env.robots.items():
        robot.episode_length_buf = initial_robot_ep_len[k]
    for k, obj in env.objects.items():
        obj.episode_length_buf = initial_obj_ep_len[k]
    
    # Return mean reward
    mean_reward = cumulative_reward / horizon
    
    return mean_reward, step_info


def compute_mppi_action(
    samples: torch.Tensor, 
    rewards: torch.Tensor, 
    lambda_: float = 1.0,
    nu: float = 0.0,
) -> torch.Tensor:
    """Compute MPPI information-theoretic weighted action.
    
    This is the core MPPI algorithm: weighted mean using exponential reward weights.
    
    Args:
        samples: Action samples, shape (num_samples, horizon, action_dim)
        rewards: Rewards for each sample, shape (num_samples,)
        lambda_: Temperature parameter (lower = more exploitation)
        nu: Baseline reward for numerical stability (auto-computed if 0)
        
    Returns:
        weighted_action: MPPI weighted action, shape (horizon, action_dim)
    """
    # Handle NaNs
    nan_mask = torch.isnan(rewards) | torch.isinf(rewards)
    if nan_mask.any():
        valid_rewards = rewards[~nan_mask]
        min_reward = valid_rewards.min() if len(valid_rewards) > 0 else torch.tensor(-1000.0, device=rewards.device)
        rewards = torch.where(nan_mask, min_reward, rewards)
    
    # Baseline for numerical stability
    if nu == 0.0:
        nu = rewards.max()
    
    # MPPI weights: exp((reward - nu) / lambda)
    weights = torch.exp((rewards - nu) / lambda_)
    weights = weights / (weights.sum() + 1e-10)
    
    # Weighted mean: sum(w_i * action_i)
    weighted_action = (weights[:, None, None] * samples).sum(dim=0)
    
    return weighted_action


def sync_env_state(env: BaseEnv, src_idx: int = 0):
    """Broadcast state from one env to all others.
    
    Args:
        env: The environment
        src_idx: Source environment index to broadcast from
    """
    num_envs = env.num_envs
    
    for robot in env.robots.values():
        pos = robot.entity.get_dofs_position()
        vel = robot.entity.get_dofs_velocity()
        pos = pos[src_idx:src_idx+1].repeat(num_envs, 1)
        vel = vel[src_idx:src_idx+1].repeat(num_envs, 1)
        robot.entity.set_dofs_position(pos)
        robot.entity.set_dofs_velocity(vel)
    
    for obj in env.objects.values():
        pos = obj.entity.get_dofs_position()
        vel = obj.entity.get_dofs_velocity()
        pos = pos[src_idx:src_idx+1].repeat(num_envs, 1)
        vel = vel[src_idx:src_idx+1].repeat(num_envs, 1)
        obj.entity.set_dofs_position(pos)
        obj.entity.set_dofs_velocity(vel)


def set_object_gains(env: BaseEnv, kp: float, kv: float, device: str = "cuda:0"):
    """Set object PD gains (only works if object is actuated).
    
    Note: This function is kept for API compatibility but won't work with actuated=False.
    With actuated=False (real-world mode), the object has no self-actuation and will
    fall unless held by the robot.
    
    Args:
        env: The environment
        kp: Stiffness 
        kv: Damping
        device: Device
    """
    obj = env.objects[env.object_names[0]]
    if obj.actuated:
        kp_tensor = torch.ones_like(obj.entity.get_dofs_kp(), device=device) * kp
        kv_tensor = torch.ones_like(obj.entity.get_dofs_kv(), device=device) * kv
        obj.entity.set_dofs_kp(kp_tensor)
        obj.entity.set_dofs_kv(kv_tensor)
    else:
        # Object is not actuated - it will fall unless held by robot (real-world physics)
        pass


def main():
    parser = argparse.ArgumentParser(description="Run MPPI residual controller with policy-guided sampling")
    parser.add_argument("--checkpoint", "-ck", type=str, required=True,
                        help="Path to trained RL policy checkpoint (.pth file)")
    parser.add_argument("--num_samples", "-N", type=int, default=32,
                        help="Number of parallel samples for MPPI (default: 32 for real-time)")
    parser.add_argument("--horizon", "-H", type=int, default=16,
                        help="MPPI planning horizon in steps (default: 16, tuned optimal)")
    parser.add_argument("--lambda_temp", "-l", type=float, default=1.0,
                        help="MPPI temperature (default: 1.0, tuned optimal)")
    parser.add_argument("--noise_scale", "-n", type=float, default=0.2,
                        help="Residual noise scale (default: 0.2, tuned optimal)")
    parser.add_argument("--record_video", "-v", action="store_true",
                        help="Record video of the execution")
    parser.add_argument("--video_res", type=int, default=720,
                        help="Video resolution (default: 720)")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/mppi_policy",
                        help="Output directory for results")
    parser.add_argument("--vis", action="store_true", help="Show visualization")
    parser.add_argument("--n_render", "-nr", type=int, default=1,
                        help="Number of environments to render")
    parser.add_argument("--max_steps", "-s", type=int, default=-1,
                        help="Maximum simulation steps (-1 for full episode)")
    parser.add_argument("--fast_mode", "-f", action="store_true",
                        help="Fast mode: task reward only (faster but may degrade long-term performance)")
    parser.add_argument("--policy_only", action="store_true",
                        help="Use policy only (no MPPI), fastest option")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"
    
    print(f"Loading environment from checkpoint: {args.checkpoint}")
    env, env_kwargs = setup_env_from_checkpoint(
        args.checkpoint, 
        num_envs=args.num_samples,
        device=device,
        vis=args.vis,
        record_video=args.record_video,
        n_render=args.n_render,
        video_res=args.video_res,
    )
    
    env.reset()
    
    # Let physics settle before starting (prevent initial bounce)
    print("Settling physics for 10 steps...")
    for _ in range(10):
        # Execute zero actions to let object settle on ground
        zero_actions = torch.zeros(args.num_samples, env.action_dim, device=device) * env.action_scale
        for k, robot in env.robots.items():
            idxs = env.action_idxs_to_robot[k]
            robot.step(zero_actions[:, idxs], list(range(args.num_samples)))
        for obj in env.objects.values():
            obj.step()
        env.scene.step(update_visualizer=False)
    print("Physics settled.")
    
    print(f"Loading policy agent...")
    agent, env_wrapper = load_policy_agent(args.checkpoint, env, device=device)
    
    action_dim = env.action_dim
    max_steps = args.max_steps if args.max_steps > 0 else env.max_episode_length
    
    # Print configuration
    print(f"\n=== MPPI Configuration ===")
    print(f"Action dim: {action_dim}, Max steps: {max_steps}")
    if args.policy_only:
        print(f"Mode: POLICY ONLY (no MPPI)")
    else:
        print(f"MPPI: samples={args.num_samples}, horizon={args.horizon}, lambda={args.lambda_temp}")
        print(f"Noise: scale={args.noise_scale} (residual)")
        print(f"Full reward: contact + imitation + task (slow but accurate)")
    print(f"==========================\n")
    
    # Tracking
    metrics = defaultdict(list)
    
    t_start = time.perf_counter()
    
    for step in range(max_steps):
        t0 = time.perf_counter()
        
        # Sync all envs to match env 0
        sync_env_state(env, src_idx=0)
        env._compute_intermediate_values()
        
        # Get observation
        obs_dict = env.get_observations()
        if isinstance(obs_dict, dict):
            obs = obs_dict["policy"]
        else:
            obs = obs_dict
        
        # Get policy action
        policy_action = get_policy_action(agent, obs[:1], deterministic=True)
        policy_action = policy_action.squeeze(0)
        
        # Policy-only mode
        if args.policy_only:
            final_action = policy_action.clone()
            rewards = torch.zeros(1, device=device)
        else:
            # MPPI optimization
            action_samples = sample_residual_actions(
                policy_action,
                num_samples=args.num_samples,
                noise_scale=args.noise_scale,
                noise_decay_horizon=0.9,
                horizon=args.horizon,
                device=device,
            )
            
            # Rollout and evaluate
            rewards, rollout_info = rollout_samples(
                env, action_samples, args.horizon, device, fast_mode=args.fast_mode
            )
            
            # MPPI weighted action
            weighted_action = compute_mppi_action(
                action_samples, rewards, lambda_=args.lambda_temp
            )
            
            # Take first timestep
            final_action = torch.clamp(weighted_action[0], -1.0, 1.0)
        
        # Execute action
        action_batch = final_action.unsqueeze(0).repeat(args.num_samples, 1)
        
        scaled_actions = action_batch * env.action_scale
        for k, robot in env.robots.items():
            idxs = env.action_idxs_to_robot[k]
            robot.step(scaled_actions[:, idxs], list(range(args.num_samples)))
        for obj in env.objects.values():
            obj.step()
        env.scene.step()
        env.episode_length_buf += 1
        env._compute_intermediate_values()
        
        # Record video frame
        if args.record_video:
            env._recording = True
            env._render_headless()
            env._recording = False
        
        # Track metrics
        obj = env.objects[env.object_names[0]]
        obj_pos = obj.entity.get_pos()[0]
        obj_quat = obj.entity.get_quat()[0]
        demo_pos = env.reward_module.match_demo_state("obj_pos", env.episode_length_buf)[0]
        demo_quat = env.reward_module.match_demo_state("obj_quat", env.episode_length_buf)[0]
        
        # Early exit if object dropped (z < 0.5 means object fell off platform)
        if obj_pos[2] < 0.5:
            print(f"\n\n[EARLY EXIT] Object dropped at step {step+1} (z={obj_pos[2]:.3f})")
            print(f"Stopping early to save time on failed run.")
            # Fill remaining metrics with failure values
            remaining_steps = max_steps - step - 1
            for _ in range(remaining_steps):
                metrics["pos_dist"].append(10.0)  # Large error to indicate failure
                metrics["rot_dist"].append(10.0)
                metrics["arti_dist"].append(10.0)
                metrics["reward_max"].append(-100.0)
                metrics["reward_mean"].append(-100.0)
                metrics["residual_norm"].append(0.0)
            break
        
        pos_dist = (obj_pos - demo_pos).norm().item()
        rot_dist = rotation_distance(demo_quat.unsqueeze(0), obj_quat.unsqueeze(0)).item()
        
        if hasattr(obj, 'dof_idxs') and len(obj.dof_idxs) > 0:
            obj_arti = obj.entity.get_dofs_position(obj.dof_idxs)[0]
            demo_arti = env.reward_module.match_demo_state("obj_arti", env.episode_length_buf)[0]
            arti_dist = (obj_arti - demo_arti).abs().item()
        else:
            arti_dist = 0.0
        
        metrics["pos_dist"].append(pos_dist)
        metrics["rot_dist"].append(rot_dist)
        metrics["arti_dist"].append(arti_dist)
        metrics["reward_max"].append(rewards.max().item())
        metrics["reward_mean"].append(rewards.mean().item())
        
        # Track residual magnitude
        residual_norm = (final_action - policy_action).norm().item()
        metrics["residual_norm"].append(residual_norm)
        
        t1 = time.perf_counter()
        step_time = t1 - t0
        rtr = env.dt / step_time if step_time > 0 else 0
        hz = 1.0 / step_time if step_time > 0 else 0
        
        print(
            f"Step {step+1}/{max_steps} | "
            f"pos: {pos_dist:.4f}, rot: {rot_dist:.4f}, arti: {arti_dist:.4f} | "
            f"rew: {rewards.max().item():.3f} | "
            f"res: {residual_norm:.3f} | "
            f"{hz:.1f}Hz ({step_time*1000:.1f}ms) RTR: {rtr:.2f}x   ",
            end="\r"
        )
    
    t_end = time.perf_counter()
    total_time = t_end - t_start
    avg_hz = max_steps / total_time if total_time > 0 else 0
    
    print(f"\n\nTotal time: {total_time:.2f}s, Avg frequency: {avg_hz:.1f}Hz")
    
    # Summary statistics
    pos_dists = np.array(metrics["pos_dist"])
    rot_dists = np.array(metrics["rot_dist"])
    arti_dists = np.array(metrics["arti_dist"])
    residual_norms = np.array(metrics["residual_norm"])
    print(f"Position distance: {pos_dists.mean():.4f} ± {pos_dists.std():.4f}")
    print(f"Rotation distance: {rot_dists.mean():.4f} ± {rot_dists.std():.4f}")
    print(f"Articulation distance: {arti_dists.mean():.4f} ± {arti_dists.std():.4f}")
    print(f"Residual norm (MPPI correction): {residual_norms.mean():.4f} ± {residual_norms.std():.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.npz")
    np.savez(metrics_path, **{k: np.array(v) for k, v in metrics.items()})
    print(f"Saved metrics to {metrics_path}")
    
    # Save video
    if args.record_video and hasattr(env, '_recorded_frames') and len(env._recorded_frames) > 0:
        fname = f"mppi_N{args.num_samples}_H{args.horizon}_lambda{args.lambda_temp}_fast{args.fast_mode}.mp4"
        video_path = os.path.join(args.output_dir, fname)
        env.export_video(video_path, wait_for_max=False)
        print(f"Saved video to {video_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
