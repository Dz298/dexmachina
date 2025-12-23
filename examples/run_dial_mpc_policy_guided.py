#!/usr/bin/env python3
"""Run DIAL MPC with Genesis (DexMachina), guided by a trained RL policy prior.

This script combines the sample-based DIAL MPC planning from spider with a trained 
RL policy that biases the action sampling towards learned behavior. The MPC acts as
a RESIDUAL corrective method on top of the policy - sampling perturbations around
the policy's predicted actions to refine them.

Key features:
1. Virtual contact constraint (object kp/kv annealing) as in SPIDER
2. Same reward function as RL training (task + imitation + contact)
3. Residual MPC - samples corrections to policy actions

Author: Based on spider/examples/run_dexmachina.py
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
    # Ensure batch size is set
    _ = agent.get_batch_size(obs, 1)
    
    # Initialize RNN states if used
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
    
    This implements residual MPC - we sample small perturbations/corrections
    to the policy's action, not completely new actions.
    
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
    # Ensure policy_action is 2D: (batch, action_dim)
    if policy_action.dim() == 1:
        policy_action = policy_action.unsqueeze(0)  # (1, action_dim)
    
    # If batch size > 1, just take the first one (single env case)
    if policy_action.shape[0] > 1:
        policy_action = policy_action[:1]  # (1, action_dim)
    
    action_dim = policy_action.shape[-1]
    
    # Create base samples by repeating policy action
    # (1, action_dim) -> unsqueeze(1) -> (1, 1, action_dim) -> repeat -> (N, H, A)
    samples = policy_action.unsqueeze(1).repeat(num_samples, horizon, 1)  # (N, H, A)
    
    # Sample residual corrections that decay over horizon
    for t in range(horizon):
        decay = noise_decay_horizon ** t
        # Residual perturbations - smaller than full action range
        residual = torch.randn(num_samples, action_dim, device=device) * noise_scale * decay
        samples[:, t, :] += residual
    
    # First sample is pure policy (zero residual) for exploitation
    # policy_action is (1, action_dim), repeat to (horizon, action_dim)
    samples[0] = policy_action.repeat(horizon, 1)
    
    # Clip actions to valid range
    samples = torch.clamp(samples, -1.0, 1.0)
    
    return samples


def setup_env_from_checkpoint(checkpoint_path: str, num_envs: int = 1024, device: str = "cuda:0", 
                               vis: bool = False, record_video: bool = False, n_render: int = 1,
                               video_res: int = 720):
    """Setup environment using saved config from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        num_envs: Number of parallel environments
        device: Compute device
        vis: Whether to show visualization
        record_video: Whether to record video
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
    
    # Modify config for MPC evaluation
    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    env_kwargs['env_cfg']['is_eval'] = True
    env_kwargs['env_cfg']['num_envs'] = num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    env_kwargs['rand_cfg']['randomize'] = False
    
    # IMPORTANT: Set env_spacing to 0 so all parallel envs overlap at same position
    # This is needed for MPC sampling where we want traces at the same origin
    env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    
    # Keep contact reward enabled for proper reward computation
    env_kwargs['env_cfg']['use_contact_reward'] = env_kwargs['reward_cfg'].get('contact_rew_weight', 0.0) > 0.0
    
    # Disable object actuation for free manipulation (will control via kp/kv annealing)
    # Ensure visualization is enabled for all entities
    for name, cfg in env_kwargs['object_cfgs'].items():
        cfg['actuated'] = False
        cfg['visualization'] = True
    
    for name, cfg in env_kwargs['robot_cfgs'].items():
        cfg['visualization'] = True
    
    # Remove curriculum
    if 'curriculum_cfg' in env_kwargs:
        env_kwargs.pop('curriculum_cfg')
    
    # Visualization settings - only render n_render environments (env 0 is the "real" one that gets executed)
    if vis or record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['n_rendered_envs'] = n_render
    
    if vis:
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    
    if record_video:
        env_kwargs['env_cfg']['record_video'] = True
        # Override camera config for higher quality video (matching eval_rl_games.py)
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
    """Compute reward matching the RL training reward function.
    
    This includes:
    - Task reward: Object position, rotation, articulation tracking
    - Imitation reward: Fingertip keypoint tracking (skipped in fast_mode)
    - Contact reward: Chamfer distance between policy and demo contacts (skipped in fast_mode)
    
    Args:
        env: The environment with reward_module
        num_samples: Number of samples (batch size)
        device: Device
        fast_mode: If True, skip expensive contact and imitation reward computation
        
    Returns:
        reward: Total reward, shape (num_samples,)
        info: Dict with reward components
    """
    # Get object state
    obj = env.objects[env.object_names[0]]
    obj_pos = obj.entity.get_pos()
    obj_quat = obj.entity.get_quat()
    obj_arti = obj.entity.get_dofs_position(obj.dof_idxs) if hasattr(obj, 'dof_idxs') and len(obj.dof_idxs) > 0 else None
    
    # Get demo states
    demo_pos = env.reward_module.match_demo_state("obj_pos", env.episode_length_buf)
    demo_quat = env.reward_module.match_demo_state("obj_quat", env.episode_length_buf)
    
    # Compute task reward (same as RL training) - always computed, relatively cheap
    task_rew, task_dict = env.reward_module.compute_task_reward(
        obj_pos, obj_quat, obj_arti, demo_pos, demo_quat, env.episode_length_buf
    )
    
    total_rew = task_rew.clone()
    # Handle boolean and float tensors when computing mean
    def _mean_safe(v):
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bool:
                return v.float().mean().item()
            return v.mean().item()
        return v
    info = {k: _mean_safe(v) for k, v in task_dict.items()}
    
    # Skip expensive rewards in fast mode for real-time deployment
    if fast_mode:
        info["total_rew"] = total_rew.mean().item()
        return total_rew, info
    
    # Compute imitation reward if enabled
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
    
    # Compute contact reward if enabled
    if env.reward_module.contact_rew_weight > 0.0 and hasattr(env, 'contact_link_pos'):
        obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
        demo_obj_pose = torch.cat([demo_pos, demo_quat], dim=1)
        wrist_pose_left = env.robots["left"].wrist_pose
        wrist_pose_right = env.robots["right"].wrist_pose
        
        # Split contact data using env's num_left_contact_links (not keypoint links!)
        num_left_links = env.num_left_contact_links
        contact_link_pos_left = env.contact_link_pos[:, :, :num_left_links]
        contact_link_valid_left = env.contact_link_valid[:, :, :num_left_links]
        contact_link_pos_right = env.contact_link_pos[:, :, num_left_links:]
        contact_link_valid_right = env.contact_link_valid[:, :, num_left_links:]
        
        # Use the appropriate contact reward based on whether retarget contacts are used
        if env.reward_module.use_retarget_contact:
            contact_rew, contact_dict = env.reward_module.compute_matched_contact_reward(
                contact_link_pos_left, contact_link_valid_left,
                contact_link_pos_right, contact_link_valid_right,
                obj_pose, demo_obj_pose, env.episode_length_buf
            )
        else:
            # Reshape contacts for non-retarget mode
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
    collect_traces: bool = False,
    fast_mode: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Rollout action samples and compute rewards using RL training reward function.
    
    Args:
        env: The environment
        action_samples: Actions of shape (num_samples, horizon, action_dim)
        horizon: Planning horizon
        device: Device
        collect_traces: Whether to collect fingertip positions for visualization
        fast_mode: If True, use simplified reward (skip contact/imitation) for speed
        
    Returns:
        rewards: Cumulative rewards, shape (num_samples,)
        info: Dict with rollout info (includes 'traces' if collect_traces=True)
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
    
    # Collect fingertip traces if requested
    traces = {'left': [], 'right': []} if collect_traces else None
    
    for t in range(horizon):
        actions = action_samples[:, t, :]  # (N, A)
        
        # Step robots
        scaled_actions = actions * env.action_scale
        for k, robot in env.robots.items():
            idxs = env.action_idxs_to_robot[k]
            robot.step(scaled_actions[:, idxs], list(range(num_samples)))
        for obj in env.objects.values():
            obj.step()
        # Don't update visualizer during rollout - only show final executed action
        env.scene.step(update_visualizer=False)
        env.episode_length_buf += 1
        env._compute_intermediate_values()
        
        # Collect fingertip positions for trace visualization
        if collect_traces:
            for k in ['left', 'right']:
                if k in env.robots:
                    kpt_pos = env.robots[k].kpt_pos.clone()  # (N, num_kpts, 3)
                    traces[k].append(kpt_pos)
        
        # Compute reward - use fast_mode to skip expensive contact/imitation rewards
        step_reward, step_info = compute_mpc_reward(env, num_samples, device, fast_mode=fast_mode)
        cumulative_reward += step_reward
    
    # Restore state
    env.scene.reset(state=initial_state)
    env.episode_length_buf = initial_ep_len
    for k, robot in env.robots.items():
        robot.episode_length_buf = initial_robot_ep_len[k]
    for k, obj in env.objects.items():
        obj.episode_length_buf = initial_obj_ep_len[k]
    
    # Return mean reward and last step info
    mean_reward = cumulative_reward / horizon
    
    # Stack traces: (N, horizon, num_kpts, 3)
    if collect_traces:
        for k in traces:
            if traces[k]:
                traces[k] = torch.stack(traces[k], dim=1)
        step_info['traces'] = traces
    
    return mean_reward, step_info


def visualize_sample_traces(
    env: BaseEnv,
    traces: dict,
    rewards: torch.Tensor,
    n_traces: int = 10,
    trace_nodes: list = None,
) -> list:
    """Visualize fingertip trajectory traces: random samples + best sample.
    
    Args:
        env: The environment with scene for debug drawing
        traces: Dict with 'left' and 'right' traces, each shape (N, horizon, num_kpts, 3)
        rewards: Rewards for each sample, shape (N,)
        n_traces: Number of random traces to visualize (plus best trace)
        trace_nodes: Previous trace nodes to remove (for updating visualization)
        
    Returns:
        trace_nodes: List of created debug nodes for cleanup
    """
    # Clear ALL previous debug objects to ensure clean slate
    try:
        env.scene.clear_debug_objects()
    except:
        pass
    
    new_nodes = []
    
    num_samples = len(rewards)
    
    # Get best sample index
    best_idx = torch.argmax(rewards).item()
    
    # Randomly sample (n_traces - 1) other indices (excluding best)
    other_indices = [i for i in range(num_samples) if i != best_idx]
    n_random = min(n_traces - 1, len(other_indices))
    if n_random > 0:
        random_indices = np.random.choice(other_indices, size=n_random, replace=False)
    else:
        random_indices = []
    
    # Combine: random samples first, then best (so best is drawn on top)
    sample_indices = list(random_indices) + [best_idx]
    
    # Collect all points for batch drawing
    other_points = []  # Points for random samples
    best_points = []   # Points for best sample
    
    for sample_idx in sample_indices:
        is_best = (sample_idx == best_idx)
        
        for side in ['left', 'right']:
            if side not in traces or traces[side] is None or len(traces[side]) == 0:
                continue
                
            # Get trace for this sample: (horizon, num_kpts, 3)
            sample_trace = traces[side][sample_idx].cpu().numpy()
            horizon = sample_trace.shape[0]
            num_kpts = sample_trace.shape[1]
            
            # Only draw traces for first 5 keypoints (the actual fingertips)
            num_fingertips = min(5, num_kpts)
            
            # Collect all points along the trace
            for kpt_idx in range(num_fingertips):
                for t in range(horizon):
                    point = sample_trace[t, kpt_idx]
                    if is_best:
                        best_points.append(point)
                    else:
                        other_points.append(point)
    
    # Draw all points at once (much faster than individual lines)
    try:
        # Draw random sample points (gray)
        if len(other_points) > 0:
            other_points = np.array(other_points)
            other_color = (0.5, 0.5, 0.6, 0.4)
            node = env.scene.draw_debug_points(other_points, colors=other_color)
            if node is not None:
                new_nodes.append(node)
        
        # Draw best sample points (red) - drawn last so on top
        if len(best_points) > 0:
            best_points = np.array(best_points)
            best_color = (1.0, 0.0, 0.0, 1.0)
            node = env.scene.draw_debug_points(best_points, colors=best_color)
            if node is not None:
                new_nodes.append(node)
    except Exception as e:
        pass  # Skip if drawing fails
    
    return new_nodes


def compute_weighted_mean(samples: torch.Tensor, rewards: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute weighted mean of samples using softmax weights on top 10%.
    
    Args:
        samples: Action samples, shape (num_samples, horizon, action_dim)
        rewards: Rewards for each sample, shape (num_samples,)
        temperature: Softmax temperature
        
    Returns:
        weighted_mean: Weighted mean action, shape (horizon, action_dim)
    """
    # Handle NaNs
    nan_mask = torch.isnan(rewards) | torch.isinf(rewards)
    if nan_mask.any():
        rewards = torch.where(nan_mask, rewards[~nan_mask].min() if (~nan_mask).any() else torch.tensor(-1000.0), rewards)
    
    # Select top 10% samples for weighting
    top_k = max(1, int(0.1 * len(rewards)))
    top_indices = torch.topk(rewards, k=top_k, largest=True).indices
    
    # Normalize rewards for top samples
    top_rewards = rewards[top_indices]
    top_rewards_normalized = (top_rewards - top_rewards.mean()) / (top_rewards.std() + 1e-2)
    
    # Compute softmax weights for top samples
    weights = torch.softmax(top_rewards_normalized / temperature, dim=0)
    
    # Weighted mean over top samples
    top_samples = samples[top_indices]  # (K, H, A)
    weighted_mean = (weights[:, None, None] * top_samples).sum(dim=0)  # (H, A)
    
    return weighted_mean


def sync_env_state(env: BaseEnv, src_idx: int = 0):
    """Broadcast state from one env to all others.
    
    Args:
        env: The environment
        src_idx: Source environment index to broadcast from
    """
    num_envs = env.num_envs
    
    # Get positions/velocities from source
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


def setup_virtual_contact_constraint(
    env: BaseEnv,
    num_iterations: int,
    kp_max: float = 100.0,
    kv_max: float = 10.0,
    device: str = "cuda:0",
) -> tuple[list, list]:
    """Setup virtual contact constraint schedule (object kp/kv annealing).
    
    High kp/kv initially "sticks" object to reference trajectory,
    then decays to 0 for free manipulation.
    
    Args:
        env: The environment
        num_iterations: Number of MPC iterations per step
        kp_max: Maximum stiffness
        kv_max: Maximum damping
        device: Device
        
    Returns:
        kp_list: List of kp values for each iteration
        kv_list: List of kv values for each iteration
    """
    # Exponential decay from kp_max to 0
    eta = 0.001 ** (1.0 / max(1, num_iterations - 1))
    kp_list = kp_max * (eta ** np.arange(num_iterations))
    kp_list[-1] = 0.0  # Final iteration is free dynamics
    kv_list = kv_max * (eta ** np.arange(num_iterations))
    kv_list[-1] = 0.0
    
    return kp_list.tolist(), kv_list.tolist()


def apply_virtual_constraint(env: BaseEnv, kp: float, kv: float, device: str = "cuda:0"):
    """Apply virtual contact constraint by setting object joint gains.
    
    Args:
        env: The environment
        kp: Stiffness value
        kv: Damping value
        device: Device
    """
    obj = env.objects[env.object_names[0]]
    kp_tensor = torch.ones_like(obj.entity.get_dofs_kp(), device=device) * kp
    kv_tensor = torch.ones_like(obj.entity.get_dofs_kv(), device=device) * kv
    obj.entity.set_dofs_kp(kp_tensor)
    obj.entity.set_dofs_kv(kv_tensor)


def main():
    parser = argparse.ArgumentParser(description="Run DIAL MPC with policy-guided residual sampling")
    parser.add_argument("--checkpoint", "-ck", type=str, required=True,
                        help="Path to trained RL policy checkpoint (.pth file)")
    parser.add_argument("--num_samples", "-N", type=int, default=1024,
                        help="Number of parallel samples for MPC")
    parser.add_argument("--horizon", "-H", type=int, default=10,
                        help="MPC planning horizon (in simulation steps)")
    parser.add_argument("--num_iterations", "-I", type=int, default=8,
                        help="Number of MPC optimization iterations per control step")
    parser.add_argument("--noise_scale", "-n", type=float, default=0.2,
                        help="Initial noise scale for residual sampling (smaller than full actions)")
    parser.add_argument("--noise_decay", "-d", type=float, default=0.8,
                        help="Noise decay factor across iterations")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                        help="Softmax temperature for reward weighting")
    parser.add_argument("--kp_max", type=float, default=0.0,
                        help="Max object stiffness for virtual constraint (0=disabled)")
    parser.add_argument("--kv_max", type=float, default=0.0,
                        help="Max object damping for virtual constraint (0=disabled)")
    parser.add_argument("--record_video", "-v", action="store_true",
                        help="Record video of the execution")
    parser.add_argument("--video_res", type=int, default=720,
                        help="Video resolution (default: 720 for 720x720)")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/dial_mpc_policy",
                        help="Output directory for results")
    parser.add_argument("--vis", action="store_true", help="Show visualization")
    parser.add_argument("--n_render", "-nr", type=int, default=1,
                        help="Number of environments to render (1=only real env)")
    parser.add_argument("--show_traces", action="store_true",
                        help="Show fingertip trajectory traces for MPC samples")
    parser.add_argument("--n_traces", type=int, default=10,
                        help="Number of top sample traces to visualize")
    parser.add_argument("--max_steps", "-s", type=int, default=-1,
                        help="Maximum simulation steps (-1 for full episode)")
    # Real-time deployment options
    parser.add_argument("--fast_mode", "-f", action="store_true",
                        help="Fast mode for real-time: skip expensive contact/imitation reward during MPC")
    parser.add_argument("--policy_only", action="store_true",
                        help="Use policy only (no MPC), fastest option for deployment")
    # Reward coefficient overrides (default: use values from RL training)
    parser.add_argument("--task_rew_weight", type=float, default=None,
                        help="Task reward weight (default: use RL training value)")
    parser.add_argument("--imi_rew_weight", type=float, default=None,
                        help="Imitation reward weight (default: use RL training value)")
    parser.add_argument("--con_rew_weight", type=float, default=None,
                        help="Contact reward weight (default: use RL training value)")
    
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
    
    # Reset environment
    env.reset()
    
    print(f"Loading policy agent...")
    agent, env_wrapper = load_policy_agent(args.checkpoint, env, device=device)
    
    # Get action/obs dimensions
    action_dim = env.action_dim
    max_steps = args.max_steps if args.max_steps > 0 else env.max_episode_length
    
    # Setup virtual contact constraint schedule
    kp_list, kv_list = setup_virtual_contact_constraint(
        env, args.num_iterations, args.kp_max, args.kv_max, device
    )
    
    # Override reward weights if specified
    if args.task_rew_weight is not None:
        print(f"Overriding task_rew_weight: {env.reward_module.task_rew_weight} -> {args.task_rew_weight}")
        env.reward_module.task_rew_weight = args.task_rew_weight
    if args.imi_rew_weight is not None:
        print(f"Overriding imi_rew_weight: {env.reward_module.imi_rew_weight} -> {args.imi_rew_weight}")
        env.reward_module.imi_rew_weight = args.imi_rew_weight
        env.reward_module.use_imi_rew = args.imi_rew_weight > 0.0
    if args.con_rew_weight is not None:
        print(f"Overriding contact_rew_weight: {env.reward_module.contact_rew_weight} -> {args.con_rew_weight}")
        env.reward_module.contact_rew_weight = args.con_rew_weight
    
    # Debug: Show reward module status
    print(f"\n=== Reward Module Debug ===")
    print(f"  use_imi_rew: {env.reward_module.use_imi_rew}")
    print(f"  use_retarget_contact: {env.reward_module.use_retarget_contact}")
    print(f"  contact_rew_weight: {env.reward_module.contact_rew_weight}")
    print(f"  has contact_link_pos: {hasattr(env, 'contact_link_pos')}")
    if hasattr(env, 'contact_link_pos'):
        print(f"  contact_link_pos shape: {env.contact_link_pos.shape}")
    print(f"===========================\n")
    
    # Print configuration
    print(f"Action dim: {action_dim}, Max steps: {max_steps}")
    if args.policy_only:
        print(f"Mode: POLICY ONLY (no MPC) - fastest for deployment")
    else:
        print(f"MPC config: samples={args.num_samples}, horizon={args.horizon}, iters={args.num_iterations}")
        print(f"Noise config: scale={args.noise_scale} (residual), decay={args.noise_decay}")
        print(f"Virtual constraint: kp_max={args.kp_max}, kv_max={args.kv_max}")
        if args.fast_mode:
            print(f"Fast mode: ON (skipping contact/imitation reward in MPC)")
    print(f"Reward config: task={env.reward_module.task_rew_weight}, "
          f"imi={env.reward_module.imi_rew_weight}, con={env.reward_module.contact_rew_weight}")
    
    # Tracking metrics
    metrics = defaultdict(list)
    
    # Trace visualization nodes (for cleanup between steps)
    trace_nodes = None
    
    t_start = time.perf_counter()
    
    for step in range(max_steps):
        t0 = time.perf_counter()
        
        # Sync all envs to match env 0
        sync_env_state(env, src_idx=0)
        env._compute_intermediate_values()
        
        # Get observation for policy
        obs_dict = env.get_observations()
        if isinstance(obs_dict, dict):
            obs = obs_dict["policy"]
        else:
            obs = obs_dict
        
        # Get policy action as the BASE action (use first env's observation)
        policy_action = get_policy_action(agent, obs[:1], deterministic=True)  # (1, A)
        policy_action = policy_action.squeeze(0)  # (A,)
        
        # Policy-only mode: skip MPC entirely for maximum speed
        if args.policy_only:
            final_action = policy_action.clone()
            rewards = torch.zeros(1, device=device)  # Dummy for logging
        else:
            # MPC optimization loop - samples RESIDUALS on top of policy
            noise_scale = args.noise_scale
            best_action = policy_action.clone()  # Start with pure policy
            
            for iteration in range(args.num_iterations):
                # Apply virtual contact constraint for this iteration
                if args.kp_max > 0 or args.kv_max > 0:
                    apply_virtual_constraint(env, kp_list[iteration], kv_list[iteration], device)
                
                # Sample residual actions around policy prediction
                action_samples = sample_residual_actions(
                    policy_action,
                    num_samples=args.num_samples,
                    noise_scale=noise_scale,
                    noise_decay_horizon=0.9,
                    horizon=args.horizon,
                    device=device,
                )
                
                # Rollout and evaluate using RL training reward
                # Only collect traces on final iteration for visualization
                collect_traces = args.show_traces and (iteration == args.num_iterations - 1)
                rewards, rollout_info = rollout_samples(
                    env, action_samples, args.horizon, device,
                    collect_traces=collect_traces,
                    fast_mode=args.fast_mode,
                )
                
                # Visualize traces if enabled (only on final iteration)
                if collect_traces and 'traces' in rollout_info:
                    trace_nodes = visualize_sample_traces(
                        env, rollout_info['traces'], rewards, 
                        n_traces=args.n_traces, trace_nodes=trace_nodes
                    )
                
                # Compute weighted mean action
                weighted_action = compute_weighted_mean(
                    action_samples, rewards, args.temperature
                )
                
                # Update best action (take first timestep)
                best_action = weighted_action[0]
                
                # Decay noise for next iteration (kernel annealing)
                noise_scale *= args.noise_decay
            
            # Final action = MPC refined action (already includes policy as base)
            final_action = torch.clamp(best_action, -1.0, 1.0)
        
        # Reset virtual constraint to 0 for actual execution
        if args.kp_max > 0 or args.kv_max > 0:
            apply_virtual_constraint(env, 0.0, 0.0, device)
        
        # Execute action in all envs
        action_batch = final_action.unsqueeze(0).repeat(args.num_samples, 1)
        
        # Step environment
        scaled_actions = action_batch * env.action_scale
        for k, robot in env.robots.items():
            idxs = env.action_idxs_to_robot[k]
            robot.step(scaled_actions[:, idxs], list(range(args.num_samples)))
        for obj in env.objects.values():
            obj.step()
        env.scene.step()
        env.episode_length_buf += 1
        env._compute_intermediate_values()
        
        # Record video frame if needed
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
        
        pos_dist = (obj_pos - demo_pos).norm().item()
        rot_dist = rotation_distance(demo_quat.unsqueeze(0), obj_quat.unsqueeze(0)).item()
        
        # Track articulation distance if applicable
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
        
        # Track residual magnitude (how much MPC corrected the policy)
        residual_norm = (final_action - policy_action).norm().item()
        metrics["residual_norm"].append(residual_norm)
        
        t1 = time.perf_counter()
        rtr = env.dt / (t1 - t0) if (t1 - t0) > 0 else 0
        
        print(
            f"Step {step+1}/{max_steps} | "
            f"pos: {pos_dist:.4f}, rot: {rot_dist:.4f}, arti: {arti_dist:.4f} | "
            f"rew: {rewards.max().item():.3f} | "
            f"res: {residual_norm:.3f} | "
            f"RTR: {rtr:.2f}x   ",
            end="\r"
        )
    
    t_end = time.perf_counter()
    print(f"\n\nTotal time: {t_end - t_start:.2f}s")
    
    # Print summary statistics
    pos_dists = np.array(metrics["pos_dist"])
    rot_dists = np.array(metrics["rot_dist"])
    arti_dists = np.array(metrics["arti_dist"])
    residual_norms = np.array(metrics["residual_norm"])
    print(f"Position distance: {pos_dists.mean():.4f} ± {pos_dists.std():.4f}")
    print(f"Rotation distance: {rot_dists.mean():.4f} ± {rot_dists.std():.4f}")
    print(f"Articulation distance: {arti_dists.mean():.4f} ± {arti_dists.std():.4f}")
    print(f"Residual norm (MPC correction): {residual_norms.mean():.4f} ± {residual_norms.std():.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.npz")
    np.savez(metrics_path, **{k: np.array(v) for k, v in metrics.items()})
    print(f"Saved metrics to {metrics_path}")
    
    # Save video if recorded
    if args.record_video and hasattr(env, '_recorded_frames') and len(env._recorded_frames) > 0:
        # Build descriptive filename with MPC config and reward weights
        task_w = env.reward_module.task_rew_weight
        imi_w = env.reward_module.imi_rew_weight
        con_w = env.reward_module.contact_rew_weight
        fast_mode = args.fast_mode
        fname = (f"dial_mpc_N{args.num_samples}_H{args.horizon}_I{args.num_iterations}"
                 f"_task{task_w}_imi{imi_w}_con{con_w}_fast{fast_mode}_kp{args.kp_max}_kv{args.kv_max}.mp4")
        video_path = os.path.join(args.output_dir, fname)
        env.export_video(video_path, wait_for_max=False)
        print(f"Saved video to {video_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
