#!/usr/bin/env python3
"""Visualize short post-reset rollouts for random-init episodes.

Default setup matches the repo's Orca-hand ketchup clip.
"""

import argparse
from copy import deepcopy
import math
import os
import pickle
import random
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.constructors import parse_clip_string
from dexmachina.envs.curriculum import get_curriculum_cfg
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
from dexmachina.envs.object import get_arctic_object_cfg
from dexmachina.envs.rewards import get_reward_cfg
from dexmachina.envs.robot import get_default_robot_cfg
from dexmachina.rl.rl_games_wrapper import RlGamesGpuEnv, RlGamesVecEnvWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record the first few steps after random-init resets."
    )
    parser.add_argument("--clip", type=str, default="ketchup-30-130-s01-u01")
    parser.add_argument("--hand", type=str, default="orca_hand")
    parser.add_argument("--retarget_name", type=str, default="para")
    parser.add_argument("--rand_init_ratio", type=float, default=1.0)
    parser.add_argument("--num_resets", type=int, default=20)
    parser.add_argument("--steps_after_reset", type=int, default=50)
    parser.add_argument("--max_reset_attempts", type=int, default=20)
    parser.add_argument(
        "--include_nonrandom_resets",
        action="store_true",
        help="Also record resets that land at demo step 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_vis/ketchup_orca_rand_init_resets.mp4"),
    )
    parser.add_argument("--render_camera", type=str, default="front")
    parser.add_argument(
        "--action_mode",
        type=str,
        default=None,
        choices=["residual", "absolute", "relative", "hybrid", "kinematic", "policy_residual"],
    )
    parser.add_argument("--hybrid_scales", type=float, nargs=2, default=None)
    parser.add_argument("--res_cap", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--stochastic_policy", action="store_true")
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument("--vis_contact", action="store_true")
    parser.add_argument("--enforce_valid_rand_init_grasp", action="store_true")
    parser.add_argument("--valid_grasp_contact_thresh", type=float, default=0.01)
    parser.add_argument("--valid_grasp_opt_samples", type=int, default=12)
    parser.add_argument("--valid_grasp_resample_attempts", type=int, default=8)
    parser.add_argument("--obj_kp", type=float, default=80.0)
    parser.add_argument("--obj_kv", type=float, default=5.0)
    parser.add_argument("--obj_force_range", type=float, default=100.0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_saved_env_kwargs(checkpoint):
    if checkpoint is None:
        return None
    checkpoint_path = Path(checkpoint).resolve()
    saved_cfg_fname = checkpoint_path.parents[1] / "params" / "env.pkl"
    if not saved_cfg_fname.exists():
        print(f"Warning: could not find saved env config at {saved_cfg_fname}")
        return None
    with open(saved_cfg_fname, "rb") as f:
        return pickle.load(f)


def load_saved_robot_cfg(saved_env_kwargs):
    if not saved_env_kwargs:
        return {}
    left_cfg = saved_env_kwargs.get("robot_cfgs", {}).get("left", {})
    return {
        "action_mode": left_cfg.get("action_mode"),
        "hybrid_scales": tuple(left_cfg.get("hybrid_scales", (0.04, 0.5))),
        "res_cap": bool(left_cfg.get("res_cap", False)),
    }


def build_env(args, saved_env_kwargs=None, saved_robot_cfg=None):
    obj_name, frame_start, frame_end, subject_name, use_clip = parse_clip_string(args.clip)
    episode_length = frame_end - frame_start
    saved_robot_cfg = saved_robot_cfg or {}
    if saved_env_kwargs:
        env_kwargs = deepcopy(saved_env_kwargs)
        env_kwargs["env_cfg"]["num_envs"] = 1
        env_kwargs["env_cfg"]["is_eval"] = True
        env_kwargs["env_cfg"]["record_video"] = True
        env_kwargs["env_cfg"]["max_video_frames"] = args.steps_after_reset
        env_kwargs["env_cfg"]["rand_init_ratio"] = args.rand_init_ratio
        env_kwargs["env_cfg"]["render_camera"] = args.render_camera
        env_kwargs["env_cfg"]["early_reset_threshold"] = 0.0
        env_kwargs["env_cfg"]["early_reset_aux_thres"] = dict(con=0.0, imi=0.0, bc=0.0)
        env_kwargs["env_cfg"]["enforce_valid_rand_init_grasp"] = args.enforce_valid_rand_init_grasp
        env_kwargs["env_cfg"]["valid_grasp_contact_thresh"] = args.valid_grasp_contact_thresh
        env_kwargs["env_cfg"]["valid_grasp_opt_samples"] = args.valid_grasp_opt_samples
        env_kwargs["env_cfg"]["valid_grasp_resample_attempts"] = args.valid_grasp_resample_attempts
        env_kwargs["env_cfg"]["scene_kwargs"]["use_visualizer"] = True
        env_kwargs["env_cfg"]["scene_kwargs"]["show_viewer"] = args.show_viewer
        env_kwargs["env_cfg"]["scene_kwargs"]["visualize_contact"] = args.vis_contact
        env_kwargs["env_cfg"]["camera_kwargs"][args.render_camera]["res"] = (args.width, args.height)
        if "rand_cfg" in env_kwargs and isinstance(env_kwargs["rand_cfg"], dict):
            env_kwargs["rand_cfg"]["randomize"] = False
        action_mode = args.action_mode or saved_robot_cfg.get("action_mode")
        hybrid_scales = (
            tuple(args.hybrid_scales)
            if args.hybrid_scales is not None
            else saved_robot_cfg.get("hybrid_scales")
        )
        res_cap = args.res_cap or saved_robot_cfg.get("res_cap", False)
        for cfg in env_kwargs.get("robot_cfgs", {}).values():
            if action_mode is not None:
                cfg["action_mode"] = action_mode
            if hybrid_scales is not None:
                cfg["hybrid_scales"] = hybrid_scales
            cfg["res_cap"] = res_cap
        env_kwargs["device"] = torch.device("cuda")
        return BaseEnv(**env_kwargs)

    env_cfg = get_env_cfg(
        use_visualizer=True,
        show_viewer=args.show_viewer,
        show_fps=False,
    )
    env_cfg["num_envs"] = 1
    env_cfg["episode_length"] = episode_length
    env_cfg["is_eval"] = True
    env_cfg["use_rl_games"] = False
    env_cfg["record_video"] = True
    env_cfg["max_video_frames"] = args.steps_after_reset
    env_cfg["rand_init_ratio"] = args.rand_init_ratio
    env_cfg["render_camera"] = args.render_camera
    env_cfg["enforce_valid_rand_init_grasp"] = args.enforce_valid_rand_init_grasp
    env_cfg["valid_grasp_contact_thresh"] = args.valid_grasp_contact_thresh
    env_cfg["valid_grasp_opt_samples"] = args.valid_grasp_opt_samples
    env_cfg["valid_grasp_resample_attempts"] = args.valid_grasp_resample_attempts
    env_cfg["scene_kwargs"]["use_visualizer"] = True
    env_cfg["scene_kwargs"]["show_viewer"] = args.show_viewer
    env_cfg["scene_kwargs"]["visualize_contact"] = args.vis_contact
    env_cfg["scene_kwargs"]["batch_dofs_info"] = True
    env_cfg["camera_kwargs"][args.render_camera]["res"] = (args.width, args.height)

    demo_data = get_demo_data(
        obj_name=obj_name,
        frame_start=frame_start,
        frame_end=frame_end,
        hand_name=args.hand,
        subject_name=subject_name,
        use_clip=use_clip,
    )
    _, retarget_data = load_genesis_retarget_data(
        obj_name=obj_name,
        hand_name=args.hand,
        frame_start=frame_start,
        frame_end=frame_end,
        save_name=args.retarget_name,
        use_clip=use_clip,
        subject_name=subject_name,
    )

    robot_cfgs = {
        side: get_default_robot_cfg(name=args.hand, side=side)
        for side in ("left", "right")
    }
    action_mode = args.action_mode or saved_robot_cfg.get("action_mode") or "hybrid"
    hybrid_scales = (
        tuple(args.hybrid_scales)
        if args.hybrid_scales is not None
        else tuple(saved_robot_cfg.get("hybrid_scales", (0.04, 0.5)))
    )
    res_cap = args.res_cap or saved_robot_cfg.get("res_cap", False)
    for cfg in robot_cfgs.values():
        cfg["action_mode"] = action_mode
        cfg["hybrid_scales"] = hybrid_scales
        cfg["res_cap"] = res_cap
    object_cfg = get_arctic_object_cfg(name=obj_name)
    object_cfg["actuated"] = True
    object_cfg["kp"] = args.obj_kp
    object_cfg["kv"] = args.obj_kv
    object_cfg["force_range"] = args.obj_force_range
    curriculum_cfg = get_curriculum_cfg(
        dict(
            kp_init=args.obj_kp,
            kv_init=args.obj_kv,
            force_range_init=args.obj_force_range,
        )
    )

    return BaseEnv(
        env_cfg=env_cfg,
        robot_cfgs=robot_cfgs,
        object_cfgs={obj_name: object_cfg},
        reward_cfg=get_reward_cfg(),
        demo_data=demo_data,
        retarget_data=retarget_data,
        curriculum_cfg=curriculum_cfg,
        device=torch.device("cuda"),
        visualize_contact=args.vis_contact,
        render_figure=False,
    )


def draw_text(frame, text, origin):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def format_action_summary(actions):
    flat = actions[0].detach().cpu().numpy()
    head = ", ".join(f"{x:+.2f}" for x in flat[: min(6, flat.shape[0])])
    return dict(
        mean_abs=float(np.mean(np.abs(flat))),
        max_abs=float(np.max(np.abs(flat))),
        first_dims=head,
    )


def annotate_frame(frame, clip_name, reset_idx, step_idx, episode_start, action_summary=None, action_source="zero"):
    rgb = np.clip(frame[:, :, :3], 0, 255).astype(np.uint8).copy()
    draw_text(rgb, f"{clip_name}", (18, 28))
    draw_text(rgb, f"reset {reset_idx + 1}  demo_start={episode_start}", (18, 56))
    draw_text(rgb, f"step {step_idx + 1}", (18, 84))
    draw_text(rgb, f"actions: {action_source}", (18, 112))
    if action_summary is not None:
        draw_text(
            rgb,
            f"|a|mean={action_summary['mean_abs']:.3f}  |a|max={action_summary['max_abs']:.3f}",
            (18, 140),
        )
        draw_text(rgb, f"a[:6]=[{action_summary['first_dims']}]", (18, 168))
    return rgb


def build_policy_driver(env, checkpoint):
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    wrapped_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})
    agent_cfg["params"]["config"]["num_actors"] = wrapped_env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(os.path.abspath(checkpoint))
    agent.reset()
    return wrapped_env, agent


def record_rollouts(env, args, policy_env=None, agent=None):
    if args.rand_init_ratio <= 0.0:
        raise ValueError("rand_init_ratio must be > 0.0 for this script.")

    zero_actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    use_policy = agent is not None and policy_env is not None
    accepted_resets = 0
    attempts = 0
    video_frames = []

    while accepted_resets < args.num_resets and attempts < args.max_reset_attempts:
        attempts += 1
        if use_policy:
            obs = policy_env.reset()
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
        else:
            env.reset()
            obs = None
        episode_start = int(env.episode_start_buf[0].item())
        is_random_start = episode_start > 0
        should_record = args.include_nonrandom_resets or is_random_start
        status = "record" if should_record else "skip"
        print(
            f"reset_attempt={attempts} demo_start={episode_start} "
            f"random_start={is_random_start} action={status}"
        )
        if not should_record:
            continue

        env.max_video_frames = args.steps_after_reset
        env.start_recording()
        action_summaries = []
        with torch.inference_mode():
            for _ in range(args.steps_after_reset):
                if use_policy:
                    actions = agent.get_action(obs, is_deterministic=not args.stochastic_policy)
                    action_summary = format_action_summary(actions)
                    print(
                        f"reset={accepted_resets + 1} step={len(action_summaries) + 1} "
                        f"|a|mean={action_summary['mean_abs']:.4f} "
                        f"|a|max={action_summary['max_abs']:.4f} "
                        f"a[:6]=[{action_summary['first_dims']}]"
                    )
                    obs, rew, dones, infos = policy_env.step(actions)
                    if agent.is_rnn and agent.states is not None and torch.any(dones):
                        for s in agent.states:
                            s[:, dones, :] = 0.0
                else:
                    actions = zero_actions
                    action_summary = format_action_summary(actions)
                    env.step(actions)
                action_summaries.append(action_summary)

        segment_frames = env.get_recorded_frames(wait_for_max=False) or []
        if len(segment_frames) == 0:
            raise RuntimeError("No frames were recorded. Check visualizer/video setup.")

        for step_idx, frame in enumerate(segment_frames[: args.steps_after_reset]):
            video_frames.append(
                annotate_frame(
                    frame=frame,
                    clip_name=args.clip,
                    reset_idx=accepted_resets,
                    step_idx=step_idx,
                    episode_start=episode_start,
                    action_summary=action_summaries[min(step_idx, len(action_summaries) - 1)],
                    action_source=("policy" if use_policy else "zero"),
                )
            )
        accepted_resets += 1

    if accepted_resets == 0:
        raise RuntimeError("No resets were recorded. Increase max_reset_attempts or rand_init_ratio.")
    if accepted_resets < args.num_resets:
        print(
            f"Recorded {accepted_resets} reset segments after {attempts} attempts "
            f"(requested {args.num_resets})."
        )
    return video_frames


def save_video(frames, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".mp4")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=False,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        logger=None,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    gs.init(backend=gs.gpu, logging_level="warning")
    saved_env_kwargs = load_saved_env_kwargs(args.checkpoint)
    saved_robot_cfg = load_saved_robot_cfg(saved_env_kwargs)
    env = build_env(args, saved_env_kwargs=saved_env_kwargs, saved_robot_cfg=saved_robot_cfg)
    policy_env = None
    agent = None
    if args.checkpoint is not None:
        policy_env, agent = build_policy_driver(env, args.checkpoint)
        print(
            f"Loaded policy checkpoint {args.checkpoint} with action_mode="
            f"{(args.action_mode or saved_robot_cfg.get('action_mode') or 'hybrid')}"
        )
    frames = record_rollouts(env, args, policy_env=policy_env, agent=agent)
    save_video(frames, args.output, args.fps)
    print(f"Saved {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
