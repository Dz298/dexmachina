"""
Verify single-hand environment setup: BaseEnv with robot_cfgs={'right': ...}
initializes correctly, obs_dim is consistent, and reward computation runs without error.
Run with: python -m dexmachina.tests.test_single_hand_env (requires genesis and assets).
"""
import os
import sys
import unittest
import numpy as np
import torch

# Add project root so we can run as python -m dexmachina.tests.test_single_hand_env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import genesis as gs
except ImportError:
    gs = None


def _skip_if_no_genesis():
    if gs is None:
        raise unittest.SkipTest("genesis not installed")


def test_single_hand_env_init():
    """Build BaseEnv with only right hand and minimal demo/retarget data; check no crash."""
    _skip_if_no_genesis()
    from dexmachina.envs.base_env import BaseEnv, get_env_cfg
    from dexmachina.envs.robot import get_default_robot_cfg
    from dexmachina.envs.object import get_arctic_object_cfg
    from dexmachina.envs.rewards import get_reward_cfg

    num_envs = 2
    ep_len = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = get_env_cfg(use_visualizer=False, show_viewer=False)
    env_cfg["num_envs"] = num_envs
    env_cfg["episode_length"] = ep_len
    env_cfg["use_contact_reward"] = False
    env_cfg["observe_tip_dist"] = False
    env_cfg["observe_contact_force"] = False
    env_cfg["observe_hand_demo_diff"] = False
    env_cfg["traj_lookahead_frames"] = 0

    hand_name = "orca_hand"
    robot_cfgs = {
        "right": get_default_robot_cfg(name=hand_name, side="right"),
    }
    for side, cfg in robot_cfgs.items():
        cfg["action_mode"] = "absolute"
        cfg["collect_data"] = False

    obj_name = "box"
    object_cfgs = {
        obj_name: get_arctic_object_cfg(name=obj_name, convexify=False),
    }
    object_cfgs[obj_name]["fixed"] = False

    # Presence of contact_links_right makes RewardModule infer active_sides = ['right']
    demo_data = {
        "obj_pos": np.zeros((ep_len, 3), dtype=np.float32),
        "obj_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (ep_len, 1)),
        "obj_arti": np.zeros(ep_len, dtype=np.float32),
        "contact_links_right": np.zeros((ep_len, 2, 16, 4), dtype=np.float32),
    }

    reward_cfg = get_reward_cfg()
    reward_cfg["task_rew_weight"] = 1.0
    reward_cfg["imi_rew_weight"] = 0.0
    reward_cfg["contact_rew_weight"] = 0.0
    reward_cfg["bc_rew_weight"] = 0.0

    # Minimal retarget_data: no kpts_data so robot keeps config kpt_link_names (avoids URDF name mismatch)
    retarget_data = {
        "right": {
            "num_frames": ep_len,
        },
    }

    env_kwargs = {
        "env_cfg": env_cfg,
        "robot_cfgs": robot_cfgs,
        "object_cfgs": object_cfgs,
        "reward_cfg": reward_cfg,
        "demo_data": demo_data,
        "retarget_data": retarget_data,
        "device": device,
        "rand_cfg": {},
        "curriculum_cfg": {},
    }

    gs.init(backend=gs.cpu if device.type == "cpu" else gs.gpu)

    env = BaseEnv(**env_kwargs)
    assert env.active_sides == ["right"]
    assert list(env.robots.keys()) == ["right"]
    assert env.obs_dim > 0
    obs = env.get_observations()
    if isinstance(obs, dict):
        obs = obs.get("policy", obs.get("critic"))
    assert obs is not None and obs.shape[0] == num_envs and obs.shape[1] == env.obs_dim
    env.reset_idx(list(range(num_envs)))
    env.step(torch.zeros((num_envs, env.action_dim), device=device))
    print("test_single_hand_env: OK (init, obs, reset, step)")


if __name__ == "__main__":
    try:
        test_single_hand_env_init()
    except unittest.SkipTest as e:
        print("SKIP:", e)
        sys.exit(0)
