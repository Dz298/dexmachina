import os
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

try:
    import genesis as gs
except ImportError:
    gs = None

_MODULE_PATH = Path(__file__).resolve().parents[1] / "envs" / "demo_data.py"
_SPEC = importlib.util.spec_from_file_location("demo_data_module", _MODULE_PATH)
_DEMO_DATA = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_DEMO_DATA)

_infer_hand_sides_from_world_coord = _DEMO_DATA._infer_hand_sides_from_world_coord
resolve_hand_sides = _DEMO_DATA.resolve_hand_sides
_GENESIS_INITIALIZED = False


def _skip_if_no_genesis():
    if gs is None:
        raise unittest.SkipTest("genesis not installed")


def _ensure_genesis_initialized(device):
    global _GENESIS_INITIALIZED
    if _GENESIS_INITIALIZED:
        return
    gs.init(backend=gs.cpu if device.type == "cpu" else gs.gpu)
    _GENESIS_INITIALIZED = True


def _build_env(hand_sides):
    _skip_if_no_genesis()
    from dexmachina.envs.base_env import BaseEnv, get_env_cfg
    from dexmachina.envs.object import get_arctic_object_cfg
    from dexmachina.envs.rewards import get_reward_cfg
    from dexmachina.envs.robot import get_default_robot_cfg

    num_envs = 2
    ep_len = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_genesis_initialized(device)

    env_cfg = get_env_cfg(use_visualizer=False, show_viewer=False)
    env_cfg["num_envs"] = num_envs
    env_cfg["episode_length"] = ep_len
    env_cfg["use_contact_reward"] = False
    env_cfg["observe_tip_dist"] = False
    env_cfg["observe_contact_force"] = False
    env_cfg["observe_hand_demo_diff"] = False
    env_cfg["traj_lookahead_frames"] = 0

    hand_name = "orca_hand"
    robot_cfgs = {}
    for side in hand_sides:
        cfg = get_default_robot_cfg(name=hand_name, side=side)
        cfg["action_mode"] = "absolute"
        cfg["collect_data"] = False
        robot_cfgs[side] = cfg

    object_cfgs = {"box": get_arctic_object_cfg(name="box", convexify=False)}
    object_cfgs["box"]["fixed"] = False

    demo_data = {
        "obj_pos": np.zeros((ep_len, 3), dtype=np.float32),
        "obj_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (ep_len, 1)),
        "obj_arti": np.zeros(ep_len, dtype=np.float32),
    }
    retarget_data = {}
    for side in hand_sides:
        demo_data[f"contact_links_{side}"] = np.zeros((ep_len, 2, 16, 4), dtype=np.float32)
        retarget_data[side] = {"num_frames": ep_len}

    reward_cfg = get_reward_cfg()
    reward_cfg["task_rew_weight"] = 1.0
    reward_cfg["imi_rew_weight"] = 0.0
    reward_cfg["contact_rew_weight"] = 0.0
    reward_cfg["bc_rew_weight"] = 0.0

    env = BaseEnv(
        env_cfg=env_cfg,
        robot_cfgs=robot_cfgs,
        object_cfgs=object_cfgs,
        reward_cfg=reward_cfg,
        demo_data=demo_data,
        retarget_data=retarget_data,
        device=device,
        rand_cfg={},
        curriculum_cfg={},
    )
    return env, device


class DemoDataHandSideTests(unittest.TestCase):
    def test_prefers_moving_side_over_static_nonzero_side(self):
        num_frames = 6
        left = np.zeros((num_frames, 21, 3), dtype=np.float32)
        left[:, 0, 0] = np.linspace(0.0, 0.1, num_frames, dtype=np.float32)
        right = np.full((num_frames, 21, 3), 0.05, dtype=np.float32)

        hand_sides = _infer_hand_sides_from_world_coord(
            {"joints.left": left, "joints.right": right}
        )

        self.assertEqual(hand_sides, ["left"])

    def test_returns_both_sides_when_both_move(self):
        num_frames = 6
        left = np.zeros((num_frames, 21, 3), dtype=np.float32)
        right = np.zeros((num_frames, 21, 3), dtype=np.float32)
        left[:, 0, 0] = np.linspace(0.0, 0.1, num_frames, dtype=np.float32)
        right[:, 0, 1] = np.linspace(0.0, 0.1, num_frames, dtype=np.float32)

        hand_sides = _infer_hand_sides_from_world_coord(
            {"joints.left": left, "joints.right": right}
        )

        self.assertEqual(hand_sides, ["left", "right"])

    def test_resolve_hand_sides_reads_processed_demo_file(self):
        num_frames = 6
        left = np.zeros((num_frames, 21, 3), dtype=np.float32)
        right = np.zeros((num_frames, 21, 3), dtype=np.float32)
        right[:, 0, 2] = np.linspace(0.0, 0.1, num_frames, dtype=np.float32)

        raw = {
            "world_coord": {"joints.left": left, "joints.right": right},
            "params": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "demo.npy")
            np.save(fname, raw)
            hand_sides = resolve_hand_sides(data_fname=fname)

        self.assertEqual(hand_sides, ["right"])

    def test_explicit_hand_side_override_is_preserved(self):
        hand_sides = resolve_hand_sides(hand_sides=["right"], data_fname="/tmp/missing.npy")
        self.assertEqual(hand_sides, ["right"])

    def test_single_hand_env_action_dim_matches_single_robot(self):
        env, device = _build_env(["right"])
        self.assertEqual(env.active_sides, ["right"])
        self.assertEqual(list(env.robots.keys()), ["right"])
        self.assertEqual(env.action_dim, env.robots["right"].get_action_dim())

        obs = env.get_observations()
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("critic"))
        self.assertIsNotNone(obs)
        self.assertEqual(obs.shape[0], env.num_envs)
        self.assertEqual(obs.shape[1], env.obs_dim)

        env.reset_idx(list(range(env.num_envs)))
        env.step(torch.zeros((env.num_envs, env.action_dim), device=device))

    def test_two_hand_env_action_dim_exceeds_single_hand(self):
        env_single, _ = _build_env(["right"])
        env_bimanual, _ = _build_env(["left", "right"])

        expected_bimanual_dim = sum(
            robot.get_action_dim() for robot in env_bimanual.robots.values()
        )
        self.assertEqual(env_bimanual.action_dim, expected_bimanual_dim)
        self.assertGreater(env_bimanual.action_dim, env_single.action_dim)


if __name__ == "__main__":
    unittest.main()
