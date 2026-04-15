"""
Tests for DexYCB single-hand environment support.

Covers the three merge-conflict resolutions:
  1. base_env.py  : try/except around object_mass_buffer setup
  2. rewards.py   : active_sides guard skips the inactive hand in the contact loop
  3. object.py    : import of points_world_to_local_np is preserved

Unit tests (no genesis required) are always run.
Integration test requires genesis + ARCTIC box assets (skipped otherwise).

Run with:
  python -m dexmachina.tests.test_dexycb_single_hand
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import genesis as gs
    _HAS_GENESIS = True
except ImportError:
    _HAS_GENESIS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_demo(ep_len, sides=("right",), n_hand_links=16, n_obj_parts=None):
    """Build a minimal demo_data dict.

    n_obj_parts=None  -> flat DexYCB-style (ep_len, n_hand_links, 4)
    n_obj_parts=int   -> ARCTIC-style       (ep_len, n_obj_parts, n_hand_links, 4)
    """
    demo_data = {
        "obj_pos": np.zeros((ep_len, 3), dtype=np.float32),
        "obj_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (ep_len, 1)),
        "obj_arti": np.zeros(ep_len, dtype=np.float32),
    }
    for side in sides:
        if n_obj_parts is None:
            shape = (ep_len, n_hand_links, 4)
        else:
            shape = (ep_len, n_obj_parts, n_hand_links, 4)
        demo_data[f"contact_links_{side}"] = np.zeros(shape, dtype=np.float32)
    return demo_data


# ---------------------------------------------------------------------------
# Unit tests  (no genesis needed)
# ---------------------------------------------------------------------------

class TestActiveSidesInference(unittest.TestCase):
    """RewardModule should infer active_sides purely from demo_data keys."""

    def _make_reward_module(self, demo_data):
        from dexmachina.envs.rewards import RewardModule, get_reward_cfg

        cfg = get_reward_cfg()
        cfg["contact_rew_weight"] = 0.0
        cfg["imi_rew_weight"] = 0.0
        cfg["task_rew_weight"] = 1.0
        return RewardModule(cfg, demo_data, retarget_data={}, device=torch.device("cpu"))

    def test_right_only_demo_gives_right_active_side(self):
        """Only contact_links_right present -> active_sides == ['right']."""
        demo = _make_minimal_demo(ep_len=10, sides=("right",))
        rm = self._make_reward_module(demo)
        self.assertEqual(rm.active_sides, ["right"])

    def test_left_only_demo_gives_left_active_side(self):
        """Only contact_links_left present -> active_sides == ['left']."""
        demo = _make_minimal_demo(ep_len=10, sides=("left",))
        rm = self._make_reward_module(demo)
        self.assertEqual(rm.active_sides, ["left"])

    def test_both_sides_demo(self):
        """Both sides present -> active_sides == ['left', 'right']."""
        demo = _make_minimal_demo(ep_len=10, sides=("left", "right"))
        rm = self._make_reward_module(demo)
        self.assertEqual(sorted(rm.active_sides), ["left", "right"])

    def test_no_contact_keys_falls_back_to_both(self):
        """No contact keys at all -> falls back to ['left', 'right']."""
        demo = {
            "obj_pos": np.zeros((10, 3), dtype=np.float32),
            "obj_quat": np.tile([1, 0, 0, 0], (10, 1)).astype(np.float32),
            "obj_arti": np.zeros(10, dtype=np.float32),
        }
        from dexmachina.envs.rewards import RewardModule, get_reward_cfg

        cfg = get_reward_cfg()
        cfg["contact_rew_weight"] = 0.0
        cfg["imi_rew_weight"] = 0.0
        rm = RewardModule(cfg, demo, retarget_data={}, device=torch.device("cpu"))
        self.assertEqual(sorted(rm.active_sides), ["left", "right"])


class TestContactRewardSkipsInactiveSide(unittest.TestCase):
    """compute_contact_reward_matched must skip sides not in active_sides."""

    def _make_reward_module_with_contact(self, demo_data, device, sides=("right",)):
        from dexmachina.envs.rewards import RewardModule, get_reward_cfg

        ep_len = demo_data["obj_pos"].shape[0]
        retarget_data = {
            side: {"wrist_pose": np.tile([0, 0, 0.3, 1, 0, 0, 0], (ep_len, 1)).astype(np.float32)}
            for side in sides
        }
        cfg = get_reward_cfg()
        cfg["contact_rew_weight"] = 1.0
        cfg["imi_rew_weight"] = 0.0
        cfg["task_rew_weight"] = 0.0
        cfg["bc_rew_weight"] = 0.0
        cfg["use_retarget_contact"] = False
        cfg["retarget_objframe"] = False
        return RewardModule(cfg, demo_data, retarget_data=retarget_data, device=device)

    def test_none_contacts_for_inactive_side_no_crash(self):
        """Passing None for the left-hand contacts must not raise when left is inactive."""
        device = torch.device("cpu")
        ep_len = 4
        n_links = 16
        demo = _make_minimal_demo(ep_len, sides=("right",), n_obj_parts=2, n_hand_links=n_links)
        rm = self._make_reward_module_with_contact(demo, device, sides=("right",))

        self.assertEqual(rm.active_sides, ["right"])

        # Minimal tensors shaped (N=1, 2, n_links, 4) for right; None for left
        N = 1
        dummy_contacts = torch.zeros(N, 2, n_links, 4)
        dummy_valid = torch.ones(N, 2, n_links, 1, dtype=torch.bool)
        dummy_pose = torch.tensor([[0, 0, 0.3, 1, 0, 0, 0]], dtype=torch.float32)
        ep_buf = torch.zeros(N, dtype=torch.long)

        # mock match_demo_state so it returns zeros of the right shape
        def _mock_match(key, ep_buf):
            if "contact_links_right" in key:
                return torch.zeros(N, 2, n_links, 4)
            if "wrist_pose_right" in key:
                return dummy_pose
            return torch.zeros(N, 2, n_links, 4)

        rm.match_demo_state = _mock_match

        con_rew, rews = rm.compute_matched_contact_reward(
            contacts_link_left=None,
            contacts_link_valid_left=None,
            contacts_link_right=dummy_contacts,
            contacts_link_valid_right=dummy_valid,
            obj_pose=dummy_pose,
            demo_obj_pose=dummy_pose,
            episode_length_buf=ep_buf,
        )
        # Left side was skipped; right side processed; reward tensor present
        self.assertIsNotNone(con_rew)
        self.assertFalse(torch.isnan(con_rew).any(), "contact reward contains NaN")
        self.assertNotIn("conrew_left_bottom", rews)
        self.assertIn("conrew_right_bottom", rews)


# ---------------------------------------------------------------------------
# Integration test (genesis + assets required)
# ---------------------------------------------------------------------------

def _skip_if_no_genesis():
    if not _HAS_GENESIS:
        raise unittest.SkipTest("genesis not installed")


def _arctic_box_urdf_exists():
    try:
        from dexmachina.envs.object import get_arctic_object_cfg
        cfg = get_arctic_object_cfg(name="box")
        return os.path.exists(cfg["urdf_path"])
    except Exception:
        return False


class TestDexYCBSingleHandEnvGenesis(unittest.TestCase):
    """Full-env test simulating DexYCB single-hand data layout."""

    @classmethod
    def setUpClass(cls):
        _skip_if_no_genesis()
        if not _arctic_box_urdf_exists():
            raise unittest.SkipTest("ARCTIC box assets not found")
        gs.init(backend=gs.cpu)

    def _build_env(self, ep_len=6, num_envs=2):
        from dexmachina.envs.base_env import BaseEnv, get_env_cfg
        from dexmachina.envs.robot import get_default_robot_cfg
        from dexmachina.envs.object import get_arctic_object_cfg
        from dexmachina.envs.rewards import get_reward_cfg

        device = torch.device("cpu")

        env_cfg = get_env_cfg(use_visualizer=False, show_viewer=False)
        env_cfg["num_envs"] = num_envs
        env_cfg["episode_length"] = ep_len
        env_cfg["use_contact_reward"] = False
        env_cfg["observe_tip_dist"] = False
        env_cfg["observe_contact_force"] = False
        env_cfg["observe_hand_demo_diff"] = False
        env_cfg["traj_lookahead_frames"] = 0

        robot_cfgs = {
            "right": get_default_robot_cfg(name="orca_hand", side="right"),
        }
        robot_cfgs["right"]["action_mode"] = "absolute"
        robot_cfgs["right"]["collect_data"] = False

        obj_name = "box"
        object_cfgs = {obj_name: get_arctic_object_cfg(name=obj_name, convexify=False)}
        object_cfgs[obj_name]["fixed"] = False

        # DexYCB-style flat contact: (ep_len, n_hand_links=16, 4) — no obj_parts dim
        demo_data = {
            "obj_pos": np.zeros((ep_len, 3), dtype=np.float32),
            "obj_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (ep_len, 1)),
            "obj_arti": np.zeros(ep_len, dtype=np.float32),
            "contact_links_right": np.zeros((ep_len, 16, 4), dtype=np.float32),
            "ycb_class_name": "002_master_chef_can",
        }

        reward_cfg = get_reward_cfg()
        reward_cfg["task_rew_weight"] = 1.0
        reward_cfg["imi_rew_weight"] = 0.0
        reward_cfg["contact_rew_weight"] = 0.0
        reward_cfg["bc_rew_weight"] = 0.0

        retarget_data = {"right": {"num_frames": ep_len}}

        return BaseEnv(
            env_cfg=env_cfg,
            robot_cfgs=robot_cfgs,
            object_cfgs=object_cfgs,
            reward_cfg=reward_cfg,
            demo_data=demo_data,
            retarget_data=retarget_data,
            device=device,
            rand_cfg={},
            curriculum_cfg={},
        ), num_envs

    def test_single_hand_dexycb_style_init(self):
        env, num_envs = self._build_env()

        # active_sides inferred from contact_links_right only
        self.assertEqual(env.active_sides, ["right"])
        self.assertEqual(list(env.robots.keys()), ["right"])
        self.assertNotIn("left", env.robots)

    def test_obs_shape_consistent(self):
        env, num_envs = self._build_env()
        obs = env.get_observations()
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        self.assertEqual(obs.shape[0], num_envs)
        self.assertEqual(obs.shape[1], env.obs_dim)

    def test_object_mass_buffer_try_except(self):
        """object_mass_buffer should be a tensor or None — never an uncaught error."""
        env, _ = self._build_env()
        # The try/except in post_scene_build_setup must have run without raising
        self.assertTrue(
            env.object_mass_buffer is None or isinstance(env.object_mass_buffer, torch.Tensor),
            f"unexpected type: {type(env.object_mass_buffer)}",
        )

    def test_reset_and_step(self):
        env, num_envs = self._build_env()
        env.reset_idx(list(range(num_envs)))
        action = torch.zeros((num_envs, env.action_dim))
        env.step(action)

    def test_step_produces_finite_reward(self):
        env, num_envs = self._build_env()
        action = torch.zeros((num_envs, env.action_dim))
        _, rew, _, _, _ = env.step(action)
        self.assertFalse(torch.isnan(rew).any(), "reward contains NaN")
        self.assertFalse(torch.isinf(rew).any(), "reward contains Inf")


if __name__ == "__main__":
    # Run unit tests without genesis; genesis tests are skipped automatically.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestActiveSidesInference))
    suite.addTests(loader.loadTestsFromTestCase(TestContactRewardSkipsInactiveSide))
    suite.addTests(loader.loadTestsFromTestCase(TestDexYCBSingleHandEnvGenesis))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
