import unittest
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]

_DEXMACHINA_ENVS_PKG = types.ModuleType("dexmachina.envs")
_DEXMACHINA_ENVS_PKG.__path__ = [str(_ROOT / "envs")]
sys.modules.setdefault("dexmachina.envs", _DEXMACHINA_ENVS_PKG)

_REWARD_UTILS_SPEC = importlib.util.spec_from_file_location(
    "dexmachina.envs.reward_utils",
    _ROOT / "envs" / "reward_utils.py",
)
_REWARD_UTILS = importlib.util.module_from_spec(_REWARD_UTILS_SPEC)
assert _REWARD_UTILS_SPEC.loader is not None
sys.modules["dexmachina.envs.reward_utils"] = _REWARD_UTILS
_REWARD_UTILS_SPEC.loader.exec_module(_REWARD_UTILS)

_REWARDS_SPEC = importlib.util.spec_from_file_location("rewards_module", _ROOT / "envs" / "rewards.py")
_REWARDS = importlib.util.module_from_spec(_REWARDS_SPEC)
assert _REWARDS_SPEC.loader is not None
_REWARDS_SPEC.loader.exec_module(_REWARDS)

RewardModule = _REWARDS.RewardModule
get_reward_cfg = _REWARDS.get_reward_cfg


class SingleHandLoggingTests(unittest.TestCase):
    def test_imitation_reward_omits_inactive_hand_metrics(self):
        ep_len = 4
        reward_cfg = get_reward_cfg()
        reward_cfg["task_rew_weight"] = 0.0
        reward_cfg["imi_rew_weight"] = 1.0
        reward_cfg["imi_wrist_weight"] = 0.0
        reward_cfg["contact_rew_weight"] = 0.0
        reward_cfg["bc_rew_weight"] = 0.0

        demo_data = {
            "obj_pos": np.zeros((ep_len, 3), dtype=np.float32),
            "obj_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (ep_len, 1)),
            "obj_arti": np.zeros(ep_len, dtype=np.float32),
        }
        retarget_data = {
            "right": {
                "kpts_data": {
                    "kpt_pos": np.zeros((ep_len, 5, 3), dtype=np.float32),
                }
            }
        }

        reward_module = RewardModule(reward_cfg, demo_data, retarget_data, device="cpu")
        imi_rew, rew_dict = reward_module.compute_imitation_reward(
            wrist_pose_left=None,
            wrist_pose_right=None,
            kpts_left=None,
            kpts_right=torch.zeros((2, 5, 3), dtype=torch.float32),
            episode_length_buf=torch.tensor([0, 1], dtype=torch.long),
        )

        self.assertEqual(list(reward_module.active_sides), ["right"])
        self.assertEqual(tuple(imi_rew.shape), (2,))
        self.assertIn("kpts_dist_right", rew_dict)
        self.assertNotIn("kpts_dist_left", rew_dict)
        self.assertIsNotNone(rew_dict["kpts_dist_right"])

if __name__ == "__main__":
    unittest.main()
