import os
import tempfile
import unittest

import numpy as np
import torch

from dexmachina.envs.curriculum import Curriculum, get_curriculum_cfg
from dexmachina.envs import demo_data as demo_data_module
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.virtual_force import compute_virtual_force


class _DummyObject:
    def __init__(self):
        self.calls = []

    def set_joint_gains(self, kp=None, kv=None, force_range=None, env_idxs=None):
        self.calls.append((kp, kv, force_range, env_idxs))


class VirtualForceAssistTests(unittest.TestCase):
    def test_compute_virtual_force_points_inward_and_clips(self):
        link_pos = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        link_vel = torch.zeros_like(link_pos)
        contact_pos = torch.tensor([[[0.01, 0.0, 0.0]]], dtype=torch.float32)
        contact_normal = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
        valid_mask = torch.tensor([[True]])

        assist = compute_virtual_force(
            link_pos=link_pos,
            link_vel=link_vel,
            contact_pos=contact_pos,
            contact_normal=contact_normal,
            valid_mask=valid_mask,
            alpha=1.0,
            delta=0.001,
            kp=100.0,
            kd=0.0,
            sigma=0.05,
            fmax=0.25,
        )
        self.assertEqual(tuple(assist.shape), (1, 1, 3))
        self.assertGreater(float(assist[0, 0, 0]), 0.0)
        self.assertLessEqual(float(torch.norm(assist[0, 0])), 0.25001)

    def test_curriculum_decays_virtual_force_gain(self):
        cfg = get_curriculum_cfg(
            dict(
                kp_init=80.0,
                kv_init=5.0,
                force_range_init=50.0,
                gravity_init=1.0,
                virtual_force_init=1.0,
                wait_epochs=0,
                interval=1,
                schedule="fixed",
                fixed_mode="exp",
                gain_mode="all",
                upper_ratios=dict(kp=0.9, kv=0.9, fr=0.9, gravity=0.95, vf=0.5),
                lower_ratios=dict(kp=0.8, kv=0.8, fr=0.8, gravity=0.9, vf=0.5),
            )
        )
        dummy = _DummyObject()
        curriculum = Curriculum(
            cfg,
            task_object=dummy,
            reward_keys=["task"],
            num_envs=1,
            achieved_length=10,
            max_episode_length=10,
        )
        zero_gains, gains_decayed, _ = curriculum.set_curriculum(1)
        self.assertFalse(zero_gains)
        self.assertTrue(gains_decayed)
        self.assertAlmostEqual(curriculum.get_current_gains()["vf"], 0.5, places=6)

    def test_curriculum_decays_virtual_force_gain_repeatedly(self):
        cfg = get_curriculum_cfg(
            dict(
                kp_init=80.0,
                kv_init=5.0,
                force_range_init=50.0,
                gravity_init=1.0,
                virtual_force_init=1.0,
                wait_epochs=0,
                interval=1,
                schedule="fixed",
                fixed_mode="exp",
                gain_mode="all",
                upper_ratios=dict(kp=0.9, kv=0.9, fr=0.9, gravity=0.95, vf=0.5),
                lower_ratios=dict(kp=0.8, kv=0.8, fr=0.8, gravity=0.9, vf=0.5),
            )
        )
        dummy = _DummyObject()
        curriculum = Curriculum(
            cfg,
            task_object=dummy,
            reward_keys=["task"],
            num_envs=1,
            achieved_length=10,
            max_episode_length=10,
        )

        expected = [0.5, 0.25, 0.125]
        for epoch, vf_expected in zip([1, 41, 81], expected):
            zero_gains, gains_decayed, _ = curriculum.set_curriculum(epoch)
            self.assertFalse(zero_gains)
            self.assertTrue(gains_decayed)
            self.assertAlmostEqual(curriculum.get_current_gains()["vf"], vf_expected, places=6)

    def test_curriculum_zero_epoch_zeros_virtual_force_gain(self):
        cfg = get_curriculum_cfg(
            dict(
                kp_init=80.0,
                kv_init=5.0,
                force_range_init=50.0,
                gravity_init=1.0,
                virtual_force_init=1.0,
                wait_epochs=0,
                interval=1,
                schedule="fixed",
                fixed_mode="exp",
                gain_mode="all",
                zero_epoch=2,
                upper_ratios=dict(kp=0.9, kv=0.9, fr=0.9, gravity=0.95, vf=0.5),
                lower_ratios=dict(kp=0.8, kv=0.8, fr=0.8, gravity=0.9, vf=0.5),
            )
        )
        dummy = _DummyObject()
        curriculum = Curriculum(
            cfg,
            task_object=dummy,
            reward_keys=["task"],
            num_envs=1,
            achieved_length=10,
            max_episode_length=10,
        )

        curriculum.set_curriculum(1)
        zero_gains, gains_decayed, _ = curriculum.set_curriculum(3)
        self.assertFalse(zero_gains)
        self.assertTrue(gains_decayed)
        self.assertEqual(curriculum.get_current_gains()["vf"], 0.0)

    def test_contact_loader_keeps_local_targets_and_normals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = demo_data_module.RETARGET_CONTACT_DIR
            demo_data_module.RETARGET_CONTACT_DIR = os.path.join(tmpdir, "contact_retarget")
            try:
                base = os.path.join(demo_data_module.RETARGET_CONTACT_DIR, "orca_hand", "s01")
                os.makedirs(base, exist_ok=True)
                fname = os.path.join(base, "ketchup_use_01.npy")
                sample = dict(
                    left=dict(
                        dexlink_contacts=np.zeros((4, 2, 3, 4), dtype=np.float32),
                        dexlink_valid_contacts=np.ones((4, 2, 3), dtype=np.float32),
                        dexlink_contacts_local=np.ones((4, 2, 3, 3), dtype=np.float32),
                        dexlink_contact_normals_local=np.repeat(
                            np.array([[[[0.0, 0.0, 1.0]]]], dtype=np.float32), 4 * 2 * 3, axis=0
                        ).reshape(4, 2, 3, 3),
                        collision_link_names=["left_thumb_tip", "left_index_tip", "left_middle_tip"],
                        collision_link_local_idxs=[1, 2, 3],
                        object_part_names=["top", "bottom"],
                    ),
                    right=dict(
                        dexlink_contacts=np.zeros((4, 2, 3, 4), dtype=np.float32),
                        dexlink_valid_contacts=np.ones((4, 2, 3), dtype=np.float32),
                        dexlink_contacts_local=np.ones((4, 2, 3, 3), dtype=np.float32) * 2.0,
                        dexlink_contact_normals_local=np.repeat(
                            np.array([[[[0.0, 1.0, 0.0]]]], dtype=np.float32), 4 * 2 * 3, axis=0
                        ).reshape(4, 2, 3, 3),
                        collision_link_names=["right_thumb_tip", "right_index_tip", "right_middle_tip"],
                        collision_link_local_idxs=[4, 5, 6],
                        object_part_names=["top", "bottom"],
                    ),
                )
                np.save(fname, sample)
                loaded = demo_data_module.load_contact_retarget_data(
                    obj_name="ketchup",
                    hand_name="orca_hand",
                    frame_start=1,
                    frame_end=3,
                    use_clip="01",
                    subject_name="s01",
                )
                self.assertIn("contact_links_local_left", loaded)
                self.assertIn("contact_normals_local_right", loaded)
                self.assertEqual(tuple(loaded["contact_links_local_left"].shape), (2, 2, 3, 3))
                self.assertEqual(loaded["right"]["object_part_names"], ["top", "bottom"])
            finally:
                demo_data_module.RETARGET_CONTACT_DIR = orig_dir

    def test_ketchup_mesh_query_returns_unit_normals(self):
        try:
            cfg = get_arctic_object_cfg("ketchup")
        except AssertionError as exc:
            self.skipTest(str(exc))
        obj = ArticulatedObject.__new__(ArticulatedObject)
        obj.cfg = cfg
        obj._part_surface_meshes = None

        import trimesh

        mesh = trimesh.load(cfg["top_mesh_fname"], force="mesh", process=False)
        query_points = np.asarray(mesh.vertices[:5], dtype=np.float32)
        closest, normals = obj.query_part_surface_local("top", query_points)
        self.assertEqual(closest.shape, (5, 3))
        self.assertEqual(normals.shape, (5, 3))
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
