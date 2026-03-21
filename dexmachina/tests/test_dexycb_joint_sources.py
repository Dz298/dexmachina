import os
import unittest

import numpy as np
import yaml

from dexmachina.retargeting.process_dexycb import (
    _reorder_dexycb_joint3d_to_internal,
    _reorder_mano_joints_to_internal,
    _transform_pts_cam_to_world,
    reconstruct_mano_from_pose_m,
    resolve_mano_root,
)


_DEXYCB_TO_INTERNAL_ORDER = np.array(
    [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20],
    dtype=np.int64,
)


class DexYCBJointSourceTests(unittest.TestCase):
    def test_joint3d_reorder_matches_internal_convention(self):
        joints = np.arange(21 * 3, dtype=np.float32).reshape(21, 3)
        reordered = _reorder_dexycb_joint3d_to_internal(joints)
        expected = joints[_DEXYCB_TO_INTERNAL_ORDER]
        np.testing.assert_array_equal(reordered, expected)

    def test_joint3d_and_pose_m_sources_agree_in_world_frame(self):
        try:
            import torch
            from manopth.manolayer import ManoLayer
        except ImportError as exc:
            raise unittest.SkipTest(f"manopth not installed ({exc})")

        dex_ycb_dir = os.environ.get("DEX_YCB_DIR")
        if not dex_ycb_dir or not os.path.isdir(dex_ycb_dir):
            raise unittest.SkipTest("DEX_YCB_DIR is not set to a valid DexYCB dataset root")

        sequence = os.environ.get(
            "DEXYCB_TEST_SEQUENCE",
            "20200709-subject-01/20200709_142211",
        )
        max_frames = int(os.environ.get("DEXYCB_TEST_MAX_FRAMES", "20"))
        max_mean_err = float(os.environ.get("DEXYCB_JOINT3D_MAX_MEAN_ERR", "0.05"))
        max_joint_err = float(os.environ.get("DEXYCB_JOINT3D_MAX_MAX_ERR", "0.12"))

        seq_dir = os.path.join(dex_ycb_dir, sequence)
        if not os.path.isdir(seq_dir):
            raise unittest.SkipTest(f"Sequence not found: {seq_dir}")

        with open(os.path.join(seq_dir, "meta.yml"), "r") as f:
            meta = yaml.safe_load(f)

        extr_file = os.path.join(
            dex_ycb_dir,
            "calibration",
            f"extrinsics_{meta['extrinsics']}",
            "extrinsics.yml",
        )
        with open(extr_file, "r") as f:
            extr = yaml.load(f, Loader=yaml.FullLoader)

        serials = meta["serials"]
        master_serial = extr["master"]
        serial = master_serial if master_serial in serials else serials[0]
        T_serial = np.array(extr["extrinsics"][serial], dtype=np.float32).reshape(3, 4)
        R_c = T_serial[:, :3]
        t_c = T_serial[:, 3]

        mano_calib_id = meta["mano_calib"][0]
        with open(
            os.path.join(dex_ycb_dir, "calibration", f"mano_{mano_calib_id}", "mano.yml"),
            "r",
        ) as f:
            mano_calib = yaml.safe_load(f)
        mano_betas = np.array(mano_calib["betas"], dtype=np.float32)

        mano_root = resolve_mano_root(None)
        mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side=meta["mano_sides"][0],
            mano_root=mano_root,
            use_pca=True,
        ).to(torch.device("cpu"))

        label_dir = os.path.join(seq_dir, serial)
        frame_errs = []
        used_frames = 0
        for frame in range(meta["num_frames"]):
            label_path = os.path.join(label_dir, f"labels_{frame:06d}.npz")
            label = np.load(label_path)

            pose_m = label["pose_m"]
            joint_3d = np.asarray(label["joint_3d"][0], dtype=np.float32)
            if np.all(pose_m == 0.0) or np.all(joint_3d == -1.0):
                continue

            _, joints_pose_m = reconstruct_mano_from_pose_m(
                pose_m, mano_betas, mano_layer, torch.device("cpu")
            )
            joints_world_pose_m = _transform_pts_cam_to_world(joints_pose_m, R_c, t_c)
            joints_world_pose_m = _reorder_mano_joints_to_internal(joints_world_pose_m)

            joints_world_joint3d = _transform_pts_cam_to_world(joint_3d, R_c, t_c)
            joints_world_joint3d = _reorder_dexycb_joint3d_to_internal(joints_world_joint3d)

            frame_errs.append(
                np.linalg.norm(joints_world_joint3d - joints_world_pose_m, axis=-1)
            )
            used_frames += 1
            if used_frames >= max_frames:
                break

        if not frame_errs:
            raise unittest.SkipTest(f"No valid frames found in sequence {sequence}")

        frame_errs = np.stack(frame_errs, axis=0)
        mean_err = float(np.mean(frame_errs))
        max_err = float(np.max(frame_errs))

        self.assertLess(
            mean_err,
            max_mean_err,
            f"joint_3d vs pose_m mean joint error too high: {mean_err:.6f} m",
        )
        self.assertLess(
            max_err,
            max_joint_err,
            f"joint_3d vs pose_m max joint error too high: {max_err:.6f} m",
        )


if __name__ == "__main__":
    unittest.main()
