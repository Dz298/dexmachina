"""Genesis wrapper for a UR5 arm visualized alongside a dexmachina hand policy.

Loads the UR5 arm as a collision-free Genesis entity, runs pinocchio/pink
differential IK to track the hand's wrist pose, and sets the arm's joint
positions each step.
"""

import numpy as np
import torch
import genesis as gs
from scipy.spatial.transform import Rotation as _R

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.ur5_ik import UR5IKSolver

_UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR5Arm:
    """Visual UR5 arm in a Genesis scene, driven by differential IK.

    Intended for eval-time visualization: the arm tracks the wrist pose of a
    dexmachina hand policy without participating in contacts or dynamics.
    """

    def __init__(
        self,
        scene: gs.Scene,
        num_envs: int,
        device: torch.device,
        base_pos: tuple = (0.0, 0.0, 0.0),
        base_quat: tuple = (1.0, 0.0, 0.0, 0.0),
        wrist_frame: str = "ur_eef_site",
        ik_solver: str = "daqp",
        ik_max_iter: int = 20,
        ik_orientation_cost: float = 0.5,
        floor_z_world: float = 0.6,
    ):
        """Add the UR5 arm entity to *scene* and create the IK solver.

        Must be called **before** ``scene.build()``.

        Args:
            scene: Genesis scene (pre-build).
            num_envs: Number of parallel environments.
            device: Torch device for tensor ops.
            base_pos: World-space position of the arm base (x, y, z).
            base_quat: World-space quaternion of the arm base (w, x, y, z).
            wrist_frame: Pinocchio frame name to track with IK. Defaults to
                ``ur_eef_site`` (the UR5 flange) so it coincides with the
                rendered end-effector of ``ur5.xml``.
            ik_solver: QP backend for pink (default ``"daqp"``).
            ik_max_iter: IK iterations per call.
            ik_orientation_cost: Weight of the orientation component of the
                frame task. Must be > 0 to avoid null-space jitter on the
                6-DOF arm (a pure 3-DOF position task leaves 3 redundant
                dimensions that otherwise thrash frame-to-frame).
            floor_z_world: World-frame z-height of the desktop surface.
                Defaults to 0.6 m (matching ``TABLE_HEIGHT`` in
                ``base_env.py``).  A ``PositionBarrier`` is added for every
                UR5 link so the arm stays above this surface.
        """
        self.num_envs = num_envs
        self.device = device

        mjcf_path = str(get_asset_path("ur5/ur5.xml"))
        self.entity = scene.add_entity(
            gs.morphs.MJCF(
                file=mjcf_path,
                pos=base_pos,
                quat=base_quat,
                collision=False,
            ),
        )

        # The pinocchio model has its base at the origin, so the floor height
        # in the model frame is the world height minus the arm's world base z.
        floor_z_base_frame = float(floor_z_world) - float(base_pos[2])

        self.ik = UR5IKSolver(
            wrist_frame_name=wrist_frame,
            solver=ik_solver,
            max_iter=ik_max_iter,
            floor_z=floor_z_base_frame,
        )
        self.ik.frame_task.set_orientation_cost(float(ik_orientation_cost))

        self._dof_idxs: list[int] = []
        self._q_prev: np.ndarray | None = None
        self._base_pos_np = np.asarray(base_pos, dtype=np.float32)

        # Cached per-step quantities for external logging (e.g. into eval npy).
        self.last_wrist_target: np.ndarray | None = None  # (num_envs, 7) wxyz, world frame
        self.last_ur5_q: np.ndarray | None = None         # (num_envs, 6)
        self.last_ee_fk_pose: np.ndarray | None = None    # (7,) wxyz, arm base frame

    def post_scene_build_setup(self) -> None:
        """Resolve UR5 DOF indices after the scene has been built.

        Must be called after ``scene.build()``.
        """
        self._dof_idxs = [
            j.dof_idx_local
            for j in self.entity.joints
            if j.name in _UR5_JOINT_NAMES
        ]
        assert len(self._dof_idxs) == 6, (
            f"Expected 6 UR5 DOFs, found {len(self._dof_idxs)}. "
            f"Joints in entity: {[j.name for j in self.entity.joints]}"
        )

    @torch.no_grad()
    def update(self, wrist_pose: torch.Tensor) -> None:
        """Run IK and set arm joint positions to track ``wrist_pose``.

        Args:
            wrist_pose: Tensor of shape ``(num_envs, 7)`` containing
                ``[x, y, z, qw, qx, qy, qz]`` in world frame (Genesis wxyz).
        """
        pose_np = wrist_pose.detach().cpu().numpy()

        # Pinocchio model has its base at the origin; express the target in
        # the arm's base frame so that Genesis rendering at ``base_pos``
        # reconstructs the world-frame target.
        pose_in_base = pose_np.copy()
        pose_in_base[:, :3] = pose_in_base[:, :3] - self._base_pos_np[None, :]

        q_batch = np.zeros((self.num_envs, 6), dtype=np.float32)
        for i in range(self.num_envs):
            q_full = self.ik.solve_ik(pose_in_base[i], q_init=self._q_prev)
            self._q_prev = q_full
            q_batch[i] = q_full[:6]

        qs = torch.from_numpy(q_batch).to(self.device)
        self.entity.set_dofs_position(qs, dofs_idx_local=self._dof_idxs)

        # Forward kinematics of the tracked frame (in the arm's base frame)
        # for downstream logging/analysis.
        ee_fk_pose = np.zeros(7, dtype=np.float32)
        try:
            T = self.ik.configuration.get_transform_frame_to_world(
                self.ik.frame_task.frame
            )
            ee_fk_pose[:3] = np.asarray(T.translation, dtype=np.float32)
            q_xyzw = _R.from_matrix(np.asarray(T.rotation)).as_quat()
            ee_fk_pose[3:] = np.array(
                [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32
            )
        except Exception:
            ee_fk_pose[:] = np.nan

        self.last_wrist_target = pose_np.copy()
        self.last_ur5_q = q_batch.copy()
        self.last_ee_fk_pose = ee_fk_pose.copy()
