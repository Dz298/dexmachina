"""Differential inverse kinematics for the UR5 + ORCA right-hand model.

Adapted from ur5_orca/kinematics.py. Uses pink's QP-based differential IK
to track a target wrist pose, subject to joint and velocity limits.
Visualization is intentionally omitted; use Genesis for rendering.
"""

from typing import Literal

import numpy as np
import pink
import pinocchio as pin
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.barriers import PositionBarrier
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation as R

from dexmachina.asset_utils import get_asset_path


WristFrame = Literal["ur_eef_site", "orca_wrist_site"]


class UR5IKSolver:
    """Differential IK solver for the UR5 arm with a mounted ORCA right hand.

    Wraps pink's QP-based IK loop around a pinocchio model loaded from
    the bundled MJCF asset. A single FrameTask tracks the chosen wrist
    site; joint and velocity limits are enforced as QP constraints.
    """

    def __init__(
        self,
        model_path: str | None = None,
        solver: str = "daqp",
        max_iter: int = 20,
        frequency: float = 100.0,
        wrist_frame_name: WristFrame = "orca_wrist_site",
        posture_cost: float = 1e-3,
        floor_z: float | None = None,
    ):
        """Initialize the solver from an MJCF scene.

        Args:
            model_path: Path to the MJCF scene. Defaults to the bundled
                ``ur5/ur5_orca_right_ik.xml`` asset.
            solver: QP backend used by pink (e.g. ``"daqp"``).
            max_iter: Number of IK integration steps per :meth:`solve_ik` call.
            frequency: Control-loop frequency in Hz; sets the integration dt.
            wrist_frame_name: Site frame tracked by the IK task.
            posture_cost: Weight of the posture regularisation task.  A small
                positive value (default 1e-3) biases the null-space solution
                toward the previous configuration, preventing elbow-flip
                oscillation between the two valid UR5 elbow branches.
            floor_z: Minimum allowed z-coordinate (in the pinocchio model's
                base frame) for every UR5 link.  When provided, a
                ``PositionBarrier`` is added for each arm link to keep them
                above this height (e.g. the desktop surface).  ``None``
                disables the floor constraint.
        """
        if model_path is None:
            model_path = str(get_asset_path("ur5/ur5_orca_right_ik.xml"))
        self.model_path = model_path
        self.solver = solver
        self.max_iter = max_iter
        self.frequency = frequency

        self.robot_wrapper = pin.RobotWrapper.BuildFromMJCF(filename=self.model_path)
        model = self.robot_wrapper.model

        self.configuration_limits = [
            ConfigurationLimit(model),
            VelocityLimit(model),
        ]
        self.configuration = pink.Configuration(
            model,
            self.robot_wrapper.data,
            self.robot_wrapper.q0,
        )

        self.dt = RateLimiter(frequency=frequency, warn=False).period
        self.frame_task = self._setup_frame_task(wrist_frame_name)

        # Posture task: keeps the solution near the current configuration by
        # updating its target to the current q before each solve.  The low
        # cost lets the frame task dominate while the null-space bias prevents
        # the solver from flipping to the mirror elbow branch.
        self.posture_task = PostureTask(cost=posture_cost)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Floor constraint: one PositionBarrier per UR5 link body, enforcing
        # z >= floor_z in the pinocchio base frame.
        _ur5_link_bodies = [
            "base",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        self.floor_barriers: list[PositionBarrier] = []
        if floor_z is not None:
            for link in _ur5_link_bodies:
                self.floor_barriers.append(
                    PositionBarrier(
                        frame=link,
                        indices=[2],
                        p_min=np.array([floor_z]),
                        gain=100.0,
                    )
                )

    def _setup_frame_task(
        self,
        wrist_frame_name: WristFrame,
        position_cost: float = 1.0,
        orientation_cost: float = 0.0,
    ) -> FrameTask:
        frame_task = FrameTask(
            wrist_frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
        )
        frame_task.set_target_from_configuration(self.configuration)
        return frame_task

    def solve_ik(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """Drive the wrist frame toward ``target_pose`` via differential IK.

        Runs ``self.max_iter`` pink-IK integration steps. When ``q_init`` is
        omitted the solver continues from its current configuration (warm-start).

        Args:
            target_pose: Desired wrist pose as ``[x, y, z, qw, qx, qy, qz]``
                (position + wxyz quaternion) in the world frame.
            q_init: Optional starting configuration for cold-start.

        Returns:
            Full joint configuration ``q`` (copy) after the iterations.
        """
        if q_init is not None:
            self.configuration.q = np.asarray(q_init, dtype=float).copy()
            self.configuration.update()

        # Pin the posture target to the starting configuration so the
        # null-space solution stays near it and elbow-flip is penalised.
        self.posture_task.set_target(self.configuration.q.copy())

        target = self.frame_task.transform_target_to_world
        target.translation = target_pose[:3]
        target.rotation = R.from_quat(target_pose[3:7], scalar_first=True).as_matrix()

        for _ in range(self.max_iter):
            velocity = solve_ik(
                self.configuration,
                [self.frame_task, self.posture_task],
                self.dt,
                solver=self.solver,
                limits=self.configuration_limits,
                barriers=self.floor_barriers if self.floor_barriers else None,
            )
            self.configuration.integrate_inplace(velocity, self.dt)

        return self.configuration.q.copy()

    def ur5_qpos(self, target_pose: np.ndarray, q_init: np.ndarray | None = None) -> np.ndarray:
        """Return only the 6 UR5 joint angles for the given wrist target pose.

        Args:
            target_pose: ``[x, y, z, qw, qx, qy, qz]`` in world frame.
            q_init: Optional full joint configuration for warm-start.

        Returns:
            Array of shape (6,) with UR5 joint positions in radians.
        """
        return self.solve_ik(target_pose, q_init)[:6]


if __name__ == "__main__":
    solver = UR5IKSolver()
    target_pose = np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0])
    q = solver.solve_ik(target_pose)
    ur5_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    print("UR5 arm joints (radians):")
    for name, val in zip(ur5_joint_names, q[:6]):
        print(f"  {name}: {val:.4f}")
