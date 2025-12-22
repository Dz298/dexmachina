"""Orca Hand right config."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
    ElectricActuator,
    reflected_inertia,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

ORCA_HAND_RIGHT_XML = (
    Path(__file__).parent / "models" / "mjcf" / "orcahand_right_no_actuator.xml"
)
assert ORCA_HAND_RIGHT_XML.exists()


def get_assets() -> dict[str, bytes]:
    meshdir = "meshes/mjcf/right/visual"
    assets: dict[str, bytes] = {}
    update_assets(assets, ORCA_HAND_RIGHT_XML.parent / meshdir, meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ORCA_HAND_RIGHT_XML))
    spec.assets = get_assets()
    return spec


##
# Actuator config.
##

# Motor specs.
ROTOR_INERTIAS_XC330_T288_T = 1.7 * 1e-8
GEARS_XC330_T288_T = 288.35
ARMATURE_XC330_T288_T = reflected_inertia(
    ROTOR_INERTIAS_XC330_T288_T, GEARS_XC330_T288_T
)

ROTOR_INERTIAS_XC430_T240BB_T = 5.0 * 1e-8
GEARS_XC430_T240BB_T = 245.22
ARMATURE_XC430_T240BB_T = reflected_inertia(
    ROTOR_INERTIAS_XC430_T240BB_T, GEARS_XC430_T240BB_T
)

ACTUATOR_XC330_T288_T = ElectricActuator(
    reflected_inertia=ARMATURE_XC330_T288_T,
    velocity_limit=6.0,
    effort_limit=1.8,
)
ACTUATOR_XC430_T240BB_T = ElectricActuator(
    reflected_inertia=ARMATURE_XC430_T240BB_T,
    velocity_limit=6.0,
    effort_limit=1.8,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_XC330_T288_T = ARMATURE_XC330_T288_T * NATURAL_FREQ**2
STIFFNESS_XC430_T240BB_T = ARMATURE_XC430_T240BB_T * NATURAL_FREQ**2

DAMPING_XC330_T288_T = 2.0 * DAMPING_RATIO * ARMATURE_XC330_T288_T * NATURAL_FREQ
DAMPING_XC430_T240BB_T = 2.0 * DAMPING_RATIO * ARMATURE_XC430_T240BB_T * NATURAL_FREQ

ORCA_HAND_RIGHT_ACTUATOR_XC330_T288_T = ActuatorCfg(
    joint_names_expr=[
        "right_thumb_.*",
        "right_index_.*",
        "right_middle_.*",
        "right_ring_.*",
        "right_pinky_.*",
    ],
    effort_limit=ACTUATOR_XC330_T288_T.effort_limit,
    armature=ACTUATOR_XC330_T288_T.reflected_inertia,
    stiffness=STIFFNESS_XC330_T288_T,
    damping=DAMPING_XC330_T288_T,
)

ORCA_HAND_RIGHT_ACTUATOR_XC430_T240BB_T = ActuatorCfg(
    joint_names_expr=["right_wrist"],
    effort_limit=ACTUATOR_XC430_T240BB_T.effort_limit,
    armature=ACTUATOR_XC430_T240BB_T.reflected_inertia,
    stiffness=STIFFNESS_XC430_T240BB_T,
    damping=DAMPING_XC430_T240BB_T,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(-0.04115, 0, 0.1),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=["right_collision_.*"],
    condim={r"^(left|right)_foot[1-7]_collision$": 3},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=["right_collision_.*"],
    contype=0,
    conaffinity=1,
    condim={r"^(left|right)_foot[1-7]_collision$": 3},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

##
# Final config.
##

ORCA_HAND_RIGHT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ORCA_HAND_RIGHT_ACTUATOR_XC330_T288_T,
        ORCA_HAND_RIGHT_ACTUATOR_XC430_T240BB_T,
    ),
    soft_joint_pos_limit_factor=0.95,
)

ORCA_HAND_RIGHT_CFG = EntityCfg(
    init_state=HOME_KEYFRAME,
    # FIXME(xzhu): ignore collision configs for now
    # collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ORCA_HAND_RIGHT_ARTICULATION,
)

ORCA_HAND_RIGHT_ACTION_SCALE: dict[str, float] = {}
for a in ORCA_HAND_RIGHT_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            ORCA_HAND_RIGHT_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(ORCA_HAND_RIGHT_CFG)

    viewer.launch(robot.spec.compile())
