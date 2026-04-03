"""
Orca Hand configuration for DexMachina.
5-fingered dexterous hand with 22 DOFs (6 wrist + 16 finger).

Finger structure:
- Thumb: mcp, abd, pip, dip (4 joints)
- Index/Middle/Ring/Pinky: abd, mcp, pip (3 joints each)

NOTE: default_qpos values are extracted from the original MJCF ref= attributes.
The URDF conversion loses these values, causing joints to initialize at 0
which can be at joint limits (e.g., thumb_abd upper limit = 0).
"""
import os
from os.path import join
from dexmachina.asset_utils import get_urdf_path

orca_asset_dir = "orca_hand/"
left_rel_urdf = join(orca_asset_dir, "orcahand_left_6dof.urdf")
right_rel_urdf = join(orca_asset_dir, "orcahand_right_6dof.urdf")

# Default qpos from MJCF ref= attributes
# These are the proper initial positions that put joints in the middle of their range
# Order must match the actuated joint order in the URDF
ORCA_LEFT_DEFAULT_QPOS = [
    0.0,       # L_forearm_tx_link_joint
    0.0,       # L_forearm_ty_link_joint
    0.0,       # L_forearm_tz_link_joint
    0.0,       # L_forearm_roll_link_joint
    0.0,       # L_forearm_pitch_link_joint
    0.0,       # L_forearm_yaw_link_joint
    0.0,       # left_thumb_mcp
    -0.4,      # left_index_abd (MJCF ref=-0.4)
    0.0,       # left_middle_abd
    0.17,      # left_ring_abd (MJCF ref=0.17)
    0.52333,   # left_pinky_abd (MJCF ref=0.52333)
    -0.73304,  # left_thumb_abd (MJCF ref=-0.73304) - CRITICAL: was stuck at 0 (upper limit)
    0.0,       # left_index_mcp
    0.0,       # left_middle_mcp
    0.0,       # left_ring_mcp
    0.0,       # left_pinky_mcp
    -0.58496,  # left_thumb_pip (MJCF ref=-0.58496)
    0.0,       # left_index_pip
    0.0,       # left_middle_pip
    0.0,       # left_ring_pip
    0.0,       # left_pinky_pip
    -0.50477,  # left_thumb_dip (MJCF ref=-0.50477)
]

ORCA_RIGHT_DEFAULT_QPOS = [
    0.0,       # R_forearm_tx_link_joint
    0.0,       # R_forearm_ty_link_joint
    0.0,       # R_forearm_tz_link_joint
    0.0,       # R_forearm_roll_link_joint
    0.0,       # R_forearm_pitch_link_joint
    0.0,       # R_forearm_yaw_link_joint
    0.0,       # right_thumb_mcp
    -0.4,      # right_index_abd (MJCF ref=-0.4)
    0.0,       # right_middle_abd
    0.17,      # right_ring_abd (MJCF ref=0.17)
    0.52333,   # right_pinky_abd (MJCF ref=0.52333)
    -0.73304,  # right_thumb_abd (MJCF ref=-0.73304)
    0.0,       # right_index_mcp
    0.0,       # right_middle_mcp
    0.0,       # right_ring_mcp
    0.0,       # right_pinky_mcp
    -0.58496,  # right_thumb_pip (MJCF ref=-0.58496)
    0.0,       # right_index_pip
    0.0,       # right_middle_pip
    0.0,       # right_ring_pip
    0.0,       # right_pinky_pip
    -0.50477,  # right_thumb_dip (MJCF ref=-0.50477)
]

## Less stiff gains for residual mode
# finger_kp = 10.0  # Tuned from replay gain sweep (coarse mode optimal)
# finger_kv = 1.58   # kv ≈ sqrt(kp)/2 for critical damping
# wrist_rot_kp = 100.0  # Tuned for stable wrist control
# wrist_rot_kv = 5.0    # Matched to kp for damping
# wrist_trans_kp = 100.0  # Conservative to prevent object drops
# wrist_trans_kv = 5.0   # Damping for translation
# force_range = 100.0  # Increased from 50 to allow stronger forces

# Stiffer gains for hybrid mode
finger_kp = 80.0  # Tuned from replay gain sweep (coarse mode optimal)
finger_kv = finger_kp ** 0.5 / 2  # kv ≈ sqrt(kp)/2 for critical damping
wrist_rot_kp = 300.0  # Tuned for stable wrist control
wrist_rot_kv = wrist_rot_kp ** 0.5 / 2    # Matched to kp for damping
wrist_trans_kp = 300.0  # Conservative to prevent object drops
wrist_trans_kv = wrist_trans_kp ** 0.5 / 2   # Damping for translation
force_range = 100.0  # Increased from 50 to allow stronger forces



ORCA_LEFT_CFG = {
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "left_palm",
    "kpt_link_names": [
        "left_thumb_fingertip",
        "left_index_fingertip",
        "left_middle_fingertip",
        "left_ring_fingertip",
        "left_pinky_fingertip",
    ],
    "default_qpos": ORCA_LEFT_DEFAULT_QPOS,
    "actuators": {
        "finger": dict(
            # Match all finger joints (mcp, abd, pip, dip)
            joint_exprs=[r'left_(thumb|index|middle|ring|pinky)_(mcp|abd|pip|dip)'],
            kp=finger_kp,
            kv=finger_kv,
            force_range=force_range,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=wrist_rot_kp,
            kv=wrist_rot_kv,
            force_range=force_range,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=wrist_trans_kp,
            kv=wrist_trans_kv,
            force_range=force_range,
        ),
    },
    # Collision groups from inspect_raw_urdf.py --gather_geoms
    # Format: {link_idx: group_id}, where group 0 is palm, 1-5 are fingers (thumb, index, middle, ring, pinky)
    # Links: left_palm(7), thumb_mp(13), index_mp(14), middle_mp(15), ring_mp(16), pinky_mp(17),
    #        thumb_pp(23), index_pp(24), middle_pp(25), ring_pp(26), pinky_pp(27),
    #        thumb_ip(33), index_ip(34), middle_ip(35), ring_ip(36), pinky_ip(37), thumb_dp(43)
    "collision_groups": {
        7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5,
        23: 1, 24: 2, 25: 3, 26: 4, 27: 5,
        33: 1, 34: 2, 35: 3, 36: 4, 37: 5, 43: 1
    },
    "collision_palm_name": "left_palm",
}

ORCA_RIGHT_CFG = {
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "right_palm",
    "kpt_link_names": [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ],
    "default_qpos": ORCA_RIGHT_DEFAULT_QPOS,
    "actuators": {
        "finger": dict(
            joint_exprs=[r'right_(thumb|index|middle|ring|pinky)_(mcp|abd|pip|dip)'],
            kp=finger_kp,
            kv=finger_kv,
            force_range=force_range,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=wrist_rot_kp,
            kv=wrist_rot_kv,
            force_range=force_range,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=wrist_trans_kp,
            kv=wrist_trans_kv,
            force_range=force_range,
        ),
    },
    # Same collision groups as left hand (symmetric structure)
    "collision_groups": {
        7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5,
        23: 1, 24: 2, 25: 3, 26: 4, 27: 5,
        33: 1, 34: 2, 35: 3, 36: 4, 37: 5, 43: 1
    },
    "collision_palm_name": "right_palm",
}

ORCA_CFGs = dict(
    left=ORCA_LEFT_CFG,
    right=ORCA_RIGHT_CFG,
)
