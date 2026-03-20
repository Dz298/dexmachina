import os 
import sys 
import torch  
import numpy as np 
from os.path import join
from dexmachina.asset_utils import get_asset_path

ARCTIC_PROCESSED_DIR = get_asset_path("arctic/processed")
DEXYCB_PROCESSED_DIR = get_asset_path("dexycb/processed")
RETARGET_DIR = get_asset_path("retargeted")
RETARGET_CONTACT_DIR = get_asset_path("contact_retarget")


def _get_processed_demo_fname(
    obj_name="box",
    subject_name="s01",
    use_clip="01",
    data_source="arctic",
    data_fname=None,
    sequence_id=None,
):
    if data_fname is not None:
        return str(data_fname)
    if data_source == "dexycb":
        if sequence_id is None:
            sequence_id = obj_name
        return str(f"{DEXYCB_PROCESSED_DIR}/{subject_name}/{sequence_id}.npy")
    return str(f"{ARCTIC_PROCESSED_DIR}/{subject_name}/{obj_name}_use_{use_clip}.npy")


def _infer_hand_sides_from_world_coord(world_coord, motion_eps=1e-4, value_eps=1e-8):
    """Infer active hand sides from processed joints, preferring hands that actually move."""
    side_stats = {}
    for side in ("left", "right"):
        key = f"joints.{side}"
        if key not in world_coord:
            continue
        arr = np.asarray(world_coord[key])
        if arr.size == 0:
            continue
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
        side_stats[side] = {
            "has_signal": bool(np.any(np.abs(flat) > value_eps)),
            "motion": float(np.mean(np.std(flat, axis=0))) if flat.shape[0] > 1 else 0.0,
        }

    moving_sides = [side for side, stats in side_stats.items() if stats["motion"] > motion_eps]
    if moving_sides:
        return moving_sides

    present_sides = [side for side, stats in side_stats.items() if stats["has_signal"]]
    if present_sides:
        return present_sides

    return ["left", "right"]


def resolve_hand_sides(
    hand_sides=None,
    obj_name="box",
    subject_name="s01",
    use_clip="01",
    data_source="arctic",
    data_fname=None,
    sequence_id=None,
):
    """Resolve active hand sides from the processed demo unless the caller overrides them."""
    if hand_sides is not None:
        return list(hand_sides)

    demo_fname = _get_processed_demo_fname(
        obj_name=obj_name,
        subject_name=subject_name,
        use_clip=use_clip,
        data_source=data_source,
        data_fname=data_fname,
        sequence_id=sequence_id,
    )
    if not os.path.exists(demo_fname):
        return ["left", "right"]

    raw = np.load(demo_fname, allow_pickle=True).item()
    world_coord = raw["world_coord"]
    return _infer_hand_sides_from_world_coord(world_coord)


def get_demo_data(
    obj_name="box",
    frame_start=10,
    frame_end=30,
    hand_name="inspire_hand",
    subject_name="s01",
    use_clip="01",
    load_retarget_contact=False,
    hand_sides=None,
    data_source="arctic",
    data_fname=None,
    sequence_id=None,
):
    """Load processed demo data (ARCTIC or DexYCB). Returns only data for present hand_sides."""
    demo_fname = _get_processed_demo_fname(
        obj_name=obj_name,
        subject_name=subject_name,
        use_clip=use_clip,
        data_source=data_source,
        data_fname=data_fname,
        sequence_id=sequence_id,
    )

    raw = np.load(demo_fname, allow_pickle=True).item()
    world_coord = raw["world_coord"]
    params = raw["params"]

    if hand_sides is None:
        hand_sides = resolve_hand_sides(
            obj_name=obj_name,
            subject_name=subject_name,
            use_clip=use_clip,
            data_source=data_source,
            data_fname=demo_fname,
            sequence_id=sequence_id,
        )

    demo_data = {
        "obj_pos": params["obj_trans"][frame_start:frame_end],
        "obj_quat": params["obj_quat"][frame_start:frame_end],
        "obj_arti": params["obj_arti"][frame_start:frame_end],
    }
    if data_source == "dexycb" and "ycb_class_name" in params:
        demo_data["ycb_class_name"] = str(params["ycb_class_name"])
    for side in hand_sides:
        demo_data[f"contact_links_{side}"] = world_coord[f"contact_links_{side}"][
            frame_start:frame_end
        ]

    if load_retarget_contact:
        retar_contact = load_contact_retarget_data(
            obj_name=obj_name,
            hand_name=hand_name,
            frame_start=frame_start,
            frame_end=frame_end,
            save_name="genesis",
            use_clip=use_clip,
            subject_name=subject_name,
            hand_sides=hand_sides,
        )
        print("Replacing demo_data with retarget contact data")
        demo_data.update(retar_contact)
    return demo_data
 
def get_joint_init_limits(joint_pos_dict):
    limits = dict()
    default_qpos = dict()
    default_margin = 0.15
    for k, v in joint_pos_dict.items():        
        margin = default_margin
        if 'tx' in k or 'ty' in k or 'tz' in k:
            margin = 0.2 # 20 cm
        if 'roll' in k or 'pitch' in k or 'yaw' in k:
            # print(f"Using 30 degrees margin for {k}")
            margin = 0.5 # 30 degrees
        limits[k] = (min(v) - margin, max(v) + margin)
        default_qpos[k] = v[0] 
    return limits, default_qpos
 
def load_genesis_retarget_data(
    obj_name="box",
    hand_name="inspire_hand",
    frame_start=0,
    frame_end=100,
    save_name="genesis",
    use_clip="01",
    subject_name="s01",
    given_data_fname=None,
    hand_sides=None,
):
    """Data saved from retargeting code. hand_sides defaults to keys in retarget_data."""
    ret_type = "vector"
    if "shadow" in hand_name:
        print(f"Using position retargeting for {hand_name}")
        ret_type = "position"
    if given_data_fname is not None:
        data_fname = given_data_fname
    else:
        data_fname = f"{RETARGET_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}_{ret_type}_{save_name}.npy"
    data_fname = str(data_fname)
    loaded_tensor = False
    if not os.path.exists(data_fname):
        data_fname = data_fname.replace(".npy", ".pt")
    if not os.path.exists(data_fname):
        # DexYCB: parallel_retarget saves as {sequence_id}_{ret_type}_{save_name}.pt (no _use_01)
        data_fname_dexycb = f"{RETARGET_DIR}/{hand_name}/{subject_name}/{obj_name}_{ret_type}_{save_name}.pt"
        if os.path.exists(data_fname_dexycb):
            data_fname = data_fname_dexycb
        else:
            raise FileNotFoundError(
                f"Retarget file not found. Tried: {data_fname} and {data_fname_dexycb}"
            )

    if data_fname.endswith(".npy"):
        data = np.load(data_fname, allow_pickle=True).item()
    else:
        data = torch.load(data_fname)
        loaded_tensor = True

    demo_data = data["demo_data"]
    expected_length = frame_end - frame_start

    first_val = next(iter(demo_data.values()))
    data_length = first_val.shape[0] if hasattr(first_val, "shape") else len(first_val)
    already_sliced = data_length == expected_length

    if already_sliced:
        demo_data = {k: v for k, v in demo_data.items()}
    else:
        demo_data = {k: v[frame_start:frame_end] for k, v in demo_data.items()}
    if len(demo_data["obj_arti"].shape) > 1:
        demo_data["obj_arti"] = demo_data["obj_arti"][:, 0]

    retarget_loaded = data["retarget_data"]
    if hand_sides is None:
        hand_sides = list(retarget_loaded.keys())
    retarget_data = dict()
    for side in hand_sides:
        loaded = retarget_loaded[side]
        residual_qpos = loaded["joint_qpos"]
        if already_sliced:
            sliced_qpos = {k: v for k, v in residual_qpos.items()}
        else:
            sliced_qpos = {k: v[frame_start:frame_end] for k, v in residual_qpos.items()}
        
        qpos_targets = None 
        if 'joint_targets' in loaded:
            print("Using joint_targets")
            qpos_targets = loaded["joint_targets"]
            if already_sliced:
                qpos_targets = {k: v for k, v in qpos_targets.items()}
            else:
                qpos_targets = {k: v[frame_start:frame_end] for k, v in qpos_targets.items()}
        limits, init_pos = get_joint_init_limits(sliced_qpos) # this is a dict 
        kpt_pos = loaded["kpt_pos"]
        if len(kpt_pos.shape) > 3:
            print("Omitting the first dimension of kpt_pos")
            kpt_pos = kpt_pos[0]
        if already_sliced:
            kpt_info = dict(
                kpt_pos=kpt_pos,
                kpt_names=loaded["kpt_names"],
            )
        else:
            kpt_info = dict(
                kpt_pos=kpt_pos[frame_start:frame_end],
                kpt_names=loaded["kpt_names"],
            )
        wrist_pose = loaded[f"wrist_pose"]
        if len(wrist_pose.shape) > 2:
            print("Omitting the first dimension of wrist_pose")
            wrist_pose = wrist_pose[0]
        if not already_sliced:
            wrist_pose = wrist_pose[frame_start:frame_end]
        num_frames = wrist_pose.shape[0]
        retarget_data[side] = dict(
            init_qpos=init_pos, 
            limits=limits, 
            residual_qpos=sliced_qpos,
            qpos_targets=qpos_targets,
            num_frames=num_frames,
            kpts_data=kpt_info,
            wrist_pose=wrist_pose, # need this for contact frame reward
            ) 
    return demo_data, retarget_data

def load_contact_retarget_data(
    obj_name="box",
    hand_name="inspire_hand",
    frame_start=0,
    frame_end=100,
    save_name="genesis",
    use_clip="01",
    subject_name="s01",
    hand_sides=None,
):
    # e.g. assets/contact_retarget/ability_hand/s01/box_use_01.npy
    fname = f"{RETARGET_CONTACT_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}.npy"
    fname = str(fname)
    if not os.path.exists(fname):
        # DexYCB: map_contacts saves as {sequence_id}.npy (same as input .npy, no _use_01)
        fname_dexycb = f"{RETARGET_CONTACT_DIR}/{hand_name}/{subject_name}/{obj_name}.npy"
        if os.path.exists(fname_dexycb):
            fname = fname_dexycb
        else:
            raise FileNotFoundError(
                f"Contact retarget file not found. Tried: {fname} and {fname_dexycb}"
            )
    loaded = np.load(fname, allow_pickle=True).item()

    if hand_sides is None:
        hand_sides = list(loaded.keys())

    expected_length = frame_end - frame_start
    first_side_data = loaded[hand_sides[0]]["dexlink_contacts"]
    data_length = first_side_data.shape[0]
    already_sliced = data_length == expected_length

    retar_contact = dict()
    for side in hand_sides:
        data = loaded[side] 
        key_map = [
            ("dexlink_contacts", f"contact_links_{side}"),
            ("dexlink_valid_contacts", f"contact_links_valid_{side}"),
            ("dexlink_contacts_local", f"contact_links_local_{side}"),
            ("dexlink_contact_normals_local", f"contact_normals_local_{side}"),
        ]
        for source_key, target_key in key_map:
            if source_key not in data:
                continue
            if already_sliced:
                retar_contact[target_key] = data[source_key]
            else:
                retar_contact[target_key] = data[source_key][frame_start:frame_end]
        retar_contact[side] = {
            key: data[key]
            for key in ["collision_link_names", "collision_link_local_idxs", "object_part_names"]
            if key in data
        }
        
    return retar_contact
