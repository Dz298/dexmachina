"""
Process DexYCB sequences to the same .npy format as ARCTIC (world_coord + params)
for single-hand retargeting. Output is in Genesis frame (Z-up, object centered on table).
Uses MANO model for hand vertices/joints and YCB meshes for object vertices.
"""
import os
import argparse
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from dexmachina.retargeting.contact_utils import (
    MANO_HAND_LINKS,
    approximate_contact_with_id,
    find_closest_link,
)
from dexmachina.retargeting.coordinate_utils import dexycb_world_to_genesis
from dexmachina.asset_utils import get_asset_path


# DexYCB/manopth MANO joint order differs from the internal order used across this repo
# (see MANO_HAND_LINKS and retarget configs where fingertip indices are 16..20).
# We remap joints here so downstream retargeting/contact code can use one consistent convention.
# dst_idx -> src_idx mapping (both length 21):
# internal: [wrist, idx(1..3), mid(1..3), pinky(1..3), ring(1..3), thumb(1..3), tips(thumb,idx,mid,ring,pinky)]
# manopth: [wrist, idx(1..4), mid(1..4), pinky(1..4), ring(1..4), thumb(1..4)]
_MANOPTH_TO_INTERNAL_JOINT_ORDER = np.array(
    [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 4, 8, 16, 12],
    dtype=np.int64,
)


def _reorder_mano_joints_to_internal(joints):
    """Reorder MANO joints from manopth output convention to DexMachina internal convention."""
    joints = np.asarray(joints)
    if joints.shape[0] != 21:
        raise ValueError(f"Expected 21 MANO joints, got shape {joints.shape}")
    return joints[_MANOPTH_TO_INTERNAL_JOINT_ORDER]


def _verify_processed_npy(args):
    """Load a processed DexYCB .npy and run sanity checks."""
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = str(get_asset_path("dexycb/processed"))
    subject = args.sequence.split("/")[0]
    seq_id = args.sequence.split("/")[1] if "/" in args.sequence else args.sequence
    path = os.path.join(save_dir, subject, f"{seq_id}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Verify failed: no file at {path}. Run with --save first.")
    data = np.load(path, allow_pickle=True).item()
    wc = data["world_coord"]
    params = data["params"]
    T = params["obj_trans"].shape[0]
    ok = True

    # Required keys
    for key in ["joints.left", "joints.right", "contact_links_left", "contact_links_right",
                "contacts.left", "contacts.right", "valid_contacts.left", "valid_contacts.right"]:
        if key not in wc:
            print(f"[FAIL] world_coord missing key: {key}")
            ok = False
    for key in ["obj_trans", "obj_quat", "obj_arti", "ycb_class_name"]:
        if key not in params:
            print(f"[FAIL] params missing key: {key}")
            ok = False
    if ok:
        print("[OK] Required world_coord and params keys present")

    # Shapes
    if params["obj_trans"].shape != (T, 3):
        print(f"[FAIL] obj_trans shape {params['obj_trans'].shape} != (T={T}, 3)")
        ok = False
    if params["obj_quat"].shape != (T, 4):
        print(f"[FAIL] obj_quat shape {params['obj_quat'].shape} != (T={T}, 4)")
        ok = False
    for side in ("left", "right"):
        j = wc[f"joints.{side}"]
        if j.shape != (T, 21, 3):
            print(f"[FAIL] joints.{side} shape {j.shape} != (T={T}, 21, 3)")
            ok = False
    if ok:
        print(f"[OK] Shapes consistent (T={T}, joints 21x3, obj 3/4)")

    # Hand sides: exactly one side should have non-zero joints
    hand_sides = []
    for side in ("left", "right"):
        key = f"joints.{side}"
        if key in wc:
            arr = wc[key]
            if arr.size > 0 and np.any(np.abs(arr) > 1e-8):
                hand_sides.append(side)
    if not hand_sides:
        hand_sides = ["left", "right"]
    if len(hand_sides) == 0:
        print("[FAIL] No hand side inferred (both joints empty?)")
        ok = False
    else:
        print(f"[OK] Inferred hand_sides: {hand_sides}")

    # NaNs
    has_nan = False
    for name, arr in [
        ("obj_trans", params["obj_trans"]),
        ("obj_quat", params["obj_quat"]),
        ("joints.left", wc["joints.left"]),
        ("joints.right", wc["joints.right"]),
    ]:
        if np.any(np.isnan(arr)):
            print(f"[FAIL] NaNs in {name}")
            ok = False
            has_nan = True
    if not has_nan:
        print("[OK] No NaNs in obj_trans, obj_quat, joints")

    # Object moved (optional)
    obj_std = np.std(params["obj_trans"], axis=0)
    if np.max(obj_std) < 1e-6:
        print("[WARN] Object position almost constant (obj_trans std < 1e-6)")
    else:
        print(f"[OK] Object moved (obj_trans std ~ {obj_std})")

    # YCB class
    ycb = params.get("ycb_class_name", "?")
    print(f"[INFO] ycb_class_name: {ycb}, num_frames: {T}")
    if ok:
        print("Verification passed.")
    else:
        raise SystemExit(1)


# YCB class names by index (match dex_ycb_toolkit)
_YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


def _rotation_matrix_to_axis_angle(R_mat):
    """Convert (3,3) rotation matrix to axis-angle (3,)."""
    r = R.from_matrix(R_mat)
    return r.as_rotvec().astype(np.float32)


def _load_ycb_mesh_vertices(dex_ycb_dir, ycb_id):
    """Load YCB mesh vertices in model frame (meters). Returns (N, 3)."""
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("process_dexycb requires trimesh: pip install trimesh") from e
    class_name = _YCB_CLASSES.get(ycb_id)
    if not class_name:
        raise ValueError(f"Unknown YCB id {ycb_id}")
    path = os.path.join(dex_ycb_dir, "models", class_name, "textured_simple.obj")
    if not os.path.exists(path):
        raise FileNotFoundError(f"YCB mesh not found: {path}")
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        verts = np.concatenate([m.vertices for m in mesh.geometry.values()], axis=0)
    else:
        verts = mesh.vertices
    return np.asarray(verts, dtype=np.float32)


def _bbox_center_to_bottom(verts):
    """Vertical distance from bbox center to bottom in model frame (Z)."""
    v = np.asarray(verts, dtype=np.float64)
    lo, hi = np.min(v, axis=0), np.max(v, axis=0)
    center_z = (lo[2] + hi[2]) * 0.5
    return float(center_z - lo[2])


def _transform_pts_cam_to_world(pts, R_c, t_c):
    """pts (N, 3), R_c (3,3), t_c (3,). Returns (N, 3) in world."""
    return (pts @ R_c.T) + t_c


def main():
    parser = argparse.ArgumentParser(
        description="Process DexYCB sequence to contact .npy (single-hand)"
    )
    parser.add_argument(
        "--sequence",
        "-seq",
        type=str,
        required=True,
        help="Sequence name, e.g. 20200709-subject-01/20200709_141754",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera index (0-based). Default: use master camera from extrinsics.",
    )
    parser.add_argument("--contact_threshold", "-ct", type=float, default=0.01)
    parser.add_argument("--max_contact_per_step", "-max", type=int, default=50)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--overwrite", "-ow", action="store_true")
    parser.add_argument(
        "--save_dir",
        "-sd",
        type=str,
        default=None,
        help="Default: assets/dexycb/processed",
    )
    parser.add_argument(
        "--mano_root",
        type=str,
        default=None,
        help="MANO model root (e.g. manopth/mano/models). Default from manopth.",
    )
    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Load saved .npy and run sanity checks (use with --sequence; optional --save_dir).",
    )
    args = parser.parse_args()

    if args.verify:
        _verify_processed_npy(args)
        return

    dex_ycb_dir = os.environ.get("DEX_YCB_DIR")
    if not dex_ycb_dir or not os.path.isdir(dex_ycb_dir):
        raise RuntimeError("Set DEX_YCB_DIR to the DexYCB dataset root")

    try:
        from manopth.manolayer import ManoLayer
    except ImportError as e:
        raise ImportError("process_dexycb requires manopth") from e

    seq_dir = os.path.join(dex_ycb_dir, args.sequence)
    meta_file = os.path.join(seq_dir, "meta.yml")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Meta not found: {meta_file}")

    with open(meta_file, "r") as f:
        meta = yaml.safe_load(f)

    num_frames = meta["num_frames"]
    serials = meta["serials"]
    ycb_ids = meta["ycb_ids"]
    ycb_grasp_ind = meta["ycb_grasp_ind"]
    mano_sides = meta["mano_sides"]
    mano_side = mano_sides[0]
    mano_calib_id = meta["mano_calib"][0]

    mano_calib_file = os.path.join(
        dex_ycb_dir, "calibration", f"mano_{mano_calib_id}", "mano.yml"
    )
    with open(mano_calib_file, "r") as f:
        mano_calib = yaml.safe_load(f)
    mano_betas = np.array(mano_calib["betas"], dtype=np.float32)

    # DexYCB coordinate frames:
    # - Each camera c has its own label dir (seq/serial_c/labels_XXXXXX.npz).
    # - pose_m, pose_y, joint_3d in those labels are in THAT camera's frame (README).
    # - Extrinsics T[serial] give camera-to-world: p_world = R_c @ p_cam + t_c.
    # - We load from ONE camera (master by default) and transform to world.
    # - Using any camera's labels + that camera's extrinsics yields the same world pose.
    extr_file = os.path.join(
        dex_ycb_dir,
        "calibration",
        f"extrinsics_{meta['extrinsics']}",
        "extrinsics.yml",
    )
    with open(extr_file, "r") as f:
        extr = yaml.load(f, Loader=yaml.FullLoader)  # FullLoader for !!python/tuple in DexYCB calibration
    T = extr["extrinsics"]
    cam_idx = args.camera
    if cam_idx is None:
        master_serial = extr["master"]
        cam_idx = serials.index(master_serial)
        print(f"[INFO] Using master camera: {master_serial} (index {cam_idx})")
    serial = serials[cam_idx]
    T_serial = np.array(T[serial], dtype=np.float32).reshape(3, 4)
    R_c = T_serial[:, :3]
    t_c = T_serial[:, 3]

    mano_root = args.mano_root
    if mano_root is None:
        # Try env, then paths relative to common repo layouts
        for candidate in [
            os.environ.get("MANOPTH_ROOT"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "dex-ycb-toolkit", "manopth", "mano", "models"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "dex-ycb-toolkit", "manopth", "mano_v1_2", "models"),
        ]:
            if candidate and os.path.isdir(candidate):
                right_pkl = os.path.join(candidate, "MANO_RIGHT.pkl")
                if os.path.isfile(right_pkl):
                    mano_root = os.path.abspath(candidate)
                    break
        if mano_root is None:
            mano_root = "manopth/mano/models"  # fallback (relative to cwd)
    right_pkl = os.path.join(mano_root, "MANO_RIGHT.pkl")
    if not os.path.isfile(right_pkl):
        raise FileNotFoundError(
            f"MANO model not found at {right_pkl}. "
            "Download from https://mano.is.tue.mpg.de and place MANO_LEFT.pkl, MANO_RIGHT.pkl in a folder, "
            "then pass --mano_root /path/to/that/folder or set MANOPTH_ROOT."
        )
    mano_layer = ManoLayer(
        flat_hand_mean=False,
        ncomps=45,
        side=mano_side,
        mano_root=mano_root,
        use_pca=True,
    )
    device = torch.device("cpu")
    mano_layer = mano_layer.to(device)

    contact_thres = args.contact_threshold
    max_contact_per_step = args.max_contact_per_step

    # Trim to frames with valid hand data (pose_m != 0)
    label_dir = os.path.join(seq_dir, serial, "")
    valid_frames = []
    for frame in range(num_frames):
        label_path = os.path.join(label_dir, f"labels_{frame:06d}.npz")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")
        label = np.load(label_path)
        if not np.all(label["pose_m"] == 0.0):
            valid_frames.append(frame)
    if not valid_frames:
        raise RuntimeError(f"No frames with valid hand data in sequence {args.sequence}")
    num_frames_trimmed = len(valid_frames)
    if num_frames_trimmed < num_frames:
        print(f"[INFO] Trimmed {num_frames} -> {num_frames_trimmed} frames (hand valid only)")
    num_frames = num_frames_trimmed

    # Allocate for both sides; only active side filled
    contacts_left = np.zeros((num_frames, max_contact_per_step, 4), dtype=np.float32)
    valid_contacts_left = np.zeros((num_frames, max_contact_per_step), dtype=bool)
    contacts_right = np.zeros((num_frames, max_contact_per_step, 4), dtype=np.float32)
    valid_contacts_right = np.zeros((num_frames, max_contact_per_step), dtype=bool)
    contacts_left_hand = np.zeros((num_frames, len(MANO_HAND_LINKS), 4), dtype=np.float32)
    contacts_right_hand = np.zeros((num_frames, len(MANO_HAND_LINKS), 4), dtype=np.float32)

    joints_left = np.zeros((num_frames, 21, 3), dtype=np.float32)
    joints_right = np.zeros((num_frames, 21, 3), dtype=np.float32)

    obj_trans_list = []
    obj_rot_list = []
    obj_quat_list = []
    trans_l_list, trans_r_list = [], []
    rot_l_list, rot_r_list = [], []
    shape_l = np.zeros((num_frames, 10), dtype=np.float32)
    shape_r = np.zeros((num_frames, 10), dtype=np.float32)
    pose_l_list, pose_r_list = [], []

    grasp_ycb_id = ycb_ids[ycb_grasp_ind]
    obj_verts_model = _load_ycb_mesh_vertices(dex_ycb_dir, grasp_ycb_id)

    for out_idx, frame in enumerate(valid_frames):
        label_path = os.path.join(label_dir, f"labels_{frame:06d}.npz")
        label = np.load(label_path)
        pose_m = label["pose_m"]
        pose_y = label["pose_y"]

        # MANO: pose_m (1, 51) -> PCA 0:48, trans 48:51 (mm)
        pose_t = torch.from_numpy(pose_m).float().to(device)
        betas = torch.from_numpy(mano_betas).float().unsqueeze(0).to(device)
        verts, joints = mano_layer(pose_t[:, 0:48], betas, pose_t[:, 48:51])
        verts = verts.squeeze(0).cpu().numpy() / 1000.0
        joints = joints.squeeze(0).cpu().numpy() / 1000.0
        verts_world = _transform_pts_cam_to_world(verts, R_c, t_c)
        joints_world = _transform_pts_cam_to_world(joints, R_c, t_c)
        joints_world = _reorder_mano_joints_to_internal(joints_world)

        if mano_side == "left":
            joints_left[out_idx] = joints_world
            trans_l_list.append(pose_t[0, 48:51].cpu().numpy() / 1000.0)
            rot_l_list.append(np.zeros(3, dtype=np.float32))
            pose_l_list.append(pose_t[0, 0:48].cpu().numpy())
            trans_r_list.append(np.zeros(3, dtype=np.float32))
            rot_r_list.append(np.zeros(3, dtype=np.float32))
            pose_r_list.append(np.zeros(48, dtype=np.float32))
        else:
            joints_right[out_idx] = joints_world
            trans_r_list.append(pose_t[0, 48:51].cpu().numpy() / 1000.0)
            rot_r_list.append(np.zeros(3, dtype=np.float32))
            pose_r_list.append(pose_t[0, 0:48].cpu().numpy())
            trans_l_list.append(np.zeros(3, dtype=np.float32))
            rot_l_list.append(np.zeros(3, dtype=np.float32))
            pose_l_list.append(np.zeros(48, dtype=np.float32))

        # Object: grasped YCB pose in camera frame (3x4)
        pose_obj = pose_y[ycb_grasp_ind]
        R_obj = pose_obj[:3, :3]
        t_obj = pose_obj[:3, 3]
        obj_verts_cam = (obj_verts_model @ R_obj.T) + t_obj
        obj_verts_world = _transform_pts_cam_to_world(obj_verts_cam, R_c, t_c)

        R_obj_world = R_c @ R_obj
        t_obj_world = R_c @ t_obj + t_c
        obj_trans_list.append(t_obj_world.astype(np.float32))
        obj_rot_list.append(_rotation_matrix_to_axis_angle(R_obj_world))
        q = R.from_matrix(R_obj_world).as_quat()
        obj_quat_list.append(np.array([q[3], q[0], q[1], q[2]], dtype=np.float32))

        # Contact: single part_id=1 for rigid YCB
        obj_part_ids = np.ones(obj_verts_world.shape[0], dtype=np.float32)
        contacts, contacts_hand = approximate_contact_with_id(
            obj_verts_world, obj_part_ids, verts_world, threshold=contact_thres
        )
        if contacts.shape[0] > 0:
            n_contacts = min(max_contact_per_step, contacts.shape[0])
            contacts = contacts[:n_contacts]
            contacts_hand = contacts_hand[:n_contacts]
            contact_links = find_closest_link(contacts_hand, joints_world)
            if mano_side == "left":
                contacts_left[out_idx, :n_contacts] = contacts
                valid_contacts_left[out_idx, :n_contacts] = True
                contacts_left_hand[out_idx] = np.array(contact_links, dtype=np.float32)
            else:
                contacts_right[out_idx, :n_contacts] = contacts
                valid_contacts_right[out_idx, :n_contacts] = True
                contacts_right_hand[out_idx] = np.array(contact_links, dtype=np.float32)

        if (out_idx + 1) % 100 == 0 or out_idx + 1 == num_frames:
            print(f"Processed frame {out_idx + 1}/{num_frames}")

    shape_l[:] = mano_betas if mano_side == "left" else 0.0
    shape_r[:] = mano_betas if mano_side == "right" else 0.0

    # Transform DexYCB world -> Genesis frame (AprilTag frame + offset to place the spawn frame on the table)
    obj_half_height = _bbox_center_to_bottom(obj_verts_model)
    T_apriltag = np.array(T["apriltag"], dtype=np.float32).reshape(3, 4)
    R_tag, t_tag = T_apriltag[:, :3], T_apriltag[:, 3]
    obj_trans = np.stack(obj_trans_list, axis=0)
    obj_quat = np.stack(obj_quat_list, axis=0)
    dexycb_world_to_genesis(
        obj_trans,
        obj_quat,
        joints_left,
        joints_right,
        contacts_left,
        contacts_right,
        R_tag,
        t_tag,
        obj_half_height=obj_half_height,
        obj_verts_model=obj_verts_model,
    )
    obj_rot = np.stack([_rotation_matrix_to_axis_angle(R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()) for q in obj_quat], axis=0)

    tosave = {
        "world_coord": {
            "joints.left": joints_left,
            "joints.right": joints_right,
            "contacts.left": contacts_left,
            "valid_contacts.left": valid_contacts_left,
            "contacts.right": contacts_right,
            "valid_contacts.right": valid_contacts_right,
            "contact_threshold": np.array(contact_thres, dtype=np.float32),
            "contact_links_left": contacts_left_hand,
            "contact_links_right": contacts_right_hand,
        },
        "params": {
            "frame": "genesis",
            "ycb_class_name": _YCB_CLASSES[grasp_ycb_id],
            "obj_trans": obj_trans,
            "obj_rot": obj_rot,
            "obj_quat": obj_quat,
            "obj_arti": np.zeros((num_frames,), dtype=np.float32),
            "trans_l": np.stack(trans_l_list, axis=0),
            "trans_r": np.stack(trans_r_list, axis=0),
            "rot_l": np.stack(rot_l_list, axis=0),
            "rot_r": np.stack(rot_r_list, axis=0),
            "shape_l": shape_l,
            "shape_r": shape_r,
            "pose_l": np.stack(pose_l_list, axis=0),
            "pose_r": np.stack(pose_r_list, axis=0),
            "cam_R_world": R_c,
            "cam_t_world": t_c,
        },
    }

    if args.save:
        save_dir = args.save_dir
        if save_dir is None:
            save_dir = str(get_asset_path("dexycb/processed"))
        os.makedirs(save_dir, exist_ok=True)
        subject = args.sequence.split("/")[0]
        seq_id = args.sequence.split("/")[1] if "/" in args.sequence else args.sequence
        save_fname = os.path.join(save_dir, subject, f"{seq_id}.npy")
        os.makedirs(os.path.dirname(save_fname), exist_ok=True)
        if os.path.exists(save_fname) and not args.overwrite:
            raise FileExistsError(f"Exists: {save_fname} (use --overwrite)")
        np.save(save_fname, tosave)
        print(f"Saved to {save_fname}")
    else:
        print("Dry run (use --save to write .npy)")


if __name__ == "__main__":
    main()
