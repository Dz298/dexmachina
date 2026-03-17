"""
DexYCB coordinate frame helpers.

- dexycb_world_to_genesis: Used in process_dexycb to output data directly in Genesis frame.
  DexYCB world (AprilTag calibration) -> AprilTag frame -> Genesis (offset to center on table).

- apply_dexycb_display_*: Legacy display transform for 2D camera overlay. No-op when data
  already has frame="genesis".
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Table top center in Genesis (cardboard box: pos=(0,-0.08,0.90), size=(0.2,0.2,0.1), top=0.95)
TABLE_TOP_CENTER = np.array([0.0, -0.08, 0.95], dtype=np.float32)

# Same as dex-ycb-toolkit visualize_pose.py: flip Y and Z for display (legacy).
DEXYCB_DISPLAY_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def dexycb_world_to_genesis(
    obj_trans,
    obj_quat,
    joints_left,
    joints_right,
    contacts_left,
    contacts_right,
    R_tag,
    t_tag,
    table_center=None,
    obj_half_height=0.0,
    obj_verts_model=None,
):
    """Transform DexYCB world frame data to Genesis frame. Modifies arrays in place.

    World -> Tag: p_tag = R_w2t @ p_world + t_w2t. AprilTag frame has Z-up (Genesis convention).
    Offset positions so the spawn frame's object bottom sits on the table top. When
    obj_verts_model is provided, compute the support height from the object's actual Genesis-frame
    orientation at the first valid frame; otherwise fall back to obj_half_height.
    """
    if table_center is None:
        table_center = TABLE_TOP_CENTER
    table_center = np.asarray(table_center, dtype=np.float32)
    obj_half_height = float(obj_half_height)
    R_tag = np.asarray(R_tag, dtype=np.float64)
    t_tag = np.asarray(t_tag, dtype=np.float64)
    R_w2t = np.linalg.inv(R_tag).astype(np.float32)
    t_w2t = (-R_w2t @ t_tag).astype(np.float32)

    valid = np.any(obj_trans != 0, axis=1)
    valid_indices = np.flatnonzero(valid)

    support_height = obj_half_height
    support_bottom_z = None
    if valid_indices.size > 0 and obj_verts_model is not None:
        first_valid = int(valid_indices[0])
        qw, qx, qy, qz = obj_quat[first_valid, 0], obj_quat[first_valid, 1], obj_quat[first_valid, 2], obj_quat[first_valid, 3]
        R_obj = R.from_quat([qx, qy, qz, qw]).as_matrix()
        R_obj_genesis = R_w2t @ R_obj
        verts_genesis = np.asarray(obj_verts_model, dtype=np.float64) @ R_obj_genesis.T
        support_height = float(-np.min(verts_genesis[:, 2]))
        first_center_tag = (obj_trans[first_valid] @ R_w2t.T) + t_w2t
        support_bottom_z = float(first_center_tag[2] + np.min(verts_genesis[:, 2]))

    target_center = table_center.copy()
    target_center[2] = table_center[2] + support_height

    if np.any(valid):
        pt_tag = (obj_trans[valid] @ R_w2t.T) + t_w2t
        median_tag = np.median(pt_tag, axis=0)
        offset = target_center - median_tag
        if support_bottom_z is not None:
            offset[2] = table_center[2] - support_bottom_z
    else:
        offset = target_center.copy()

    obj_trans[:] = (obj_trans @ R_w2t.T) + t_w2t + offset
    for arr in (joints_left, joints_right):
        if arr.size > 0:
            arr[:] = (arr @ R_w2t.T) + t_w2t + offset
    for arr in (contacts_left, contacts_right):
        if arr.ndim >= 2 and arr.shape[-1] >= 3:
            pts = arr[..., :3].reshape(-1, 3)
            pts[:] = (pts @ R_w2t.T) + t_w2t + offset
            arr[..., :3] = pts.reshape(arr.shape[:-1] + (3,))

    for i in range(obj_quat.shape[0]):
        qw, qx, qy, qz = obj_quat[i, 0], obj_quat[i, 1], obj_quat[i, 2], obj_quat[i, 3]
        R_obj = R.from_quat([qx, qy, qz, qw]).as_matrix()
        R_obj_genesis = R_w2t @ R_obj
        q = R.from_matrix(R_obj_genesis).as_quat()
        obj_quat[i] = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def world_to_display_positions(R_c, t_c, pts):
    """Transform positions from DexYCB world to display frame (camera + Y,Z flip).
    pts: (N, 3) or (3,). Returns same shape."""
    R_c = np.asarray(R_c, dtype=np.float64)
    t_c = np.asarray(t_c, dtype=np.float64)
    R_display = DEXYCB_DISPLAY_FLIP @ R_c.T
    t_display = -(R_display @ t_c)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    out = (pts @ R_display.T) + t_display
    return out.astype(np.float32).squeeze()


def world_to_display_quats_wxyz(R_c, t_c, quats):
    """Transform quaternions (w,x,y,z) from world to display frame.
    quats: (N, 4) or (4,). Returns same shape."""
    R_c = np.asarray(R_c, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float64)
    if quats.ndim == 1:
        quats = quats.reshape(1, -1)
    out = []
    for i in range(quats.shape[0]):
        qw, qx, qy, qz = quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]
        R_world = R.from_quat([qx, qy, qz, qw]).as_matrix()
        R_cam = R_c.T @ R_world
        R_disp = DEXYCB_DISPLAY_FLIP @ R_cam
        q_xyzw = R.from_matrix(R_disp).as_quat()
        out.append([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return np.array(out, dtype=np.float32).squeeze()


def apply_dexycb_display_to_loaded_pt(loaded_data):
    """Apply world->display transform to loaded .pt contents. No-op if data is already Genesis frame."""
    params = loaded_data.get("params") or {}
    if params.get("frame") == "genesis":
        return
    R_c = params.get("cam_R_world")
    t_c = params.get("cam_t_world")
    if R_c is None or t_c is None:
        return
    R_c = np.asarray(R_c)
    t_c = np.asarray(t_c)

    demo_data = loaded_data.get("demo_data") or {}
    if demo_data.get("obj_pos") is not None:
        demo_data["obj_pos"] = world_to_display_positions(R_c, t_c, demo_data["obj_pos"])
    if demo_data.get("obj_quat") is not None:
        demo_data["obj_quat"] = world_to_display_quats_wxyz(R_c, t_c, demo_data["obj_quat"])

    ret = loaded_data.get("retargeter_results") or {}
    for side in ret:
        # hand_qpos: (num_steps, num_dofs); first 7 DOF are base (pos 3, quat 4) for floating-base hands
        hq = ret[side].get("hand_qpos")
        if hq is not None:
            hq = np.asarray(hq)
            if hq.ndim == 2 and hq.shape[1] >= 7:
                pos = world_to_display_positions(R_c, t_c, hq[:, :3])
                quat = world_to_display_quats_wxyz(R_c, t_c, hq[:, 3:7])
                ret[side]["hand_qpos"] = np.concatenate([
                    pos.reshape(-1, 3),
                    quat.reshape(-1, 4),
                    hq[:, 7:],
                ], axis=1)


def apply_dexycb_display_to_processed_and_retargeter(loaded_data, retargeter_results):
    """Apply world->display to processed .npy and retargeter. No-op if data is already Genesis frame."""
    params = loaded_data.get("params") or {}
    if params.get("frame") == "genesis":
        return
    R_c = params.get("cam_R_world")
    t_c = params.get("cam_t_world")
    if R_c is None or t_c is None:
        return
    R_c = np.asarray(R_c)
    t_c = np.asarray(t_c)

    if params.get("obj_trans") is not None:
        params["obj_trans"] = world_to_display_positions(R_c, t_c, params["obj_trans"])
    if params.get("obj_quat") is not None:
        params["obj_quat"] = world_to_display_quats_wxyz(R_c, t_c, params["obj_quat"])

    wc = loaded_data.get("world_coord") or {}
    for key in ("joints.left", "joints.right"):
        if key in wc:
            wc[key] = world_to_display_positions(R_c, t_c, wc[key])
    for key in ("contacts.left", "contacts.right"):
        if key in wc and wc[key].ndim >= 2 and wc[key].shape[-1] >= 3:
            pts = wc[key][..., :3].reshape(-1, 3)
            tr = world_to_display_positions(R_c, t_c, pts).reshape(wc[key].shape[:-1] + (3,))
            wc[key] = np.concatenate([tr, wc[key][..., 3:]], axis=-1)

    for side in retargeter_results:
        hq = retargeter_results[side].get("hand_qpos")
        if hq is not None:
            hq = np.asarray(hq)
            if hq.ndim == 2 and hq.shape[1] >= 7:
                pos = world_to_display_positions(R_c, t_c, hq[:, :3])
                quat = world_to_display_quats_wxyz(R_c, t_c, hq[:, 3:7])
                retargeter_results[side]["hand_qpos"] = np.concatenate([
                    pos.reshape(-1, 3), quat.reshape(-1, 4), hq[:, 7:],
                ], axis=1)
