"""Shared contact approximation utilities for ARCTIC and DexYCB processing."""
import numpy as np
from sklearn.neighbors import KDTree

MANO_HAND_LINKS = [
    ("palm", [0, 13, 1, 4, 10, 7]),
    ("thumb1", [13, 14]),
    ("thumb2", [14, 15]),
    ("thumb3", [15, 16]),
    ("index1", [1, 2]),
    ("index2", [2, 3]),
    ("index3", [3, 17]),
    ("middle1", [4, 5]),
    ("middle2", [5, 6]),
    ("middle3", [6, 18]),
    ("ring1", [10, 11]),
    ("ring2", [11, 12]),
    ("ring3", [12, 19]),
    ("pinky1", [7, 8]),
    ("pinky2", [8, 9]),
    ("pinky3", [9, 20]),
]


def approximate_contact_with_id(
    obj_verts, obj_part_ids, hand_verts, threshold=0.01, dist_min=0, dist_max=100
):
    """
    Return contact points on object and hand with part id.
    Returns (num_contact, 4), (num_contact, 4) for object and hand.
    """
    tree_hand = KDTree(hand_verts)
    dist, idx = tree_hand.query(obj_verts, k=1)
    dist = np.clip(dist, dist_min, dist_max)
    contact_mask = dist.ravel() < threshold
    if not np.any(contact_mask):
        return np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)
    contact_on_hand = hand_verts[idx[contact_mask].ravel()]
    contact_on_obj = obj_verts[contact_mask]
    part_id_on_obj = obj_part_ids[contact_mask]
    return (
        np.concatenate([contact_on_obj, part_id_on_obj[:, None]], axis=-1),
        np.concatenate([contact_on_hand, part_id_on_obj[:, None]], axis=-1),
    )


def find_closest_link(contact_points, joint_points):
    """
    Given (N, 3 or 4) contact points and (21, 3) joint points, return
    (num_links, 4) averaged contact position per link with part id, or zeros.
    """
    if contact_points.size == 0:
        return [np.zeros(4) for _ in range(len(MANO_HAND_LINKS))]
    all_avg_dists = []
    for idx, (_, joint_idxs) in enumerate(MANO_HAND_LINKS):
        dists = []
        for joint_idx in joint_idxs:
            joint_point = joint_points[joint_idx]
            dist = np.linalg.norm(contact_points[:, :3] - joint_point, axis=-1)
            dists.append(dist)
        avg_dist = np.mean(np.stack(dists, axis=0), axis=0)
        all_avg_dists.append(avg_dist)
    all_avg_dists = np.stack(all_avg_dists, axis=0)
    closest_link_idx = np.argmin(all_avg_dists, axis=0)
    closest_link_dist = np.min(all_avg_dists, axis=0)
    avg_contact_positions = [np.zeros(4) for _ in range(len(MANO_HAND_LINKS))]
    for idx, (_, joint_idxs) in enumerate(MANO_HAND_LINKS):
        if not np.any(closest_link_idx == idx):
            continue
        mask = closest_link_idx == idx
        part_ids = contact_points[mask, 3]
        voted_part_id = np.argmax(np.bincount(part_ids.astype(int)))
        weighted_avg_position = np.average(
            contact_points[mask][:, :3], axis=0, weights=1 / (closest_link_dist[mask] + 1e-8)
        )
        avg_contact_positions[idx] = np.concatenate([weighted_avg_position, [voted_part_id]])
    return avg_contact_positions
