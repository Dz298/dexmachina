import numpy as np
import torch

from dexmachina.envs.math_utils import matrix_from_quat


def points_world_to_local_np(points_world, part_pos, part_quat):
    points_world = np.asarray(points_world, dtype=np.float32)
    part_pos = np.asarray(part_pos, dtype=np.float32)
    part_quat = np.asarray(part_quat, dtype=np.float32)
    rot = matrix_from_quat(torch.from_numpy(part_quat).unsqueeze(0)).squeeze(0).cpu().numpy()
    return (points_world - part_pos[None]) @ rot


def points_local_to_world_torch(points_local: torch.Tensor, part_pos: torch.Tensor, part_quat: torch.Tensor) -> torch.Tensor:
    rot = matrix_from_quat(part_quat)
    return torch.matmul(points_local, rot.transpose(1, 2)) + part_pos.unsqueeze(1)


def normals_local_to_world_torch(normals_local: torch.Tensor, part_quat: torch.Tensor) -> torch.Tensor:
    rot = matrix_from_quat(part_quat)
    normals = torch.matmul(normals_local, rot.transpose(1, 2))
    return torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)


def compute_virtual_force(
    link_pos: torch.Tensor,
    link_vel: torch.Tensor,
    contact_pos: torch.Tensor,
    contact_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float,
    delta: float,
    kp: float,
    kd: float,
    sigma: float,
    fmax: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        return torch.zeros_like(link_pos)

    target_pos = contact_pos - delta * contact_normal
    error = target_pos - link_pos
    assist_force = kp * error - kd * link_vel

    if sigma > 0.0:
        dist = torch.norm(contact_pos - link_pos, dim=-1)
        gate = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
        assist_force = assist_force * gate.unsqueeze(-1)

    assist_force = assist_force * alpha
    assist_force = torch.where(valid_mask.unsqueeze(-1), assist_force, torch.zeros_like(assist_force))

    if fmax > 0.0:
        force_norm = torch.norm(assist_force, dim=-1, keepdim=True)
        clip_scale = torch.clamp(fmax / torch.clamp(force_norm, min=1e-6), max=1.0)
        assist_force = assist_force * clip_scale

    return assist_force
