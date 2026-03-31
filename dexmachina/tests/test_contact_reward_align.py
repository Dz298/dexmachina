"""Tests for contact reward normal alignment (matched retarget path).

Run: python dexmachina/tests/test_contact_reward_align.py
Or:  python -m pytest ... (if pytest is installed)
"""

import torch

from dexmachina.envs.rewards import RewardModule, get_reward_cfg


def _minimal_retarget_demo(T: int = 1, n_parts: int = 2, n_links: int = 1, device: str = "cpu"):
    """Build minimal demo_data / retarget_data for RewardModule with contact reward."""
    obj_pos = torch.zeros(T, 3, device=device)
    obj_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(T, -1).clone()
    obj_arti = torch.zeros(T, device=device)

    # (T, parts, links, 4): xyz + positive contact id
    cl = torch.zeros(T, n_parts, n_links, 4, device=device)
    cl[:, 0, :, -1] = 1.0
    cl[:, 1, :, -1] = 2.0

    normals = torch.zeros(T, n_parts, n_links, 3, device=device)
    normals[..., 2] = 1.0  # outward +Z in local / world with identity object

    demo_data = {
        "obj_pos": obj_pos.cpu().numpy(),
        "obj_quat": obj_quat.cpu().numpy(),
        "obj_arti": obj_arti.cpu().numpy(),
        "contact_links_left": cl.cpu().numpy(),
        "contact_links_right": cl.cpu().numpy(),
        "contact_normals_local_left": normals.cpu().numpy(),
        "contact_normals_local_right": normals.cpu().numpy(),
    }

    wrist = torch.zeros(T, 7, device=device)
    wrist[:, 3] = 1.0  # wxyz
    retarget_data = {
        "left": {"wrist_pose": wrist},
        "right": {"wrist_pose": wrist},
    }
    return demo_data, retarget_data


def test_contact_align_disabled_when_kappa_zero():
    device = "cpu"
    demo_data, retarget_data = _minimal_retarget_demo(device=device)
    cfg = get_reward_cfg()
    cfg["contact_rew_weight"] = 1.0
    cfg["use_retarget_contact"] = True
    cfg["contact_align_kappa"] = 0.0
    m = RewardModule(cfg, demo_data, retarget_data, device)
    assert not m._contact_align_enabled

    episode = torch.zeros(2, dtype=torch.long)
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    contacts = torch.zeros(2, 2, 1, 4)
    contacts[..., 2] = 0.05  # +Z from demo (aligned)
    valids = torch.ones(2, 2, 1, dtype=torch.bool)
    dist, align = m.compute_matched_contact_per_hand(
        contacts, valids, episode, pose, pose, side="left"
    )
    assert dist.shape == (2, 2, 1)
    assert torch.allclose(align, torch.ones_like(align))


def test_contact_align_enabled_and_scales_reward():
    device = "cpu"
    demo_data, retarget_data = _minimal_retarget_demo(device=device)
    kappa = 3.0
    cfg = get_reward_cfg()
    cfg["contact_rew_weight"] = 1.0
    cfg["use_retarget_contact"] = True
    cfg["contact_align_kappa"] = kappa
    m = RewardModule(cfg, demo_data, retarget_data, device)
    assert m._contact_align_enabled

    episode = torch.zeros(2, dtype=torch.long)
    demo_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

    # Env 0: policy on +Z from demo (aligned with normal +Z)
    # Env 1: policy on +X from demo (orthogonal → dot 0)
    contacts = torch.zeros(2, 2, 1, 4)
    contacts[0, :, :, 2] = 0.05
    contacts[1, :, :, 0] = 0.05
    valids = torch.ones(2, 2, 1, dtype=torch.bool)

    dist, align = m.compute_matched_contact_per_hand(
        contacts, valids, episode, demo_pose, demo_pose, side="left"
    )
    # Same distance magnitude for both envs (0.05 per link)
    assert torch.allclose(dist[0], dist[1])
    # Aligned env: weight 1; misaligned: exp(-kappa)
    assert align[0].mean() > 0.99
    expected_low = float(torch.exp(torch.tensor(-kappa)))
    assert align[1].mean() < 0.05
    assert torch.allclose(align[1].mean(), torch.tensor(expected_low), rtol=1e-4, atol=1e-5)

    obj_pose = demo_pose.clone()
    con_rew, rew = m.compute_matched_contact_reward(
        contacts,
        valids,
        contacts,
        valids,
        obj_pose,
        demo_pose,
        episode,
    )
    assert "contact_align_left_bottom" in rew
    assert con_rew.shape == (2,)
    # Matched contact reward is higher when approach matches demo normal (+Z).
    assert con_rew[0].item() > con_rew[1].item()


def test_contact_align_kappa_without_normals_disables():
    device = "cpu"
    demo_data, retarget_data = _minimal_retarget_demo(device=device)
    del demo_data["contact_normals_local_left"]
    del demo_data["contact_normals_local_right"]
    cfg = get_reward_cfg()
    cfg["contact_rew_weight"] = 1.0
    cfg["use_retarget_contact"] = True
    cfg["contact_align_kappa"] = 2.0
    m = RewardModule(cfg, demo_data, retarget_data, device)
    assert not m._contact_align_enabled


if __name__ == "__main__":
    test_contact_align_disabled_when_kappa_zero()
    test_contact_align_enabled_and_scales_reward()
    test_contact_align_kappa_without_normals_disables()
    print("test_contact_reward_align: all passed")
