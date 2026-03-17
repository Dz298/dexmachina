"""
Debug script: curriculum kp/kd/gravity on DexYCB object (7 DOF).
With gains on, the object should stay near demo position; as gains decay it should drift.
Requires: processed DexYCB data and retarget data for the clip (run retarget_dexycb.sh first).
"""
import genesis as gs
import torch
import numpy as np

from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import get_default_robot_cfg
from dexmachina.envs.object import get_ycb_object_cfg, YCB_CLASS_NAMES
from dexmachina.envs.rewards import get_reward_cfg
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
from dexmachina.envs.curriculum import get_curriculum_cfg
from dexmachina.envs.constructors import parse_dexycb_clip
from dexmachina.asset_utils import get_asset_path

# Same clip as train_dexycb.sh; override via env or edit
CLIP = "20200709-subject-01/20200709_141754-0-63"
HAND = "orca_hand"
RETARGET_NAME = "para"


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def infer_active_sides(subject_name, sequence_id, eps=1e-4):
    """Infer active hand side(s) from temporal variation in processed MANO joints."""
    npy_path = get_asset_path(f"dexycb/processed/{subject_name}/{sequence_id}.npy")
    raw = np.load(npy_path, allow_pickle=True).item()
    world = raw["world_coord"]

    side_var = {}
    for side in ("left", "right"):
        key = f"joints.{side}"
        if key not in world:
            continue
        joints = np.asarray(world[key])
        if joints.size == 0:
            continue
        side_var[side] = float(np.mean(np.std(joints, axis=0)))

    active_sides = [side for side, var in side_var.items() if var > eps]
    if not active_sides and side_var:
        active_sides = [max(side_var, key=side_var.get)]
    if not active_sides:
        active_sides = ["left", "right"]
    return active_sides, side_var


def build_env():
    subject_name, sequence_id, start, end = parse_dexycb_clip(CLIP)
    obj_name = sequence_id
    active_sides, side_var = infer_active_sides(subject_name, sequence_id)
    print(f"Inferred active hand side(s): {active_sides} (variance={side_var})")

    demo_data = get_demo_data(
        obj_name=obj_name,
        frame_start=start,
        frame_end=end,
        hand_name=HAND,
        subject_name=subject_name,
        data_source="dexycb",
        sequence_id=sequence_id,
        hand_sides=active_sides,
    )
    ycb_class = demo_data.get("ycb_class_name")
    if not ycb_class or ycb_class not in YCB_CLASS_NAMES:
        ycb_class = "002_master_chef_can"
    demo_data["ycb_class_name"] = ycb_class

    retarget = {}
    try:
        _, retarget = load_genesis_retarget_data(
            obj_name=obj_name,
            hand_name=HAND,
            frame_start=start,
            frame_end=end,
            save_name=RETARGET_NAME,
            subject_name=subject_name,
            hand_sides=active_sides,
        )
    except FileNotFoundError as e:
        print(f"Warning: retarget data not found ({e}). Continuing without; some obs may be zeroed.")

    env_cfg = get_env_cfg()
    env_cfg.update(
        dict(
            num_envs=1,
            is_eval=True,
            record_video=True,
            episode_length=200,
        )
    )
    env_cfg["scene_kwargs"]["use_visualizer"] = True
    env_cfg["scene_kwargs"]["show_viewer"] = False
    env_cfg["scene_kwargs"]["batch_dofs_info"] = True

    robot_cfgs = {side: get_default_robot_cfg(side=side, name=HAND) for side in active_sides}
    # 7 DOF YCB (6 pose + 1 dummy joint) for VOC
    object_cfgs = {obj_name: get_ycb_object_cfg(ycb_class, voc_7dof=True)}
    object_cfgs[obj_name]["actuated"] = True
    object_cfgs[obj_name]["kp"] = 80.0
    object_cfgs[obj_name]["kv"] = 5.0
    object_cfgs[obj_name]["force_range"] = 50.0

    curr_cfg = get_curriculum_cfg(
        dict(
            wait_epochs=0,
            interval=1,
            first_stop_iter=10,
            second_stop_iter=20,
            first_ratio=0.5,
            schedule="fixed",
            fixed_mode="lin",
            gain_mode="all",
        )
    )

    return BaseEnv(
        env_cfg=env_cfg,
        robot_cfgs=robot_cfgs,
        object_cfgs=object_cfgs,
        reward_cfg=get_reward_cfg(),
        demo_data=demo_data,
        retarget_data=retarget,
        curriculum_cfg=curr_cfg,
        render_figure=False,
    )


def validate_voc_object_setup(env, pos_tol=5e-2, dof_tol=5e-2):
    """Fail-fast checks that DexYCB VOC object reset/control is consistent with demo step 0."""
    obj = env.object
    voc_dof_idxs = getattr(obj, "_voc_dof_idxs", None)
    if voc_dof_idxs is None or len(voc_dof_idxs) != 7:
        raise RuntimeError(
            f"Expected DexYCB VOC with 7 control dofs, got _voc_dof_idxs={voc_dof_idxs}"
        )
    if obj.demo_dofs is None:
        raise RuntimeError("obj.demo_dofs is None; VOC control targets were not built.")

    demo_dofs = _to_np(obj.demo_dofs)
    demo_state0 = _to_np(obj.demo_states[0])

    # Ensure VOC targets are meaningful (not collapsed to constants).
    voc_std = np.std(demo_dofs[:, :6], axis=0)
    print("VOC target std (xyz+rotvec):", np.array2string(voc_std, precision=5))
    if np.max(voc_std) < 1e-5:
        raise RuntimeError("VOC targets are near-constant; object cannot be controlled effectively.")

    # Reset and verify object aligns with demo step 0 pose/targets.
    env.reset()
    obj.update_value_buffers()
    root0 = _to_np(obj.root_pos[0])
    dof0 = _to_np(obj.entity.get_dofs_position())
    if dof0.ndim == 1:
        dof0 = dof0[None]
    dof0 = dof0[0]

    pos_err = float(np.linalg.norm(root0 - demo_state0[:3]))
    dof_err = float(np.max(np.abs(dof0[:7] - demo_dofs[0][:7])))
    print(f"VOC reset check: pos_err={pos_err:.6f}, dof_err={dof_err:.6f}")
    if pos_err > pos_tol:
        raise RuntimeError(
            f"Initial object position misaligned with demo (err={pos_err:.4f} > {pos_tol})."
        )
    if dof_err > dof_tol:
        raise RuntimeError(
            f"Initial VOC dof target misaligned with demo (err={dof_err:.4f} > {dof_tol})."
        )


def main():
    gs.init(backend=gs.gpu, logging_level="warning")
    env = build_env()
    validate_voc_object_setup(env)
    if env.object.demo_dofs is not None:
        demo_dofs = env.object.demo_dofs.detach().cpu().numpy()
        print(
            "Demo DOF target std (first 7):",
            np.array2string(np.std(demo_dofs, axis=0)[:7], precision=5),
        )
    device = env.device
    zero_actions = torch.zeros((env.num_envs, env.action_dim), device=device)

    num_epochs = 30
    steps_per_epoch = env.max_episode_length  # full episode so object can follow demo trajectory
    num_steps = min(steps_per_epoch, env.object.num_demo_frames)
    env.max_video_frames = num_epochs * num_steps
    env.start_recording()

    # Demo: initial (step 0) vs final (last step of episode). With kp/kd, object should follow demo and end near demo_final.
    demo_states = env.object.demo_states
    demo_pos_0 = demo_states[0, :3].cpu().numpy()
    final_step = min(num_steps, demo_states.shape[0]) - 1
    demo_pos_final = demo_states[final_step, :3].cpu().numpy()

    print(f"DexYCB clip={CLIP}, ycb={env.object.cfg['name']}, 7-DOF actuated.")
    print(f"Full episode: {num_steps} steps. Demo: initial pos {demo_pos_0}, target (final step {final_step}) pos {demo_pos_final}")
    print("With kp/kd on, object should follow demo → final obj pos ~ demo_final (e.g. lifted), not stuck at initial.")
    print("Epoch | kp     | kv     | gravity | obj_initial (z)  | obj_final (x,y,z)     | demo_final (z) | err_final")
    print("-" * 105)

    for epoch in range(num_epochs):
        env.reset()
        obj_pos_initial = env.object.root_pos[0].cpu().numpy().copy()

        zero_gains, gains_decayed, reason = env.curriculum.set_curriculum(epoch)
        gains = env.curriculum.get_current_gains()

        for _ in range(num_steps):
            env.step(zero_actions)

        obj_pos_final = env.object.root_pos[0].cpu().numpy()
        err_final = np.linalg.norm(obj_pos_final - demo_pos_final)
        kp = gains.get("kp", 0.0)
        kv = gains.get("kv", 0.0)
        grav = gains.get("gravity", 0.0)
        print(
            f"{epoch:5d} | {kp:6.1f} | {kv:6.1f} | {grav:7.4f} | "
            f"({obj_pos_initial[2]:.3f})        | "
            f"({obj_pos_final[0]:.3f},{obj_pos_final[1]:.3f},{obj_pos_final[2]:.3f}) | "
            f"({demo_pos_final[2]:.3f})     | {err_final:.4f}"
        )

    frames = env.get_recorded_frames(wait_for_max=False)
    if frames:
        from moviepy.editor import ImageSequenceClip
        output_path = "/tmp/gravity_test_dexycb.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(output_path)
        print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
