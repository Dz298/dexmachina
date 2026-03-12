import torch
import numpy as np
import random
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import get_default_robot_cfg
from dexmachina.envs.demo_data import load_genesis_retarget_data, load_contact_retarget_data
from dexmachina.envs.rewards import get_reward_cfg
from dexmachina.envs.object import get_arctic_object_cfg
import genesis as gs

# HAND_NAME = "orca_hand"
HAND_NAME = "allegro_hand"
OBJECT_NAME = "ketchup"
FRAME_START = 160
FRAME_END = 260
RETARGET_SAVE_NAME = "para"
RENDER_RES = 512
REFINE_MODE = "virtual_force"  # one of: sampling, ik, virtual_force
RUN_ANNEAL_VIS = False
ANNEAL_VIS_STEPS = 50
RUN_OPT_QUALITY = False
NUM_TEST_FRAMES = 5
TIGHT_SQUEEZE_DEBUG = True
DEBUG_OBJECT_STATE_MATCH = True

TIGHT_SQUEEZE_CFG = dict(
    # Pre-tension phase: pinned, squeeze + J^T attract
    valid_grasp_vf_attract_steps=40,       # pre-tension steps (pinned)
    valid_grasp_vf_attract_gain=0.5,       # J^T attraction gain per step
    valid_grasp_vf_attract_dq_clip=0.03,   # max joint delta per attract step
    valid_grasp_vf_close_bias=0.15,        # total squeeze bias accumulated over pre-tension steps
    valid_grasp_vf_squeeze_torque=0.8,     # direct torque on flex joints during pre-tension
    # Soft-unpin phase
    valid_grasp_vf_soft_unpin_steps=50,    # steps with decreasing gravity compensation
    valid_grasp_vf_hold_steps=20,          # free-physics steps after gravity ramp-down
    # Validation gates (tighter than defaults)
    valid_grasp_hold_max_drop=0.1,        # max Z-drop (m)
    valid_grasp_hold_max_vel=1.0,         # max linear velocity (m/s)
    valid_grasp_hold_max_ang_vel=15,      # max angular velocity (rad/s)
    valid_grasp_functional_opposition_min=0.3,  # antipodal score gate
)

def build_env(enforce=True):
    env_cfg = get_env_cfg()
    env_cfg.update(
        dict(
            num_envs=1,
            is_eval=True,
            record_video=True,  # so front camera is built
            rand_init_ratio=1.0,
            enforce_valid_rand_init_grasp=enforce,
            valid_grasp_min_contacts=3,
            valid_grasp_require_both_hands=False,
            valid_grasp_contact_thresh=0.01,
            valid_grasp_sample_std=0.08,
            valid_grasp_guided_gain=0.25,
            valid_grasp_guided_abd_gain=0.12,
            valid_grasp_sample_count=128,
            valid_grasp_sample_iters=3,
            valid_grasp_refine_mode=REFINE_MODE,
            valid_grasp_use_ik=(REFINE_MODE == "ik"),
            valid_grasp_ik_iters=50,
            valid_grasp_ik_damping=1e-2,
            valid_grasp_ik_step_clip=0.15,
            valid_grasp_ik_close_bias=0.03,
            valid_grasp_settle_steps=20,
            valid_grasp_slip_weight=10.0,   # high weight: slip matters more than alignment
            valid_grasp_vel_weight=2.0,
            valid_grasp_ang_vel_weight=0.5,
            valid_grasp_opposition_weight=0.0,
            valid_grasp_hold_max_drop=0.01,
            valid_grasp_hold_max_vel=0.30,
            valid_grasp_hold_max_ang_vel=2.0,
            valid_grasp_hold_ignore_steps=0,
            valid_grasp_contact_persistence_min=0.0,
            valid_grasp_vf_close_bias=0.02,
            valid_grasp_vf_squeeze_torque=0.0,
            valid_grasp_gravity_ramp_steps=0,
            valid_grasp_vf_attract_steps=50,
            valid_grasp_vf_hold_steps=50,
            valid_grasp_vf_anneal_probe_hold_steps=8,
            valid_grasp_debug=True,
            observe_tip_dist=True,
            episode_length=100,
        )
    )
    if TIGHT_SQUEEZE_DEBUG:
        env_cfg.update(TIGHT_SQUEEZE_CFG)
        print(f"[debug_grasp] tight squeeze enabled: {TIGHT_SQUEEZE_CFG}")
    env_cfg['scene_kwargs']['use_visualizer'] = True  
    env_cfg['scene_kwargs']['show_viewer'] = False
    env_cfg['record_video'] = True 
    print(f"Setting render resolution to {RENDER_RES}") 
    env_cfg['camera_kwargs']['front'] = dict(
        res=(RENDER_RES, RENDER_RES),
        fov=40,
        pos=(0.5, -1.5, 1.2),
        lookat=(0.0, -1.58, 2.0),
    ) 
    demo_data, retarget = load_genesis_retarget_data(
        obj_name=OBJECT_NAME,
        hand_name=HAND_NAME,
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        save_name=RETARGET_SAVE_NAME,
    )
    contact_data = load_contact_retarget_data(
        obj_name=OBJECT_NAME,
        hand_name=HAND_NAME,
        frame_start=FRAME_START,
        frame_end=FRAME_END,
    )
    demo_data.update(contact_data)
    robot_cfgs = {
        "left": get_default_robot_cfg(side="left", name=HAND_NAME),
        "right": get_default_robot_cfg(side="right", name=HAND_NAME),
    }
    object_cfgs = {
        OBJECT_NAME: get_arctic_object_cfg(name=OBJECT_NAME),
    }
    reward_cfg = get_reward_cfg()
    reward_cfg["contact_rew_weight"] = 1.0
    reward_cfg["use_retarget_contact"] = True
    return BaseEnv(
        env_cfg=env_cfg,
        robot_cfgs=robot_cfgs,
        object_cfgs=object_cfgs,
        reward_cfg=reward_cfg,
        demo_data=demo_data,
        retarget_data=retarget,
        render_figure=False,
    )

def write_video(frames_out, output_path, fps):
    if not frames_out:
        return
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames_out, fps=fps)
    clip.write_videofile(output_path)
    print(f"saved video to {output_path}")

def _quat_angle_error_rad(q_a, q_b):
    q_a = q_a / torch.clamp(torch.norm(q_a), min=1e-8)
    q_b = q_b / torch.clamp(torch.norm(q_b), min=1e-8)
    dot = torch.clamp(torch.abs(torch.sum(q_a * q_b)), -1.0, 1.0)
    return float((2.0 * torch.acos(dot)).item())

def debug_object_state_vs_demo(env, req_frame_idx, stage):
    if env.object is None:
        print(f"[state-check] {stage}: object=None")
        return
    if env.object.demo_states is None or int(env.object.num_demo_frames) <= 0:
        print(f"[state-check] {stage}: no demo_states")
        return

    req_frame_idx = int(req_frame_idx)
    demo_idx = int(np.clip(req_frame_idx, 0, int(env.object.num_demo_frames) - 1))
    demo_state = env.object.demo_states[demo_idx]

    curr_pos = env.object.root_pos[0].detach().clone()
    curr_quat = env.object.root_quat[0].detach().clone()
    curr_dof = env.object.dof_pos[0].detach().clone()
    demo_pos = demo_state[:3].detach().clone()
    demo_quat = demo_state[3:7].detach().clone()
    demo_dof = demo_state[7: 7 + env.object.num_joints].detach().clone()

    pos_err = float(torch.norm(curr_pos - demo_pos).item())
    quat_err_rad = _quat_angle_error_rad(curr_quat, demo_quat)
    quat_err_deg = float(quat_err_rad * 180.0 / np.pi)
    dof_abs = torch.abs(curr_dof - demo_dof)
    dof_mae = float(torch.mean(dof_abs).item()) if dof_abs.numel() > 0 else 0.0
    dof_max = float(torch.max(dof_abs).item()) if dof_abs.numel() > 0 else 0.0
    ep_start = int(env.episode_start_buf[0].item())
    print(
        f"[state-check] {stage}: req={req_frame_idx:03d} ep_start={ep_start:03d} demo_idx={demo_idx:03d} "
        f"pos_err={pos_err:.4f} quat_err_deg={quat_err_deg:.2f} dof_mae={dof_mae:.4f} dof_max={dof_max:.4f}"
    )
    if pos_err > 0.05 or quat_err_deg > 10.0:
        print(
            f"[state-check] {stage}: MISMATCH "
            f"curr_pos={curr_pos.tolist()} demo_pos={demo_pos.tolist()}"
        )

def force_reset_to_frame(env, frame_idx, apply_opt=True, allow_fallback=False):
    saved_rand_ratio = env.rand_init_ratio
    env.rand_init_ratio = 0.0
    env.reset_idx([0])
    env.rand_init_ratio = saved_rand_ratio
    env.episode_start_buf[0] = int(frame_idx)
    env.episode_length_buf[0] = int(frame_idx)
    episode_start = env.episode_start_buf[0:1]
    for _, robot in env.robots.items():
        robot.reset_idx(env_idxs=[0], episode_start=episode_start)
    for _, obj in env.objects.items():
        obj.reset_idx(env_idxs=[0], episode_start=episode_start)
    if DEBUG_OBJECT_STATE_MATCH:
        debug_object_state_vs_demo(env, frame_idx, stage="after_reset_before_opt")
    if apply_opt and env.enforce_valid_rand_init_grasp:
        saved_fallback = env.valid_grasp_fallback_to_zero
        if not allow_fallback:
            env.valid_grasp_fallback_to_zero = False
        env._ensure_valid_rand_init_grasp([0], [True])
        env.valid_grasp_fallback_to_zero = saved_fallback
        if DEBUG_OBJECT_STATE_MATCH:
            debug_object_state_vs_demo(env, frame_idx, stage="after_optimize")

def snapshot_hold_cycles(env, tag, frames, steps_per_cycle=120, seed=0, apply_opt=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.max_video_frames = max(env.max_video_frames, len(frames) * steps_per_cycle * 2)
    env.start_recording()
    viewer = getattr(env.scene, "viewer", None)
    if viewer is not None and getattr(viewer, "camera", None) is not None:
        viewer.camera.follow(env.robots["left"].entity)

    for frame_idx in frames:
        force_reset_to_frame(env, frame_idx, apply_opt=apply_opt, allow_fallback=False)
        env._compute_intermediate_values()
        hold_targets = {
            name: robot.dof_pos[0].clone()
            for name, robot in env.robots.items()
        }
        qpos_mean = float(torch.mean(torch.cat([v for v in hold_targets.values()])).item())
        print(f"[hold] frame={frame_idx} ep_start={int(env.episode_start_buf[0].item())} qpos_mean={qpos_mean:.4f}")
        for _ in range(steps_per_cycle):
            for name, robot in env.robots.items():
                robot.control_joint_position(hold_targets[name][None], env_idxs=[0])
            for _, obj in env.objects.items():
                obj.step()
            env.scene.step()
            env.episode_length_buf += 1
            env._compute_intermediate_values()
            env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
            env.reset_buf[:] = env.reset_terminated | env.reset_time_outs
            if env.record_video:
                env._render_headless()

    frames_out = env.get_recorded_frames(wait_for_max=False)
    if frames_out:
        output_path = f"/tmp/{tag}_hold.mp4"
        write_video(frames_out, output_path, fps=int(1 / env.dt))
        print(f"{tag}: saved video to {output_path}")

def visualize_close_annealing(env, frame_idx, steps=50, seed=0, tag="anneal_close", probe_hold_steps=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    force_reset_to_frame(env, frame_idx, apply_opt=False)
    env._compute_intermediate_values()
    targets = env._get_demo_contact_targets_for_env(0)
    if len(targets) == 0:
        print(f"[anneal-vis] frame={frame_idx}: no contact targets")
        return

    changed = env._solve_contact_ik_lm_for_env(0, targets)
    if not changed:
        changed = env._solve_contact_ik_for_env(0, targets)
    if not changed:
        print(f"[anneal-vis] frame={frame_idx}: IK failed")
        return

    env._compute_intermediate_values()
    ik_err, ik_cnt, _ = env._contact_alignment_error(0, targets)
    ik_per_link = ik_err / max(ik_cnt, 1)
    print(f"[anneal-vis] frame={frame_idx} IK per_link_err={ik_per_link:.6f} count={ik_cnt}")

    if env.object is None or steps <= 0:
        return
    steps = int(steps)
    close_bias = float(env.valid_grasp_vf_close_bias)
    pin_pos = env.object.root_pos[0].clone()
    pin_quat = env.object.root_quat[0].clone()
    pin_dof = env.object.dof_pos[0].clone()

    active_fingers = env._get_active_contact_fingers(targets)
    side_meta = {}
    for side, robot in env.robots.items():
        q = robot.dof_pos[0].clone()
        name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
        finger_groups = robot.get_finger_joint_groups()
        joint_to_finger = {}
        for finger_name, joints in finger_groups.items():
            for ji in joints:
                joint_to_finger[int(ji)] = finger_name
        active_side = set(active_fingers.get(side, []))
        flex_idxs = []
        for dof_idx in robot.actuated_dof_idxs:
            lname = name_by_idx.get(dof_idx, "").lower()
            if "abd" in lname or "spread" in lname or "forearm" in lname:
                continue
            finger_name = joint_to_finger.get(int(dof_idx), None)
            if len(active_side) > 0 and finger_name not in active_side:
                continue
            flex_idxs.append(int(dof_idx))
        if len(flex_idxs) == 0:
            continue
        probe_delta = max(0.01, 0.5 * close_bias)
        close_signs = {}
        for dof_idx in flex_idxs:
            base = q.clone()
            plus = q.clone()
            minus = q.clone()
            plus[dof_idx] = torch.clamp(
                plus[dof_idx] + probe_delta, robot.dof_limits[dof_idx, 0], robot.dof_limits[dof_idx, 1]
            )
            minus[dof_idx] = torch.clamp(
                minus[dof_idx] - probe_delta, robot.dof_limits[dof_idx, 0], robot.dof_limits[dof_idx, 1]
            )
            robot.set_joint_position(plus[None], env_idxs=[0])
            plus_err, plus_cnt, _ = env._contact_alignment_error(0, targets)
            plus_metric = plus_err / max(plus_cnt, 1) if plus_cnt > 0 else float("inf")
            robot.set_joint_position(minus[None], env_idxs=[0])
            minus_err, minus_cnt, _ = env._contact_alignment_error(0, targets)
            minus_metric = minus_err / max(minus_cnt, 1) if minus_cnt > 0 else float("inf")
            close_signs[dof_idx] = 1.0 if plus_metric <= minus_metric else -1.0
            robot.set_joint_position(base[None], env_idxs=[0])
        side_meta[side] = dict(robot=robot, base_q=q, flex_idxs=flex_idxs, close_signs=close_signs)
    if len(side_meta) == 0:
        print(f"[anneal-vis] frame={frame_idx}: no flex joints")
        return

    env.max_video_frames = max(env.max_video_frames, steps + 8)
    env.start_recording()
    if env.record_video:
        env._render_headless()

    def _candidate_metrics():
        env._compute_intermediate_values()
        err, cnt, _ = env._contact_alignment_error(0, targets)
        per_link = err / max(cnt, 1) if cnt > 0 else float("inf")
        side_contacts = env._count_actual_contact_pairs_by_side(0, targets=targets)
        contacts = int(sum(side_contacts.values()))
        contacts_ok = env._passes_contact_gate(side_contacts)
        stable = True
        drop = 0.0
        vel = 0.0
        ang_vel = 0.0
        if probe_hold_steps > 0:
            probe_state = env._snapshot_env_state(0)
            hold_targets = {name: robot.dof_pos[0].clone() for name, robot in env.robots.items()}
            min_z, max_vel, max_ang_vel = env._simulate_settle(0, hold_targets, int(probe_hold_steps))
            stable, drop, vel, ang_vel = env._passes_stability_gate(
                0, float(pin_pos[2].item()), min_z=min_z, max_vel=max_vel, max_ang_vel=max_ang_vel
            )
            env._restore_env_state(0, probe_state)
            env._compute_intermediate_values()
        return per_link, contacts, side_contacts, contacts_ok, stable, drop, vel, ang_vel

    def _is_better(curr, best):
        def _rank(v):
            per_link, contacts, _, contacts_ok, stable, drop, vel, ang_vel = v
            if stable and contacts_ok:
                return (2, -drop, -vel, -ang_vel, contacts, -per_link)
            if contacts_ok:
                return (1, -per_link, contacts, -drop, -vel, -ang_vel)
            return (0, -per_link, contacts, -drop, -vel, -ang_vel)

        return _rank(curr) > _rank(best)

    best = _candidate_metrics()
    best_step = 0
    for step_i in range(steps):
        alpha = float(step_i + 1) / float(steps)
        env.object.set_object_state(pin_pos[None], pin_quat[None], pin_dof[None], env_idxs=[0])
        for _, sm in side_meta.items():
            robot = sm["robot"]
            target = sm["base_q"].clone()
            for dof_idx in sm["flex_idxs"]:
                sign = sm["close_signs"].get(dof_idx, 1.0)
                target[dof_idx] += sign * alpha * close_bias
            target = torch.clamp(target, robot.dof_limits[:, 0], robot.dof_limits[:, 1])
            robot.control_joint_position(target[None], env_idxs=[0])
        for _, obj in env.objects.items():
            obj.step()
        env.scene.step()
        env.object.set_object_state(pin_pos[None], pin_quat[None], pin_dof[None], env_idxs=[0])
        curr = _candidate_metrics()
        if _is_better(curr, best):
            best = curr
            best_step = step_i + 1
        print(
            f"[anneal-vis] frame={frame_idx} step={step_i} alpha={alpha:.2f} "
            f"per_link_err={curr[0]:.6f} contacts={curr[1]} side_contacts={curr[2]} contacts_ok={curr[3]} stable={curr[4]} "
            f"drop={curr[5]:.4f} vel={curr[6]:.4f} ang_vel={curr[7]:.4f} "
            f"best_per_link={best[0]:.6f} best_contacts={best[1]} best_side_contacts={best[2]} "
            f"best_contacts_ok={best[3]} best_stable={best[4]}"
        )
        if env.record_video:
            env._render_headless()

    print(
        f"[anneal-vis] frame={frame_idx} selected_step={best_step}/{steps} "
        f"best_per_link={best[0]:.6f} best_contacts={best[1]} best_side_contacts={best[2]} "
        f"best_contacts_ok={best[3]} best_stable={best[4]} "
        f"best_drop={best[5]:.4f} best_vel={best[6]:.4f} best_ang_vel={best[7]:.4f}"
    )
    frames_out = env.get_recorded_frames(wait_for_max=False)
    if frames_out:
        out = f"/tmp/{tag}_frame{frame_idx}.mp4"
        write_video(frames_out, out, fps=int(1 / env.dt))

def evaluate_opt_quality(raw_env, opt_env, frames, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    errors_raw = []
    errors_opt = []
    scores_raw = []
    scores_opt = []
    qpos_l1_deltas = []
    obj_pos_deltas = []
    contacts_raw = []
    contacts_opt = []
    warned = False
    for frame_idx in frames:
        force_reset_to_frame(raw_env, frame_idx, apply_opt=False)
        raw_env._compute_intermediate_values()
        raw_targets = raw_env._get_demo_contact_targets_for_env(0)
        err_raw = raw_env._evaluate_grasp_quality(0, raw_targets)
        if err_raw is None or not np.isfinite(err_raw):
            err_raw = raw_env._keypoint_grasp_distance(0)
        raw_init_obj_z = float(raw_env.object.root_pos[0, 2].item()) if raw_env.object is not None else 0.0
        score_raw, _, raw_count = raw_env._score_candidate(0, raw_targets, raw_init_obj_z)
        raw_contact_pairs = raw_env._count_actual_contact_pairs(0, targets=raw_targets)

        force_reset_to_frame(opt_env, frame_idx, apply_opt=True, allow_fallback=False)
        opt_env._compute_intermediate_values()
        opt_targets = opt_env._get_demo_contact_targets_for_env(0)
        err_opt = opt_env._evaluate_grasp_quality(0, opt_targets)
        if err_opt is None or not np.isfinite(err_opt):
            err_opt = opt_env._keypoint_grasp_distance(0)
        opt_init_obj_z = float(opt_env.object.root_pos[0, 2].item()) if opt_env.object is not None else 0.0
        score_opt, _, opt_count = opt_env._score_candidate(0, opt_targets, opt_init_obj_z)
        opt_contact_pairs = opt_env._count_actual_contact_pairs(0, targets=opt_targets)

        if err_raw is None or err_opt is None:
            continue
        if (not np.isfinite(err_raw) or not np.isfinite(err_opt)):
            if not warned:
                warned = True
                link_nan = False
                for side, _ in opt_env.contact_link_meta.items():
                    robot = opt_env.robots.get(side)
                    if robot is None:
                        continue
                    link_pos = robot.entity.get_links_pos()
                    link_nan = link_nan or torch.isnan(link_pos[0]).any().item()
                obj_nan = False
                if opt_env.object is not None:
                    obj_nan = torch.isnan(opt_env.object.root_pos[0]).any().item() or torch.isnan(opt_env.object.root_lin_vel[0]).any().item()
                print(f"[warn] invalid opt-quality values; link_nan={link_nan}, obj_nan={obj_nan}, raw_count={raw_count}, opt_count={opt_count}")
            continue
        if raw_count == 0:
            score_raw = err_raw
        if opt_count == 0:
            score_opt = err_opt
        if not np.isfinite(score_raw):
            score_raw = err_raw
        if not np.isfinite(score_opt):
            score_opt = err_opt
        errors_raw.append(err_raw)
        errors_opt.append(err_opt)
        scores_raw.append(score_raw)
        scores_opt.append(score_opt)
        contacts_raw.append(raw_contact_pairs)
        contacts_opt.append(opt_contact_pairs)
        raw_q = torch.cat([raw_env.robots["left"].dof_pos[0], raw_env.robots["right"].dof_pos[0]], dim=0)
        opt_q = torch.cat([opt_env.robots["left"].dof_pos[0], opt_env.robots["right"].dof_pos[0]], dim=0)
        qpos_l1_deltas.append(float(torch.mean(torch.abs(opt_q - raw_q)).item()))
        if raw_env.object is not None and opt_env.object is not None:
            obj_pos_deltas.append(float(torch.norm(opt_env.object.root_pos[0] - raw_env.object.root_pos[0]).item()))
    if len(errors_raw) == 0:
        return None
    eps = 1e-6
    improved = sum(1 for b, a in zip(errors_raw, errors_opt) if a < b - eps)
    equal = sum(1 for b, a in zip(errors_raw, errors_opt) if abs(a - b) <= eps)
    score_improved = sum(1 for b, a in zip(scores_raw, scores_opt) if a < b - eps)
    score_equal = sum(1 for b, a in zip(scores_raw, scores_opt) if abs(a - b) <= eps)
    contact_improved = sum(1 for b, a in zip(contacts_raw, contacts_opt) if a > b)
    contact_equal = sum(1 for b, a in zip(contacts_raw, contacts_opt) if a == b)
    return dict(
        mean_raw=float(np.mean(errors_raw)),
        mean_opt=float(np.mean(errors_opt)),
        median_raw=float(np.median(errors_raw)),
        median_opt=float(np.median(errors_opt)),
        score_mean_raw=float(np.mean(scores_raw)),
        score_mean_opt=float(np.mean(scores_opt)),
        score_median_raw=float(np.median(scores_raw)),
        score_median_opt=float(np.median(scores_opt)),
        contact_mean_raw=float(np.mean(contacts_raw)),
        contact_mean_opt=float(np.mean(contacts_opt)),
        contact_median_raw=float(np.median(contacts_raw)),
        contact_median_opt=float(np.median(contacts_opt)),
        improved=improved,
        equal=equal,
        score_improved=score_improved,
        score_equal=score_equal,
        contact_improved=contact_improved,
        contact_equal=contact_equal,
        mean_qpos_l1_delta=float(np.mean(qpos_l1_deltas)) if len(qpos_l1_deltas) > 0 else 0.0,
        mean_obj_pos_delta=float(np.mean(obj_pos_deltas)) if len(obj_pos_deltas) > 0 else 0.0,
        total=len(errors_raw),
    )

def run_opt_quality_test(raw_env, opt_env, frames, seed=0):
    stats = evaluate_opt_quality(raw_env, opt_env, frames, seed=seed)
    if stats is None:
        print("opt quality: no valid frames")
        return
    print(
        "opt quality raw->opt: "
        f"mean {stats['mean_raw']:.4f} -> {stats['mean_opt']:.4f}, "
        f"median {stats['median_raw']:.4f} -> {stats['median_opt']:.4f}, "
        f"improved {stats['improved']}/{stats['total']} equal {stats['equal']}/{stats['total']}; "
        f"score mean {stats['score_mean_raw']:.4f} -> {stats['score_mean_opt']:.4f}, "
        f"score median {stats['score_median_raw']:.4f} -> {stats['score_median_opt']:.4f}, "
        f"score improved {stats['score_improved']}/{stats['total']} equal {stats['score_equal']}/{stats['total']}; "
        f"actual contacts mean {stats['contact_mean_raw']:.2f} -> {stats['contact_mean_opt']:.2f}, "
        f"actual contacts median {stats['contact_median_raw']:.2f} -> {stats['contact_median_opt']:.2f}, "
        f"contact improved {stats['contact_improved']}/{stats['total']} equal {stats['contact_equal']}/{stats['total']}; "
        f"mean qpos_l1_delta {stats['mean_qpos_l1_delta']:.6f} mean obj_pos_delta {stats['mean_obj_pos_delta']:.6f}"
    )

def debug_contact_targets(env, frames):
    for frame_idx in frames:
        force_reset_to_frame(env, frame_idx, apply_opt=False)
        env._compute_intermediate_values()
        targets = env._get_demo_contact_targets_for_env(0)
        left_valid = 0
        right_valid = 0
        for side in ["left", "right"]:
            t = targets.get(side, None)
            if t is None:
                continue
            valid = t.get("valid", None)
            if valid is None:
                continue
            count = int(valid.sum().item())
            if side == "left":
                left_valid = count
            else:
                right_valid = count
        has_left = "contact_links_left" in env.reward_module.demo_tensors
        has_right = "contact_links_right" in env.reward_module.demo_tensors
        meta_left = "left" in env.contact_link_meta
        meta_right = "right" in env.contact_link_meta
        missing_left = 0
        missing_right = 0
        for side in ["left", "right"]:
            demo_side = env.demo_data.get(side, {})
            link_names = demo_side.get("collision_link_names", [])
            robot = env.robots.get(side)
            if robot is None or len(link_names) == 0:
                continue
            missing = [name for name in link_names if name not in robot.link_name_to_local_idx]
            if side == "left":
                missing_left = len(missing)
            else:
                missing_right = len(missing)
        left_keys = list(env.demo_data.get("left", {}).keys())
        right_keys = list(env.demo_data.get("right", {}).keys())
        left_link_len = len(env.demo_data.get("left", {}).get("collision_link_names", []))
        right_link_len = len(env.demo_data.get("right", {}).get("collision_link_names", []))
        print(
            f"frame {frame_idx}: valid_contacts left={left_valid} right={right_valid} "
            f"demo_tensors left={has_left} right={has_right} meta left={meta_left} right={meta_right} "
            f"missing_links left={missing_left} right={missing_right} "
            f"collision_link_names left={left_link_len} right={right_link_len} "
            f"demo_keys left={left_keys} right={right_keys}"
        )

if __name__ == "__main__":
    seed = 42
    gs.init(backend=gs.gpu, logging_level='warning')
    raw_env = build_env(enforce=False)
    opt_env = build_env(enforce=True)
    end_t = min(100, raw_env.max_episode_length - 1)
    rng = random.Random(seed)
    frames = [rng.randint(0, end_t - 1) for _ in range(NUM_TEST_FRAMES)]
    if RUN_ANNEAL_VIS and len(frames) > 0:
        visualize_close_annealing(
            opt_env,
            frame_idx=frames[0],
            steps=ANNEAL_VIS_STEPS,
            seed=seed + 123,
            tag="anneal_close",
            probe_hold_steps=8,
        )
        # Anneal visualization can drive the solver into unstable states; rebuild before quality eval.
        opt_env = build_env(enforce=True)
    if RUN_OPT_QUALITY:
        run_opt_quality_test(raw_env, opt_env, frames, seed=seed + 10)
    snapshot_hold_cycles(opt_env, "after_opt", frames, steps_per_cycle=120, seed=seed, apply_opt=True)
    snapshot_hold_cycles(raw_env, "before_opt", frames, steps_per_cycle=120, seed=seed, apply_opt=False)
