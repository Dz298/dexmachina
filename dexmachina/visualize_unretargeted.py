#!/usr/bin/env python3
"""
Headless visualizer for unretargeted (human hand) data from processed .npy files.
Renders human hand joints as spheres + object. Supports DexYCB and ARCTIC.

process_dexycb.py outputs DexYCB data already in Genesis frame; ARCTIC is also Genesis-compatible.
"""
import os
import argparse
import numpy as np
import torch
import genesis as gs
from tqdm import tqdm

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg, get_ycb_object_cfg
from dexmachina.envs.demo_data import _infer_hand_sides_from_world_coord

NUM_JOINTS = 21  # MANO hand joints per hand
# Match parallel_retarget cardboard_box: pos=(0,-0.08,0.90), size=(0.2,0.2,0.1), top at 0.95m
TABLE_TOP_Z = 0.95
TABLE_BOX_POS = (0, -0.08, 0.90)
TABLE_BOX_SIZE = (0.2, 0.2, 0.1)
JOINT_SPHERE_RADIUS = 0.008
LEFT_COLOR = (0.2, 0.5, 1.0)
RIGHT_COLOR = (1.0, 0.4, 0.2)


def load_processed_npy(data_source, npy_path=None, subject=None, sequence_id=None, obj=None, use_clip=None, debug_pose=False):
    """Load processed .npy; return raw dict, hand_sides, object config hint, and camera hint."""
    if npy_path and os.path.isfile(npy_path):
        path = npy_path
    elif data_source == "dexycb" and subject and sequence_id:
        base = get_asset_path("dexycb/processed")
        path = os.path.join(base, subject, f"{sequence_id}.npy")
    elif data_source == "arctic" and subject and obj and use_clip:
        base = get_asset_path("arctic/processed")
        path = os.path.join(base, subject, f"{obj}_use_{use_clip}.npy")
    else:
        raise FileNotFoundError(
            "Provide --npy path or (--data_source dexycb --subject X --sequence_id Y) or "
            "(--data_source arctic --subject X --obj Y --use_clip Z)"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=True).item()
    if debug_pose and "ycb_class_name" in raw.get("params", {}):
        params = raw["params"]
        idx = min(1, params["obj_trans"].shape[0] - 1) if params["obj_trans"].shape[0] > 0 else 0
        print(f"[DEBUG] frame {idx}: obj_trans={params['obj_trans'][idx]}, obj_quat={params['obj_quat'][idx]}")
    params = raw["params"]
    world_coord = raw["world_coord"]
    hand_sides = _infer_hand_sides_from_world_coord(world_coord)
    ycb_class = params.get("ycb_class_name")
    obj_name = str(ycb_class) if ycb_class else (obj or "box")
    return raw, hand_sides, obj_name, path


def create_scene(object_cfg, demo_data, hand_sides, device, cam_pos, cam_lookat, table_height=None):
    """Scene with plane, object, and joint spheres (no robot).
    table_height: if set, add a Box table with top surface at this z (for pickup/place)."""
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
            gravity=(0, 0, -9.81),
        ),
        vis_options=gs.options.VisOptions(
            n_rendered_envs=1,
            show_world_frame=False,
            visualize_contact=False,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=1 / 60,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=False,
            enable_joint_limit=True,
        ),
        use_visualizer=True,
        show_viewer=False,
        show_FPS=False,
    )
    scene = gs.Scene(**scene_cfg)
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    if table_height is not None:
        # Same cardboard_box as parallel_retarget (ARCTIC table)
        scene.add_entity(
            gs.morphs.Box(
                pos=TABLE_BOX_POS,
                size=TABLE_BOX_SIZE,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Smooth(roughness=0.1),
        )

    obj_cfg = {**object_cfg, "fixed": True, "collect_data": False}
    obj = ArticulatedObject(
        obj_cfg,
        device=device,
        scene=scene,
        num_envs=1,
        demo_data=demo_data,
        visualize_contact=False,
    )

    markers = []
    for side in hand_sides:
        color = LEFT_COLOR if side == "left" else RIGHT_COLOR
        for _ in range(NUM_JOINTS):
            m = scene.add_entity(
                gs.morphs.Sphere(
                    radius=JOINT_SPHERE_RADIUS,
                    pos=(0, 0, 0),
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Smooth(roughness=0.3, color=(*color, 1.0)),
            )
            markers.append((side, m))

    camera = scene.add_camera(
        pos=cam_pos,
        lookat=cam_lookat,
        res=(1280, 720),
        fov=65,
        GUI=False,
    )
    scene.build(n_envs=1, env_spacing=(1.5, 1.5), n_envs_per_row=8)
    obj.post_scene_build_setup()
    return scene, obj, markers, camera


def main(args):
    raw, hand_sides, obj_name, _ = load_processed_npy(
        data_source=args.data_source,
        npy_path=args.npy,
        subject=args.subject,
        sequence_id=args.sequence_id,
        obj=args.obj,
        use_clip=args.use_clip,
        debug_pose=args.debug_pose,
    )
    world_coord = raw["world_coord"]
    params = raw["params"]
    T = params["obj_trans"].shape[0]

    obj_trans = params["obj_trans"]
    obj_quat = params["obj_quat"]
    obj_arti = params["obj_arti"]
    if obj_arti.ndim == 1:
        obj_arti = obj_arti[:, None]

    demo_data = {
        "obj_pos": obj_trans,
        "obj_quat": obj_quat,
        "obj_arti": obj_arti,
    }
    if args.data_source == "dexycb" and "ycb_class_name" in params:
        demo_data["ycb_class_name"] = str(params["ycb_class_name"])

    if "ycb_class_name" in params:
        object_cfg = get_ycb_object_cfg(str(params["ycb_class_name"]), voc_7dof=False)  # fixed for viz
    else:
        object_cfg = get_arctic_object_cfg(name=obj_name, convexify=False)

    if args.num_frames > 0:
        step_indices = np.linspace(0, T - 1, min(args.num_frames, T), dtype=int)
    else:
        step_indices = range(0, T, args.frame_skip)
    step_indices = list(step_indices)
    print(f"Unretargeted visualization: {args.data_source}, hands={hand_sides}, frames={len(step_indices)}")

    cam_pos, cam_lookat = [-1, -1, 1.3], [0, -0.08, 0.95]
    print(f"Camera pos={[round(v,3) for v in cam_pos]}, lookat={[round(v,3) for v in cam_lookat]}")

    gs.init(backend=gs.gpu)
    device = torch.device("cuda")
    table_height = TABLE_TOP_Z  # Cardboard table for pickup/place (same as parallel_retarget)
    scene, obj, markers, camera = create_scene(
        object_cfg, demo_data, hand_sides, device, cam_pos, cam_lookat, table_height=table_height
    )

    obj_pos = torch.tensor(obj_trans, device=device)
    obj_quat_t = torch.tensor(obj_quat, device=device)
    obj_arti_t = torch.tensor(obj_arti, device=device)

    render_frames = []
    import cv2

    for frame_idx, step_idx in enumerate(tqdm(step_indices, desc="Rendering")):
        obj.set_object_state(
            root_pos=obj_pos[step_idx : step_idx + 1],
            root_quat=obj_quat_t[step_idx : step_idx + 1],
            joint_qpos=obj_arti_t[step_idx : step_idx + 1],
            env_idxs=[0],
        )
        marker_idx = 0
        for side in hand_sides:
            joints = world_coord[f"joints.{side}"][step_idx]
            for j in range(min(NUM_JOINTS, joints.shape[0])):
                if marker_idx < len(markers):
                    _, m = markers[marker_idx]
                    m.set_pos(joints[j : j + 1].astype(np.float32))
                marker_idx += 1
        while marker_idx < len(markers):
            _, m = markers[marker_idx]
            m.set_pos(np.zeros((1, 3), dtype=np.float32))
            marker_idx += 1

        scene.step()
        img = camera.render()[0]
        render_frames.append(img)
        if args.save_frames:
            os.makedirs(args.output_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

    if args.create_video and render_frames:
        out_path = os.path.join(args.output_dir, args.output_name)
        os.makedirs(args.output_dir, exist_ok=True)
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(render_frames, fps=args.fps)
        clip.write_videofile(out_path)
        print(f"Video: {out_path} ({len(render_frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize unretargeted (human hand) data from processed .npy")
    parser.add_argument("--data_source", choices=["dexycb", "arctic"], default="dexycb")
    parser.add_argument("--npy", type=str, default=None, help="Path to processed .npy (overrides subject/sequence/obj)")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--sequence_id", type=str, default=None, help="DexYCB: e.g. 20200709_141754_vector_para or sequence id")
    parser.add_argument("--obj", type=str, default="box", help="ARCTIC object name")
    parser.add_argument("--use_clip", type=str, default="01")
    parser.add_argument("--num_frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="visualization_output")
    parser.add_argument("--output_name", type=str, default="unretargeted.mp4")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--create_video", action="store_true", default=True)
    parser.add_argument("--debug_pose", action="store_true", help="Print object pose for debugging")
    args = parser.parse_args()
    main(args)
