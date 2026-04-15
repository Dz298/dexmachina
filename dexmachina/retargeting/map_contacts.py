import os 
import cv2
import yaml
import numpy as np
import pickle

from os.path import join
import genesis as gs
import torch
 
import argparse
import xml.etree.ElementTree as ET
from sklearn.neighbors import KDTree 
from copy import deepcopy
from collections import defaultdict
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg, get_ycb_object_cfg
from dexmachina.envs.math_utils import matrix_from_quat
from dexmachina.envs.demo_data import _infer_hand_sides_from_world_coord

from dexmachina.asset_utils import get_asset_path
from dexmachina.retargeting.coordinate_utils import apply_dexycb_display_to_processed_and_retargeter
"""

python retargeting/map_contacts.py --hand allegro_hand --show_object  --num_markers 100  --record_video # --raytrace

Go from mesh vertice contacts (on object surface) to robot hand links
- use "contacts.left", "valid_contacts.left": shape (T, N=50, 4)
- use raw retargeted hand poses, no object no collision, and use AABB to approximate center positions for all collision links
- for each raw contact pos: find the closest link center pos
- for each link: find all the matched contact points -> average them?

Render only grouped contacts with raytracing
python retargeting/map_contacts.py --hand ${HAND} --record_video  --show_object  --num_markers 15 --load_fname assets/arctic/processed/${job} --render_only --raytrace --show_grouped_contact_only
"""

def show_contact_plt(contact_links):
    # contact_links: shape (T, num_mano_links, 4) assign each link a color and show all frames
    import matplotlib
    matplotlib.use('tkAgg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    # set fixed xyz limits
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0.8, 1.4)
    scatter = None
    num_links = contact_links.shape[1]
    colors = plt.cm.jet(np.linspace(0, 1, num_links))

    t = 0
    max_t = contact_links.shape[0]
    while True:
        points = contact_links[t, :, :3]
        sizes = 50.0 * np.ones(points.shape[0])
        if scatter is None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes)
        else:
            scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            scatter.set_color(colors)
            scatter.set_sizes(sizes)
        plt.pause(0.05)
        t += 1
        if t == max_t:
            t = 0

def render_transparent_img(cam):
    img, _, seg_arr, _ = cam.render(segmentation=True)
    rgb_img = img.copy() # do this BEFOFE
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # do this first!! do channel after
    max_id = seg_arr.max() # assuming this is background 
    channel = np.ones(img.shape[0:2], dtype=np.uint8) * 255
    channel[seg_arr == max_id] = 0
    # make a 3 channel rgb image but the background is all white
    
    rgb_img[seg_arr == max_id] = 255
    img = np.concatenate([img, channel[:, :, None]], axis=-1)
    return img, rgb_img

def _get_object_part_names(object_cfg):
    if object_cfg is not None and object_cfg.get("object_type") == "ycb":
        return ["object"]
    return ["top", "bottom"]


def create_scene(args, object_name, urdfs, num_raw_contact_markers=50, num_grouped_contact_markers=50,
                object_cfg=None, object_cls=None):
    import genesis as gs
    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0,0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            enable_joint_limit=True,
        ),
        show_viewer=args.vis_scene,
        use_visualizer=(args.vis_scene or args.record_video),
        show_FPS=False,
        vis_options = gs.options.VisOptions( 
            plane_reflection = True,
            ambient_light    = (0.4, 0.4, 0.4),
            lights = [
                {"type": "directional", "dir": (0, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
            ]
        ),
        viewer_options=gs.options.ViewerOptions( 
            camera_pos=(1.5, 0.8, 2.1),
            camera_lookat=(0.0, -0.1, 1.1),
            camera_fov=25,
        ),
    )
    plane_urdf = 'urdf/plane/plane.urdf' # NOTE this is Genesis default plane
    if args.raytrace:
        scene_cfg['renderer'] = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=10.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 1.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
        plane_urdf = join(get_asset_path('plane'), 'plane_custom.urdf') # use custom plane with texture
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')
    
    cam = None 
    if args.record_video:
        if args.raytrace:
            cam = scene.add_camera(
            pos=scene_cfg['viewer_options'].camera_pos, lookat=scene_cfg['viewer_options'].camera_lookat,
            res=(1024, 1024), fov=20, GUI=False) 
        else:
            cam = scene.add_camera(
            pos=scene_cfg['viewer_options'].camera_pos, lookat=scene_cfg['viewer_options'].camera_lookat,
            res=(512, 512), fov=20, GUI=False)
    
    hand_entities = dict()
    for side, urdf_path in urdfs.items():
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path, 
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                collision=True, # has to be true for get_AABB to work
                # collision=False, 
            ),
            material=gs.materials.Rigid(
                gravity_compensation=0.8
                ),
            # surface=gs.surfaces.Smooth(color=(0, 0, 0.8, 0.5)),            
        )
        hand_entities[side] = hand 
    obj = None
    if args.show_object and object_cfg is not None and object_cls is not None:
        obj_cfg = object_cfg.copy()
        obj_cfg["fixed"] = False
        obj_cfg["disable_collision"] = True
        obj_cfg["color"] = (1.0, 0.423, 0.039, 0.3)
        obj = object_cls(obj_cfg, device=device, scene=scene, num_envs=1)
    markers = dict()
    if args.show_grouped_contact_only:
        num_raw_contact_markers = 0
    # YCB/DexYCB has one rigid surface part ("object"); ARCTIC has two articulated parts.
    mesh_parts = _get_object_part_names(object_cfg)
    for palette, marker_type, num_markers in zip(['rocket', 'crest'],['raw', 'grouped'], [num_raw_contact_markers, num_grouped_contact_markers]):
        if num_markers > 0:
            import seaborn as sns
            marker_colors = sns.color_palette(palette, max(2, len(mesh_parts)))
            marker_colors = np.array(marker_colors)
            for i, part in enumerate(mesh_parts):
                color = marker_colors[i]
                
                marker_ents = [
                    scene.add_entity(
                        gs.morphs.Sphere(
                            radius=0.008 if marker_type == 'raw' else 0.015, 
                            fixed=False, 
                            collision=False), 
                            surface=gs.surfaces.Smooth(color=color),
                            ) for _ in range(num_markers)
                ]
                markers[f"{marker_type}_{part}"] = marker_ents
    # add this last for segmentation to work
    ground = scene.add_entity(gs.morphs.URDF(file=plane_urdf, fixed=True))
    scene.build(n_envs=1, env_spacing=(2.0, 2.0)) 
    return scene, hand_entities, markers, obj, cam, mesh_parts

def show_hand_joints_links_plt(hand_entites): 
    import matplotlib
    matplotlib.use('tkAgg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # plot two subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # set fixed xyz limits
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
    scatter = None

    for side, hand in hand_entities.items():
        link_pos = np.array([link.pos for link in hand.links])
        link_names = [link.name for link in hand.links]
        joint_pos = np.array([joint.pos for joint in hand.joints if 'forearm' not in joint.name])
        joint_names = [joint.name for joint in hand.joints if 'forearm' not in joint.name]
        ax = ax1 if side == 'left' else ax2 
        link_colors = plt.cm.summer(np.linspace(0, 1, len(link_pos)))
        joint_colors = plt.cm.spring(np.linspace(0, 1, len(joint_pos)))
        sizes = 100.0 * np.ones(len(link_pos))
        joint_sizes = 200.0 * np.ones(len(joint_pos))
        positions = np.concatenate([link_pos, joint_pos], axis=0)
        colors = np.concatenate([link_colors, joint_colors], axis=0)
        sizes = np.concatenate([sizes, joint_sizes], axis=0)
        labels = link_names + joint_names
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=sizes)
        # use annotate 
        for i, label in enumerate(labels):
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label, size=8, zorder=1)
    plt.show()
    
def show_hand_kpts_scene(scene, hand_entites, markers):
    marker_offset = 0
    for side, hand in hand_entities.items():
        # link_pos = np.array([link.pos for link in hand.links])
        link_pos = np.array([link.inertial_pos + link.pos for link in hand.links if link.geoms])
        aabbs = [link.get_AABB()[0] for link in hand.links if link.geoms] #each is (2, 3)
        aabb_centers = [ 0.5 * (aabb[0] + aabb[1]) for aabb in aabbs]
        # link_names = [link.name for link in hand.links]
        joint_pos = np.array([joint.pos for joint in hand.joints if 'forearm' not in joint.name])
        joint_names = [joint.name for joint in hand.joints if 'forearm' not in joint.name]

        # positions = np.concatenate([link_pos, joint_pos], axis=0)
        positions = aabb_centers
        for i, pos in enumerate(positions):
            idx = marker_offset + i
            if idx >= len(markers):
                break
            markers[idx].set_pos(pos[None])
        marker_offset += len(positions)
    scene.step()
    breakpoint()

def group_contacts(links, raw_contacts, valids, num_obj_parts=2):
    """
    Grouping per-step contacts, input:
    - links: raw contacts shape (N=50, 4) 
    NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top' 
    returns:
    - grouped_contacts: shape (num_obj_parts=2, num_dex_links, 4)
    - grouped_valids: shape (num_obj_parts=2, num_dex_links) 
    -> note here that to be consistent with environment contact readings, need a separate set of contacts for each obj part 
    - target_positions: shape (N=50, 3) target positions for each raw contacts after grouping 
    """
    aabbs = [link.get_AABB()[0].cpu().numpy() for link in links] # each is (2, 3)
    link_center_pos = np.array([0.5 * (aabb[0] + aabb[1]) for aabb in aabbs])
    # now for each raw_contact, find the closest link center pos
    kdtree = KDTree(link_center_pos)
    positions = raw_contacts[:, :3]
    distances, indices = kdtree.query(positions, k=1)
    # for the invalid raw contacts, set the index to -1
    indices[~valids] = -1
    nlinks = len(links)
    # now for each link, find all the matched contact points
    grouped_contacts = np.zeros((num_obj_parts, nlinks, 4))
    grouped_valids = np.zeros((num_obj_parts, nlinks,))
    target_positions = np.zeros(positions.shape)
    for i in range(len(links)):
        for j in range(num_obj_parts):
            pidx = j + 1 # valid index should be 1 or 2, NOT 0
            part_match = raw_contacts[:, 3] == pidx # (50,)
            has_valid = (indices == i).flatten() # (50,1)->(50,)
            part_valid = part_match & has_valid
            if np.sum(part_valid) > 0: 
                mean_pos = np.mean(positions[part_valid], axis=0)
                grouped_contacts[j, i, :3] = mean_pos
                # part_ids = raw_contacts[part_valid, 3]
                # voted_part_id = np.argmax(np.bincount(part_ids.astype(int))) 
                grouped_contacts[j, i, 3] = pidx # no voting needed 
                grouped_valids[j, i] = 1 
                target_positions[has_valid] = mean_pos
    return grouped_contacts, grouped_valids, target_positions


def _quat_to_matrix_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = torch.tensor(quat_wxyz, dtype=torch.float32).unsqueeze(0)
    return matrix_from_quat(quat).squeeze(0).cpu().numpy()


def _axis_angle_to_matrix_np(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8 or abs(angle) < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / axis_norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float32)


def _rpy_to_matrix_np(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


class ArcticObjectMeshHelper:
    # part_id 1 -> "top", part_id 2 -> "bottom"
    part_ids = [1, 2]

    def __init__(self, object_name: str):
        import trimesh

        self.cfg = get_arctic_object_cfg(object_name)
        self.meshes = {
            "bottom": trimesh.load(self.cfg["bottom_mesh_fname"], force="mesh", process=False),
            "top": trimesh.load(self.cfg["top_mesh_fname"], force="mesh", process=False),
        }
        self.joint_origin_xyz = np.zeros(3, dtype=np.float32)
        self.joint_origin_rpy = np.zeros(3, dtype=np.float32)
        self.joint_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._load_joint_from_urdf(self.cfg["urdf_path"])

    def part_id_to_name(self, part_id: int) -> str:
        if int(part_id) == 1:
            return "top"
        if int(part_id) == 2:
            return "bottom"
        raise ValueError(f"Invalid ARCTIC part id: {part_id}")

    def _load_joint_from_urdf(self, urdf_path: str):
        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            if parent.attrib.get("link") != "bottom" or child.attrib.get("link") != "top":
                continue
            origin = joint.find("origin")
            if origin is not None:
                if "xyz" in origin.attrib:
                    self.joint_origin_xyz = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=np.float32)
                if "rpy" in origin.attrib:
                    self.joint_origin_rpy = np.fromstring(origin.attrib["rpy"], sep=" ", dtype=np.float32)
            axis = joint.find("axis")
            if axis is not None and "xyz" in axis.attrib:
                self.joint_axis = np.fromstring(axis.attrib["xyz"], sep=" ", dtype=np.float32)
            break

    def get_part_pose(self, part: str, root_pos: np.ndarray, root_quat: np.ndarray, joint_qpos: float):
        root_pos = np.asarray(root_pos, dtype=np.float32)
        root_rot = _quat_to_matrix_np(np.asarray(root_quat, dtype=np.float32))
        if part == "bottom":
            return root_pos, root_rot
        joint_rot = _axis_angle_to_matrix_np(self.joint_axis, float(joint_qpos))
        origin_rot = _rpy_to_matrix_np(self.joint_origin_rpy)
        part_rot = root_rot @ origin_rot @ joint_rot
        part_pos = root_pos + root_rot @ self.joint_origin_xyz
        return part_pos, part_rot

    def query_part_surface_world(self, part: str, points_world: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, joint_qpos: float):
        mesh = self.meshes[part]
        part_pos, part_rot = self.get_part_pose(part, root_pos, root_quat, joint_qpos)
        points_world = np.asarray(points_world, dtype=np.float32)
        points_local = (points_world - part_pos[None]) @ part_rot
        closest_local, _, tri_ids = mesh.nearest.on_surface(points_local)
        normals_local = mesh.face_normals[tri_ids]
        normals_local = normals_local / np.clip(np.linalg.norm(normals_local, axis=-1, keepdims=True), 1e-8, None)
        return closest_local.astype(np.float32), normals_local.astype(np.float32)


class YCBObjectMeshHelper:
    # YCB is a rigid single-part object; process_dexycb stamps all contacts with part_id=1.
    part_ids = [1]

    def __init__(self, object_cfg: dict):
        import trimesh

        mesh_fname = object_cfg["mesh_fname"]
        assert os.path.exists(mesh_fname), f"YCB mesh not found: {mesh_fname}"
        self.meshes = {
            "object": trimesh.load(mesh_fname, force="mesh", process=False),
        }

    def part_id_to_name(self, part_id: int) -> str:
        return "object"

    def get_part_pose(self, part: str, root_pos: np.ndarray, root_quat: np.ndarray, joint_qpos: float):
        root_pos = np.asarray(root_pos, dtype=np.float32)
        root_rot = _quat_to_matrix_np(np.asarray(root_quat, dtype=np.float32))
        return root_pos, root_rot

    def query_part_surface_world(self, part: str, points_world: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, joint_qpos: float = 0):
        mesh = self.meshes["object"]
        part_pos, part_rot = self.get_part_pose(part, root_pos, root_quat, joint_qpos)
        points_world = np.asarray(points_world, dtype=np.float32)
        points_local = (points_world - part_pos[None]) @ part_rot
        closest_local, _, tri_ids = mesh.nearest.on_surface(points_local)
        normals_local = mesh.face_normals[tri_ids]
        normals_local = normals_local / np.clip(np.linalg.norm(normals_local, axis=-1, keepdims=True), 1e-8, None)
        return closest_local.astype(np.float32), normals_local.astype(np.float32)


def compute_local_contact_targets(raw_contacts, valids, mesh_helper, obj_state):
    local_positions = np.zeros((raw_contacts.shape[0], 3), dtype=np.float32)
    local_normals = np.zeros((raw_contacts.shape[0], 3), dtype=np.float32)
    if not np.any(valids):
        return local_positions, local_normals

    for part_id in mesh_helper.part_ids:
        mask = valids & (raw_contacts[:, 3] == part_id)
        if not np.any(mask):
            continue
        part_name = mesh_helper.part_id_to_name(part_id)
        closest_local, normals_local = mesh_helper.query_part_surface_world(
            part_name,
            raw_contacts[mask, :3],
            root_pos=obj_state["root_pos"],
            root_quat=obj_state["root_quat"],
            joint_qpos=obj_state["joint_qpos"],
        )
        local_positions[mask] = closest_local
        local_normals[mask] = normals_local
    return local_positions, local_normals


def group_local_contact_targets(links, raw_contacts, valids, local_positions, local_normals, num_obj_parts=2):
    aabbs = [link.get_AABB()[0].cpu().numpy() for link in links]
    link_center_pos = np.array([0.5 * (aabb[0] + aabb[1]) for aabb in aabbs])
    kdtree = KDTree(link_center_pos)
    _, indices = kdtree.query(raw_contacts[:, :3], k=1)
    indices[~valids] = -1

    nlinks = len(links)
    grouped_local_positions = np.zeros((num_obj_parts, nlinks, 3), dtype=np.float32)
    grouped_local_normals = np.zeros((num_obj_parts, nlinks, 3), dtype=np.float32)
    for i in range(nlinks):
        for j in range(num_obj_parts):
            part_id = j + 1
            matched = (indices == i).flatten() & (raw_contacts[:, 3] == part_id) & valids
            if np.sum(matched) == 0:
                continue
            grouped_local_positions[j, i] = np.mean(local_positions[matched], axis=0)
            mean_normal = np.mean(local_normals[matched], axis=0)
            norm = np.linalg.norm(mean_normal)
            if norm > 1e-8:
                grouped_local_normals[j, i] = mean_normal / norm
    return grouped_local_positions, grouped_local_normals

def set_entities_to_step(hand_entities, retargeter_results, step):
    for side, hand in hand_entities.items():
        hand_qpos = retargeter_results[side]["hand_qpos"][step]
        hand_qpos = torch.tensor(hand_qpos).to(device)[None]
        joint_idxs = [joint.dof_idx_local for joint in hand.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        hand.set_dofs_position(position=hand_qpos, dofs_idx_local=joint_idxs)
    return 

def set_object_to_step(obj, obj_states, step):
    obj.set_object_state(
        root_pos=obj_states["root_pos"][step][None],
        root_quat=obj_states["root_quat"][step][None],
        joint_qpos=obj_states["joint_qpos"][step][None],
    )
    return

def visualize_markers(markers_dict, raw_contacts, grouped_contacts, part_names):
    # first set all markers to 0! avoid delays in vis
    for k, markers in markers_dict.items():
        for marker in markers:
            marker.set_pos(np.zeros((1, 3))) 
    for i, part in enumerate(part_names):
        pid = i + 1
        raw_markers = markers_dict.get(f"raw_{part}", [])
        part_mask = raw_contacts[:, 3] == pid
        raw_pos = raw_contacts[part_mask, :3]
        for k, pos in enumerate(raw_pos):
            if k >= len(raw_markers):
                break
            raw_markers[k].set_pos(pos[None])
        grouped_markers = markers_dict.get(f"grouped_{part}", [])
        grouped_pos = []
        for side, contacts in grouped_contacts.items(): 
            # contacts are shaped (num_obj_parts, num_links, 4)
            grouped_pos.append(contacts[i, :, :3])
        grouped_pos = np.concatenate(grouped_pos, axis=0)
        for k, pos in enumerate(grouped_pos):
            if k >= len(grouped_markers):
                break
            grouped_markers[k].set_pos(pos[None])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='xhand')
    parser.add_argument('--load_fname', '-lf', type=str, default='/home/mandi/chiral/assets/arctic/processed/s01/box_use_01.npy')
    # parser.add_argument('--retar_fname', type=str, default='para') -> look up this automatically
    parser.add_argument('--save_dir', type=str, default='contact_retarget')
    parser.add_argument('--show_mano_plt', action='store_true', help='Whether to show the matplotlib plot')
    parser.add_argument('--show_hand_links', action='store_true', help='Whether to show the hand links')
    parser.add_argument('--vis_scene', '-v', action='store_true', help='Whether to visualize the scene')
    parser.add_argument('--show_object', action='store_true', help='Whether to show the object')
    parser.add_argument('--num_markers', type=int, default=0, help='Number of markers to show')
    parser.add_argument('--record_video', action='store_true', help='Whether to record video')
    parser.add_argument('--raytrace', action='store_true', help='Whether to use raytracer')
    parser.add_argument('--render_only', action='store_true', help='if true, skip saving data')
    parser.add_argument('--show_grouped_contact_only', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.load_fname), f"load_fname={args.load_fname} does not exist"
    subject_name = args.load_fname.split("/")[-2]
    hand_name = args.hand if 'hand' in args.hand else f"{args.hand}_hand"
    retarget_type = 'position' if hand_name == 'shadow_hand' else 'vector'
    retarget_fname = str(get_asset_path(join("retargeter_results", hand_name, subject_name, args.load_fname.split("/")[-1].replace(".npy", f"_{retarget_type}.npy"))))
    assert os.path.exists(retarget_fname), f"retarget_fname={retarget_fname} does not exist"

    retargeter_results = np.load(retarget_fname, allow_pickle=True).item()
    loaded_data = np.load(args.load_fname, allow_pickle=True).item()
    apply_dexycb_display_to_processed_and_retargeter(loaded_data, retargeter_results)
    if "ycb_class_name" in loaded_data.get("params", {}):
        object_name = str(loaded_data["params"]["ycb_class_name"])
        object_cfg = get_ycb_object_cfg(object_name, voc_7dof=False)  # fixed base for contact mapping
        object_cls = ArticulatedObject
    else:
        object_name = args.load_fname.split("/")[-1].split("_")[0]
        object_cfg = get_arctic_object_cfg(object_name)
        object_cls = ArticulatedObject

    full_save_dir = get_asset_path(args.save_dir)
    save_path = os.path.join(full_save_dir, hand_name, subject_name)
    os.makedirs(save_path, exist_ok=True)
    if args.record_video:
        frame_path = os.path.join(save_path, f"frames_{object_name}")
        os.makedirs(frame_path, exist_ok=True)
    save_fname = os.path.join(save_path, args.load_fname.split("/")[-1])
    world_coord = loaded_data["world_coord"]
    hand_sides = _infer_hand_sides_from_world_coord(world_coord)

    obj_states = {
        "root_pos": loaded_data["params"]["obj_trans"],
        "root_quat": loaded_data["params"]["obj_quat"],
        "joint_qpos": loaded_data["params"]["obj_arti"],
    }
    if object_cfg is not None and object_cfg.get("object_type") == "ycb":
        mesh_helper = YCBObjectMeshHelper(object_cfg)
    else:
        mesh_helper = ArcticObjectMeshHelper(object_name)

    if args.show_mano_plt:
        contact_links = [world_coord[f"contact_links_{s}"] for s in hand_sides]
        show_contact_plt(np.concatenate(contact_links, axis=1))
        breakpoint()

    urdfs = dict()
    robot_dir = get_asset_path(args.hand)
    config_path = join(robot_dir, "retarget_config.yaml")
    config = yaml.safe_load(open(config_path, "r"))
    for side in hand_sides:
        urdf_path = config[side]["urdf_path"]
        urdfs[side] = join(robot_dir, urdf_path)  
    
    # use genesis to create hand entities
    scene, hand_entities, markers, obj, cam, obj_part_names = create_scene(
        args, object_name, urdfs,
        num_raw_contact_markers=args.num_markers, num_grouped_contact_markers=args.num_markers,
        object_cfg=object_cfg, object_cls=object_cls,
    )
    device = torch.device('cuda:0')
    if args.show_hand_links: 
        # show the hand links and joints
        # show_hand_joints_links_plt(hand_entities)
        show_hand_kpts_scene(scene, hand_entities, markers)
    
    collision_links = dict()
    for side, hand in hand_entities.items():
        links = [link for link in hand.links if link.geoms]
        collision_links[side] = links
    
    num_steps = retargeter_results[hand_sides[0]]["hand_qpos"].shape[0]
    step = 0
    frames = []
    saved_vid = False
    clip_name = args.load_fname.split("/")[-1].replace(".npy", "")
    tosave = {side: defaultdict(list) for side in hand_sides}
    num_obj_parts = len(obj_part_names)
    saved_contacts = False
    while True:
        set_entities_to_step(hand_entities, retargeter_results, step) 
        if obj is not None:
            set_object_to_step(obj, obj_states, step)
        
        # raw_contacts = [loaded_data['world_coord'][f"contacts.{side}"][step] for side in hand_entities.keys()]
        # raw_positions = np.concatenate(raw_contacts, axis=0)[:, :3]
        
        # scene.step() 
        grouped_contacts = dict()
        grouped_valids = dict()
        grouped_contacts_local = dict()
        grouped_normals_local = dict()
        target_pos = dict()
        for side, hand in hand_entities.items():
            raw_contacts = loaded_data['world_coord'][f"contacts.{side}"]
            valid_contacts = loaded_data['world_coord'][f"valid_contacts.{side}"]
            local_positions, local_normals = compute_local_contact_targets(
                raw_contacts[step],
                valid_contacts[step],
                mesh_helper,
                dict(
                    root_pos=obj_states["root_pos"][step],
                    root_quat=obj_states["root_quat"][step],
                    joint_qpos=float(obj_states["joint_qpos"][step]),
                ),
            )
            hand_link_contacts, hand_link_valids, target_positions = group_contacts(
                collision_links[side], raw_contacts[step], valid_contacts[step], num_obj_parts=num_obj_parts
            )
            hand_link_contacts_local, hand_link_normals_local = group_local_contact_targets(
                collision_links[side], raw_contacts[step], valid_contacts[step], local_positions, local_normals
            )
            grouped_contacts[side] = hand_link_contacts # shape (num_obj_parts, num_dex_links, 4)
            grouped_valids[side] = hand_link_valids
            grouped_contacts_local[side] = hand_link_contacts_local
            grouped_normals_local[side] = hand_link_normals_local
            tosave[side]['dexlink_contacts'].append(hand_link_contacts)
            tosave[side]['dexlink_valid_contacts'].append(hand_link_valids)
            tosave[side]['dexlink_contacts_local'].append(hand_link_contacts_local)
            tosave[side]['dexlink_contact_normals_local'].append(hand_link_normals_local)
            target_pos[side] = target_positions

        if args.num_markers > 0: 
            all_raw_contacts = np.concatenate([loaded_data['world_coord'][f"contacts.{side}"][step] for side in hand_entities.keys()], axis=0)
            visualize_markers(markers, all_raw_contacts, grouped_contacts, obj_part_names)
            
            # set_entities_to_step(hand_entities, retargeter_results, step)
            # set_object_to_step(obj, obj_states, step)
        scene.step()
        # Print frame info with articulation angle
        obj_arti = obj_states["joint_qpos"][step] if args.show_object else 0
        print(f"Frame {step:4d}/{num_steps} | Lid angle: {obj_arti:.3f} rad ({np.degrees(obj_arti):.1f}°)", end='\r')
        
        if args.record_video and not saved_vid:
            if args.raytrace:
                # render segmentation 
                transparent_img, white_bg_img = render_transparent_img(cam)
                img_fname = os.path.join(frame_path, f"{clip_name}_{step:04d}.png")
                cv2.imwrite(img_fname, transparent_img)
                frames.append(white_bg_img)
            else:
                img, _, _, _ = cam.render()
                frames.append(img) 
            # os.makedirs(os.path.join(save_path, clip_name), exist_ok=True)
            # fname = os.path.join(save_path, clip_name, f"{step:04d}.png")
            # import cv2
            # cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
             
        step += 1
        if step >= num_steps:
            step = 0
            if not saved_contacts:
                for side in hand_sides:
                    data = tosave[side]
                    for key, val in data.items():
                        val = np.stack(val, axis=0)
                        tosave[side][key] = val
                        print(f"Key={key}, val.shape={val.shape}")
                for side, links in collision_links.items():
                    link_names = [link.name for link in links]
                    link_local_idxs = [link.idx_local for link in links]
                    tosave[side]['collision_link_names'] = link_names
                    tosave[side]['collision_link_local_idxs'] = link_local_idxs
                    tosave[side]['object_part_names'] = obj_part_names
                
                if not args.render_only:
                    np.save(save_fname, tosave)
                    print(f"Saved contacts to {save_fname}")
                if not args.record_video:
                    break
                
                
            if args.record_video and not saved_vid:
                video_fname = os.path.join(save_path, f"{args.load_fname.split('/')[-1].replace('.npy', '.mp4')}")
                from moviepy.editor import ImageSequenceClip
                clip = ImageSequenceClip(frames, fps=15)
                clip.write_videofile(video_fname)
                print(f"Saved video to {video_fname}")
                saved_vid = True
                break
    exit()
