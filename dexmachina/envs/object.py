import os 
import cv2
import torch
import numpy as np
import genesis as gs
from os.path import join
from collections import defaultdict

from dexmachina.envs.math_utils import matrix_from_quat
from dexmachina.envs.virtual_force import points_world_to_local_np
from dexmachina.asset_utils import get_asset_path

# YCB class names (DexYCB); mesh at {dex_ycb_dir}/models/{name}/textured_simple.obj
YCB_CLASS_NAMES = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser",
    "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors",
    "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick",
]


def _get_ycb_1dof_urdf_path(mesh_dir):
    """Ensure a 1-DOF (dummy revolute) URDF exists for ArticulatedObject and return its path."""
    urdf_path = join(mesh_dir, "object_1dof.urdf")
    if not os.path.exists(urdf_path):
        # One revolute joint with limit 0-0 so the virtual controller has 1 DOF (no motion in dataset).
        urdf_content = '''<?xml version="1.0"?>
<robot name="ycb_object">
  <link name="base"/>
  <link name="object">
    <visual>
      <geometry>
        <mesh filename="textured_simple.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="textured_simple.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="dummy_joint" type="revolute">
    <parent link="base"/>
    <child link="object"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0" effort="0" velocity="0"/>
  </joint>
</robot>
'''
        with open(urdf_path, "w") as f:
            f.write(urdf_content)
    return urdf_path


def _get_ycb_7dof_urdf_path(mesh_dir):
    """Ensure a 7-DOF URDF exists: 6 DOF free-floating base (position + rotation) + 1 dummy revolute (0-0)."""
    urdf_path = join(mesh_dir, "object_7dof.urdf")
    if not os.path.exists(urdf_path):
        urdf_content = '''<?xml version="1.0"?>
<robot name="ycb_object_7dof">
  <link name="world"/>
  <link name="base"/>
  <link name="object">
    <visual>
      <geometry>
        <mesh filename="textured_simple.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="textured_simple.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="free_joint" type="floating">
    <parent link="world"/>
    <child link="base"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="dummy_joint" type="revolute">
    <parent link="base"/>
    <child link="object"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0" effort="0" velocity="0"/>
  </joint>
</robot>
'''
        with open(urdf_path, "w") as f:
            f.write(urdf_content)
    return urdf_path


def get_ycb_object_cfg(ycb_class_name, dex_ycb_dir=None, voc_7dof=True):
    """Config for YCB object (DexYCB). voc_7dof=True (default): 7 DOF (6 pose + 1 dummy joint) for training/VOC."""
    if dex_ycb_dir is None:
        dex_ycb_dir = os.environ.get("DEX_YCB_DIR")
    if dex_ycb_dir and os.path.isdir(dex_ycb_dir):
        mesh_dir = join(dex_ycb_dir, "models", ycb_class_name)
    else:
        mesh_dir = str(get_asset_path(f"dex_ycb/models/{ycb_class_name}"))
    mesh_fname = join(mesh_dir, "textured_simple.obj")
    assert os.path.exists(mesh_fname), f"YCB mesh not found: {mesh_fname}"
    urdf_path = _get_ycb_7dof_urdf_path(mesh_dir) if voc_7dof else _get_ycb_1dof_urdf_path(mesh_dir)
    # 7-DOF URDF has explicit floating joint; root (world) must be fixed so we get 6+1=7 DOF (not 6+6+1).
    fixed = True if voc_7dof else False
    return {
        "name": ycb_class_name,
        "object_type": "ycb",
        "mesh_fname": mesh_fname,
        "urdf_path": urdf_path,
        "base_init_pos": [0.0, 0.0, 0.3],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "base_init_qpos": [0.0],
        "convexify": True,
        "fixed": fixed,
        "actuated": False,
        "offset_pos": [0.0, 0.0, 0.0],
        "color": None,
        "kp": 1000.0,
        "kv": 100.0,
        "force_range": 200.0,
    }


def get_arctic_object_cfg(name="box", convexify=True, decomp=True, texture_mesh=False):
    obj_cfg = {
        "name": name,
        "base_init_pos": [0.0, 0.0, 0.3], # make this above the ground to avoid startup delay
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "base_init_qpos": [0.0], # joint angle at rest 
        "num_sample_vertics": 300,
        "convexify": convexify,
        "fixed": False, 
        "actuated": False, # whether to control the object joint!
        "kp": 1000.0,
        "kv": 100.0,
        "force_range": 200.0,
        "collect_data": False,
        "offset_pos": [0.0, 0.0, 0.0], # useful for visualization
        "color": None,
        "show_link_frame": False,
    }

    data_dir = get_asset_path("arctic")
    urdf_path = join(data_dir, f"{name}/{name}.urdf")
    if decomp:
        urdf_path = join(data_dir, f"{name}/decomp/{name}_decomp.urdf")
        print(f"Using the decomp-ed urdf: {urdf_path}")
    assert os.path.exists(urdf_path), f"{urdf_path} does not exist"
    if texture_mesh:
        obj_cfg["texture_meshes"] = dict()
        for part in ["top", "bottom"]:
            mesh_fname = join(data_dir, f"{name}/{part}_textured.obj")
            mesh_tex = join(data_dir, f"{name}/{part}_texture.jpg")
            # mesh_tex = join(data_dir, f"{name}/decomp/material.jpg")
            obj_cfg["texture_meshes"][part] = dict(fname=mesh_fname, tex=mesh_tex)

    obj_cfg["urdf_path"] = urdf_path

    if name == "box":
        obj_cfg["base_init_pos"] = [0.0597, -0.2476,  1.0354]
        obj_cfg["base_init_quat"] = [-0.6413,  0.2875,  0.6467, -0.2964]
    
    obj_cfg['bottom_mesh_fname'] = join(data_dir, name, "bottom_watertight_tiny.stl")
    if not os.path.exists(obj_cfg['bottom_mesh_fname']):
        # try the obj file, some object don't have both stl and obj 
        obj_cfg['bottom_mesh_fname'] = join(data_dir, name, "bottom_watertight_tiny.obj")
    obj_cfg['top_mesh_fname'] = join(data_dir, name, "top_watertight_tiny.stl")
    if not os.path.exists(obj_cfg['top_mesh_fname']):
        obj_cfg['top_mesh_fname'] = join(data_dir, name, "top_watertight_tiny.obj")
    assert os.path.exists(obj_cfg['bottom_mesh_fname']), f"{obj_cfg['bottom_mesh_fname']} does not exist"
    assert os.path.exists(obj_cfg['top_mesh_fname']), f"{obj_cfg['top_mesh_fname']} does not exist"
    return obj_cfg

 
class ArticulatedObject:
    """ Assume only one joint """
    def __init__(
        self, 
        obj_cfg, 
        device, 
        scene, 
        num_envs,
        obs_scale={'contact_norm': 0.05, 'root_lin_vel': 2.0, 'root_ang_vel': 0.25},
        demo_data=None,
        visualize_contact=False,   
        disable_collision=False,
    ):
        self.name = obj_cfg["name"]
        self.cfg = obj_cfg
        self.kp = obj_cfg["kp"]
        self.kv = obj_cfg["kv"]
        self.force_range = obj_cfg["force_range"]
        self.device = device
        self.initialized = False
        self.entity = None
        self.num_joints = 1 
        self.dof_idxs = None
        self.obs_scale = obs_scale
        self.has_revolute_joint = True
 
        base_pos = obj_cfg["base_init_pos"]
        base_quat = obj_cfg["base_init_quat"]
        self.offset_pos = obj_cfg["offset_pos"]
        base_pos = [base_pos[i] + self.offset_pos[i] for i in range(3)]
        self.demo_states = None 
        self.demo_dofs = None
        self.num_demo_frames = 0
        if demo_data is not None and demo_data != {}:
            # overwrite base pose 
            base_pos, base_quat = self.set_demo_states(demo_data)
            self.num_demo_frames = self.demo_states.shape[0]

        self.init_pos = torch.tensor(
            base_pos, dtype=torch.float32, device=self.device
        ) 
        self.init_quat = torch.tensor(
            base_quat, dtype=torch.float32, device=self.device
        )
        
        self.init_qpos = torch.tensor(self.cfg["base_init_qpos"], dtype=torch.float32, device=self.device)
        if self.demo_states is not None:
            self.init_qpos = self.demo_states[0, 7:8].clone()

        self.num_envs = num_envs
        self.scene = scene 
        
        spawn_pos = self.init_pos.cpu().numpy()
        spawn_quat = self.init_quat.cpu().numpy()
        # YCB VOC (7-DOF) already encodes absolute root pose in floating-joint targets.
        # Spawn at origin to avoid adding init pose twice (URDF base pose + floating translation).
        if obj_cfg.get("object_type") == "ycb" and obj_cfg.get("fixed", False):
            spawn_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            spawn_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        entity = scene.add_entity(
            gs.morphs.URDF(
                fixed=self.cfg.get("fixed", False),
                file=self.cfg["urdf_path"],
                pos=spawn_pos,
                quat=spawn_quat,
                convexify=self.cfg.get("convexify", True),
                recompute_inertia=False,
                collision=(not disable_collision), 
                visualization=(False if 'texture_meshes' in obj_cfg else True),
            ),
            visualize_contact=visualize_contact,
            # vis_mode="collision", 
            surface=gs.surfaces.Smooth(color=obj_cfg.get("color")) if obj_cfg.get("color") is not None else None,
        )
        movable_joints = [joint for joint in entity.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        is_ycb = obj_cfg.get("object_type") == "ycb"
        self.is_ycb = is_ycb
        if is_ycb:
            # YCB: fixed base (retarget) -> 1 DOF; free base (training) -> 7 DOFs for VOC (position, rotation, virtual joint).
            if entity.n_dofs == 1:
                self.dof_idxs = [0]
                self._voc_dof_idxs = None
            else:
                assert entity.n_dofs == 7, f"YCB object expected 1 or 7 DOFs, got {entity.n_dofs}"
                self.dof_idxs = [6]  # only the revolute for state_diff / dof_pos
                self._voc_dof_idxs = list(range(7))
            self.num_joints = 1
            # DexYCB objects are rigid in semantics; the revolute DOF is a virtual/dummy control slot.
            self.has_revolute_joint = False
        else:
            assert len(movable_joints) == 1, f"len(movable_joints)={len(movable_joints)}"
            assert all([isinstance(joint.dof_idx_local, int) for joint in movable_joints]), "Only one dof per joint is supported"
            self.dof_idxs = [joint.dof_idx_local for joint in movable_joints]
            self._voc_dof_idxs = None  # use dof_idxs for VOC
            self.num_joints = 1
            self.has_revolute_joint = True

        self.texture_meshes = dict()
        self._part_surface_meshes = None
        if 'texture_meshes' in obj_cfg:
            # add two more meshes for top and bottom
            for part, info in obj_cfg["texture_meshes"].items():
                mesh = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=info["fname"],
                    fixed=False,
                    scale=0.001,
                    collision=False,  
                ),
                surface=gs.surfaces.Default( 
                        diffuse_texture=gs.textures.ImageTexture(
                            image_path=info["tex"], 
                        ),
                    ),
                )
                self.texture_meshes[part] = mesh
        self.link_frames = dict()
        if obj_cfg.get("show_link_frame", False):
            frame_parts = ["object"] if is_ycb else ["top", "bottom"]
            for part in frame_parts:
                for axis, color in zip(["x", "y", "z"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
                    mesh = scene.add_entity(
                        morph=gs.morphs.Mesh(
                            file=f"/home/mandi/chiral/assets/{axis}_axis.stl",
                            fixed=False,
                            scale=0.2,
                            collision=False,  
                        ),
                        surface=gs.surfaces.Rough(color=color)
                    )
                    self.link_frames[f"{part}_{axis}"] = mesh

                
        self.entity = entity
        self.n_links = len(entity.links)
        self.link_names = [link.name for link in entity.links] # this is ordered 'bottom', 'top'!!
        self.link_name_to_local_idx = {name: i for i, name in enumerate(self.link_names)}
        self.surface_link_names = self._infer_surface_link_names()
        self.surface_link_idxs = [self.link_name_to_local_idx[name] for name in self.surface_link_names]
        self.coll_idxs_global = [link.idx for link in entity.links if len(link.geoms) > 0]

        self.actuated = self.cfg.get("actuated", False)
        self.initialize_value_buffers()
        self.initialized = True
        self.obs_dim, self.obs_dims = self.compute_obs_dim()
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        self.collect_data = self.cfg.get("collect_data", False)
        if self.collect_data:
            print("Collecting data from object")
        self.episode_data = defaultdict(list)
        self.post_built = False

    def _infer_surface_link_names(self):
        """Surface-part names used for mesh sampling/contact/tip-distance observations."""
        if self.is_ycb:
            if "object" in self.link_name_to_local_idx:
                return ["object"]
            # Fallback: use collision-bearing links if custom URDF differs.
            return [link.name for link in self.entity.links if len(link.geoms) > 0]

        preferred = [name for name in ("top", "bottom") if name in self.link_name_to_local_idx]
        if len(preferred) > 0:
            return preferred
        return [link.name for link in self.entity.links if len(link.geoms) > 0]

    def get_surface_part_names(self):
        return list(self.surface_link_names)
    
    def post_scene_build_setup(self):
        obj_cfg = self.cfg  
        if self.actuated:
            # set kp kv!
            self.set_joint_gains(self.kp, self.kv, self.force_range)
            if getattr(self, "_voc_dof_idxs", None) is not None:
                # YCB 7-DOF path: drive floating-base dofs directly from demo root pose.
                self.demo_dofs = self._compose_voc_targets(
                    self.demo_states[:, :3],
                    self.demo_states[:, 3:7],
                    self.demo_states[:, 7:8],
                )
            else:
                demo_dofs = [] # use this to control actuated obj
                for i in range(self.demo_states.shape[0]):
                    _state = self.demo_states[i]
                    self.set_object_state(
                        root_pos=_state[:3][None].repeat(self.num_envs, 1),
                        root_quat=_state[3:7][None].repeat(self.num_envs, 1),
                        joint_qpos=_state[7:8][None].repeat(self.num_envs, 1),
                    )
                    dofs_pos = self.entity.get_dofs_position()
                    demo_dofs.append(dofs_pos[0])
                self.demo_dofs = torch.stack(demo_dofs, dim=0)
        self.post_built = True

    def _quat_wxyz_to_rotvec(self, quat):
        """Convert quaternion (w,x,y,z) to rotation vector (rx, ry, rz)."""
        q = torch.as_tensor(quat, dtype=torch.float32, device=self.device)
        q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)
        w = torch.clamp(q[..., 0], -1.0, 1.0)
        xyz = q[..., 1:4]
        sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, w[..., None])
        axis = xyz / torch.clamp(sin_half, min=1e-8)
        rotvec = axis * angle
        small = sin_half < 1e-6
        # For tiny angles: sin(theta/2) ~= theta/2  ->  rotvec ~= 2*xyz
        rotvec = torch.where(small.expand_as(rotvec), 2.0 * xyz, rotvec)
        return rotvec

    def _compose_voc_targets(self, root_pos, root_quat, joint_qpos):
        """Compose YCB VOC control targets: [tx, ty, tz, rx, ry, rz, dummy_joint]."""
        root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=self.device)
        root_quat = torch.as_tensor(root_quat, dtype=torch.float32, device=self.device)
        joint_qpos = torch.as_tensor(joint_qpos, dtype=torch.float32, device=self.device)
        if root_pos.ndim == 1:
            root_pos = root_pos[None]
        if root_quat.ndim == 1:
            root_quat = root_quat[None]
        if joint_qpos.ndim == 1:
            joint_qpos = joint_qpos[:, None]
        rotvec = self._quat_wxyz_to_rotvec(root_quat)
        return torch.cat([root_pos, rotvec, joint_qpos], dim=-1)

    def set_demo_states(self, demo_data):
        obj_pos = demo_data["obj_pos"]
        obj_quat = demo_data["obj_quat"]
        obj_arti = demo_data["obj_arti"]
        if len(obj_arti.shape) == 1:
            obj_arti = obj_arti[:, None]    
        obj_pos += np.array(self.offset_pos)[None] # shift!!
        
        base_pos, base_quat = obj_pos[0], obj_quat[0]
        self.demo_states = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
        self.demo_states = torch.tensor(self.demo_states, dtype=torch.float32, device=self.device)
        return base_pos, base_quat
    
    def set_to_demo_step(self, step=0):
        assert self.demo_states is not None, "demo_states is None"
        assert step < self.demo_states.shape[0], f"step={step} >= demo_states.shape[0]={self.demo_states.shape[0]}"
        demo_state = self.demo_states[step]
        self.set_object_state(
            root_pos=demo_state[:3][None].repeat(self.num_envs, 1),
            root_quat=demo_state[3:7][None].repeat(self.num_envs, 1),
            joint_qpos=demo_state[7:8][None].repeat(self.num_envs, 1),
        )
        return
        

    def fill_gain_tensor(self, val, num_dofs, num_envs, device):
        # shape is either 1d, num_dofs, or num_envs x num_dofs, fill in the missing dim 
        if isinstance(val, (int, float)):
            val = torch.tensor([val], dtype=torch.float32, device=self.device)
            batched_val = val.repeat(num_dofs)[None].repeat(num_envs, 1)
        elif isinstance(val, torch.Tensor):
            if len(val.shape) == 1 and val.shape[0] == num_dofs:
                batched_val = val[None].repeat(num_envs, 1) # (num_dofs,) -> (1, num_dofs) -> (num_envs, num_dofs)
            elif len(val.shape) == 1 and val.shape[0] == 1:
                batched_val = val.repeat(num_dofs)[None].repeat(num_envs, 1) # (1,) -> (1, num_dofs) -> (num_envs, num_dofs)
            elif len(val.shape) == 1 and val.shape[0] == num_envs:
                batched_val = val[:, None].repeat(1, num_dofs) # (num_envs,) -> (num_envs, 1) -> (num_envs, num_dofs)
            else:
                assert val.shape == (num_envs, num_dofs), f"val.shape={val.shape}"
                batched_val = val
        else:
            raise not NotImplementedError
        if num_envs == 1:
            batched_val = batched_val[0] # remove first dim
        return batched_val.to(device)
    
    def interpolate_demo_states(self, multiplier=1.0):
        """ interpolate between demo states, shape (T, 8) -> (T*multiplier, 8) """
        assert self.demo_states is not None, "demo_states is None"
        old_T = self.demo_states.shape[0]
        if multiplier == 1.0:
            return self.demo_states

        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation, Slerp

        new_T = int(num_frames * multiplier) 
        old_times = np.arange(old_T)
        target_times = np.linspace(0, old_T-1, new_T)
        # linear interpolation for position and articulation
        new_pos = torch.zeros((new_T, 3), dtype=torch.float32, device=self.device)
        func = interp1d(old_times, self.demo_states[:, :3].cpu().numpy(), axis=0)
        new_pos = func(target_times)
        
        # slerp for quaternion
        rotations = Rotation.from_quat(self.demo_states[:, 3:7].cpu().numpy())
        slerp = Slerp(old_times, rotations)
        new_quat = slerp(target_times).as_quat()

        # linear interpolation for articulation
        new_arti = torch.zeros((new_T, 1), dtype=torch.float32, device=self.device)
        func = interp1d(old_times, self.demo_states[:, 7:8].cpu().numpy(), axis=0)
        new_arti = func(target_times)
        
        new_states = torch.tensor(
            np.concatenate([new_pos, new_quat, new_arti], axis=1), 
            dtype=torch.float32, device=self.device
        )
        assert new_states.shape[0] == new_T, f"new_states.shape={new_states.shape}, new_T={new_T}"
        return new_states

    def set_joint_gains(self, kp=None, kv=None, force_range=None, env_idxs=None):
        num_envs = self.num_envs if env_idxs is None else len(env_idxs)
        dof_idxs = getattr(self, "_voc_dof_idxs", None) or self.dof_idxs
        num_dofs = len(dof_idxs)
        if kp is not None:
            batched_kp = self.fill_gain_tensor(kp, num_dofs, num_envs, self.device) 
            self.entity.set_dofs_kp(batched_kp, dof_idxs, envs_idx=env_idxs)
        if kv is not None:
            batched_kv = self.fill_gain_tensor(kv, num_dofs, num_envs, self.device)
            self.entity.set_dofs_kv(batched_kv, dof_idxs, envs_idx=env_idxs)
        if force_range is not None:
            lower_f = self.fill_gain_tensor(-1 * force_range, num_dofs, num_envs, self.device)
            upper_f = self.fill_gain_tensor(force_range, num_dofs, num_envs, self.device)
            self.entity.set_dofs_force_range(lower_f, upper_f, dof_idxs, envs_idx=env_idxs)
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs) 
        
    def sample_mesh_vertices(self, num_samples: int, part="top", seed=42) -> torch.Tensor:
        import trimesh
        # YCB/single-mesh objects use "mesh_fname"; ARCTIC uses "top_mesh_fname", "bottom_mesh_fname"
        mesh_fname = self.cfg.get(f"{part}_mesh_fname") or self.cfg.get("mesh_fname")
        assert mesh_fname is not None, f"No mesh_fname or {part}_mesh_fname in object config"
        mesh = trimesh.load(mesh_fname)
        if isinstance(mesh, trimesh.Scene):
            vertices = np.concatenate([m.vertices for m in mesh.geometry.values()], axis=0)
        else:
            vertices = mesh.vertices
        num_vertices = vertices.shape[0]
        replace = num_samples > num_vertices
        np.random.seed(seed)
        idxs = np.random.choice(num_vertices, num_samples, replace=replace)
        return torch.tensor(vertices[idxs].astype(np.float32), device=self.device)

    def _load_part_surface_meshes(self):
        if self._part_surface_meshes is not None:
            return self._part_surface_meshes
        import trimesh
        self._part_surface_meshes = dict()
        for part in ["top", "bottom"]:
            mesh_fname = self.cfg.get(f"{part}_mesh_fname", None)
            if mesh_fname is None or not os.path.exists(mesh_fname):
                continue
            self._part_surface_meshes[part] = trimesh.load(mesh_fname, force="mesh", process=False)
        return self._part_surface_meshes

    def query_part_surface_local(self, part: str, points_local: np.ndarray):
        meshes = self._load_part_surface_meshes()
        if part not in meshes:
            raise KeyError(f"Mesh for part '{part}' is not available")
        mesh = meshes[part]
        query = np.asarray(points_local, dtype=np.float32)
        closest_points, _, tri_ids = mesh.nearest.on_surface(query)
        normals = mesh.face_normals[tri_ids]
        normals = normals / np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
        return closest_points.astype(np.float32), normals.astype(np.float32)

    def query_part_surface_world(self, part: str, points_world: np.ndarray, env_idx: int = 0):
        env_idx = int(env_idx)
        link_idx = self.link_names.index(part)
        part_pos = self.part_pos[env_idx, link_idx].detach().cpu().numpy()
        part_quat = self.part_quat[env_idx, link_idx].detach().cpu().numpy()
        local_points = points_world_to_local_np(points_world, part_pos, part_quat)
        return self.query_part_surface_local(part, local_points)

    def initialize_value_buffers(self):
        self.root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.root_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        
        self.root_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.root_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.part_pos = torch.zeros((self.num_envs, self.n_links, 3), dtype=torch.float32, device=self.device)
        self.part_quat = torch.zeros((self.num_envs, self.n_links, 4), dtype=torch.float32, device=self.device)

        self.dof_pos = torch.zeros((self.num_envs, self.num_joints), dtype=torch.float32, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_joints), dtype=torch.float32, device=self.device)
        
        self.contact_force = torch.zeros((self.num_envs, self.n_links, 3), dtype=torch.float32, device=self.device)
        self.state_diff = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)

    def update_value_buffers(self):
        assert self.initialized, "Object not initialized"
        entity = self.entity
        assert self.dof_idxs is not None, "dof_idxs is None"
        self.part_pos[:] = entity.get_links_pos()
        self.part_quat[:] = entity.get_links_quat()
        # 7-DOF YCB: root is "world" (fixed); use "base" link (idx 1) for moving pose.
        if getattr(self, "_voc_dof_idxs", None) is not None:
            base_idx = self.link_name_to_local_idx.get("base", 1)
            self.root_pos[:] = self.part_pos[:, base_idx]
            self.root_quat[:] = self.part_quat[:, base_idx]
            self.root_ang_vel[:] = entity.get_ang()
            self.root_lin_vel[:] = entity.get_vel()
        else:
            self.root_pos[:] = entity.get_pos()
            self.root_quat[:] = entity.get_quat()
            self.root_ang_vel[:] = entity.get_ang()
            self.root_lin_vel[:] = entity.get_vel()

        self.dof_pos[:] = entity.get_dofs_position(self.dof_idxs)
        self.dof_vel[:] = entity.get_dofs_velocity(self.dof_idxs)
        self.contact_force[:] = entity.get_links_net_contact_force()
        if self.demo_states is not None:
            demo_goal_t = torch.where(
                self.episode_length_buf >= self.num_demo_frames - 1, self.num_demo_frames - 1, self.episode_length_buf + 1)
            self.state_diff[:] = self.demo_states[demo_goal_t] - torch.cat(
                [self.root_pos, self.root_quat, self.dof_pos], dim=-1)
    
    def get_nan_envs(self):
        """ check if obj state values has nan """
        nan_envs = torch.isnan(self.root_pos).any(dim=-1)
        for values in [self.root_quat, self.root_ang_vel, self.root_lin_vel, self.dof_pos, self.dof_vel]:
            nan_envs |= torch.isnan(values).any(dim=-1)
        return nan_envs
        
    def get_observations(self):
        assert self.initialized, "Object not initialized" 
        obs_dict = { 
            "parts_pos": self.part_pos.flatten(start_dim=1),
            "parts_quat": self.part_quat.flatten(start_dim=1),
            "dof_pos": self.dof_pos, 
            "state_diff": self.state_diff,
            "root_ang_vel": self.root_ang_vel,
            "root_lin_vel": self.root_lin_vel, 
        }
        for k, scale in self.obs_scale.items():
            if k in obs_dict:
                obs_dict[k] *= scale 
        return obs_dict  

    def get_part_pose(self, part='top'):
        if part not in self.link_name_to_local_idx and self.is_ycb:
            # Keep legacy callers robust for rigid YCB objects.
            part = self.surface_link_names[0]
        idx = self.link_names.index(part)
        pose = torch.cat([self.part_pos[:, idx], self.part_quat[:, idx]], dim=-1)
        return pose

    def compute_obs_dim(self):
        dims = dict( 
            parts_pose_dim=7*self.n_links,
            root_ang_vel_dim=3,
            root_lin_vel_dim=3,
            dof_pos_dim=self.num_joints, 
            state_diff_dim=8,
        )
        return sum(dims.values()), dims

    def reset_idx(self, env_idxs=None, episode_start=None, reset_gains=False):
        assert self.initialized, "Object not initialized"
        if env_idxs is None:
            env_idxs = np.arange(self.num_envs)
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        if episode_start is not None:
            assert episode_start.shape[0] == len(env_idxs), f"reset_episode_start.shape={episode_start.shape}"
        
        init_qpos = self.init_qpos
        if episode_start is not None and torch.any(episode_start): # non-zero starts
            init_qpos = self.demo_states[episode_start, 7: 8].clone() # shape should be (N, 1)
        
        # try randomize the kp/kp
        if env_idxs is None:
            env_idxs = np.arange(self.num_envs)
        
        if self.actuated and reset_gains:
            dof_idxs = getattr(self, "_voc_dof_idxs", None) or self.dof_idxs
            num_dofs = len(dof_idxs) 
            rand_scale = torch.rand((len(env_idxs), 1)) * 0.3 + 0.7 # range [0.5, 1.0]
            kp = torch.ones(len(env_idxs), num_dofs) * self.kp * rand_scale
            kp = kp.to(self.device)
            kv = torch.ones(len(env_idxs), num_dofs) * self.kv * rand_scale 
            kv = kv.to(self.device)
            self.entity.set_dofs_kp(kp, dof_idxs, envs_idx=env_idxs)
            self.entity.set_dofs_kv(kv, dof_idxs, envs_idx=env_idxs)


        self.dof_pos[env_idxs, :] = init_qpos
        self.dof_vel[env_idxs, :] = 0.0
        init_pos = self.init_pos
        if episode_start is not None and torch.any(episode_start):
            init_pos = self.demo_states[episode_start, :3].clone()
        self.root_pos[env_idxs, :] = init_pos

        init_quat = self.init_quat
        if episode_start is not None and torch.any(episode_start):
            init_quat = self.demo_states[episode_start, 3:7].clone()
        self.root_quat[env_idxs, :] = init_quat

        if getattr(self, "_voc_dof_idxs", None) is not None:
            voc_targets = self._compose_voc_targets(
                self.root_pos[env_idxs],
                self.root_quat[env_idxs],
                self.dof_pos[env_idxs],
            )
            self.entity.set_dofs_position(
                position=voc_targets,
                dofs_idx_local=self._voc_dof_idxs,
                zero_velocity=True,
                envs_idx=env_idxs,
            )
        else:
            self.entity.set_dofs_position(
                position=self.dof_pos[env_idxs],
                dofs_idx_local=self.dof_idxs,
                zero_velocity=True,
                envs_idx=env_idxs,
            )
            self.entity.set_pos(
                pos=self.root_pos[env_idxs],
                envs_idx=env_idxs,
            )
            self.entity.set_quat(
                quat=self.root_quat[env_idxs],
                envs_idx=env_idxs,
            )

        self.root_ang_vel[env_idxs, :] = 0.0
        self.root_lin_vel[env_idxs, :] = 0.0
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)

        self.part_pos[env_idxs, :, :] = 0.0
        self.part_quat[env_idxs, :, :] = 0.0

        self.contact_force[env_idxs, :] = 0.0
        self.episode_length_buf[env_idxs] = 0
        if episode_start is not None:
            self.episode_length_buf[env_idxs] = episode_start
        self.state_diff[env_idxs, :] = 0.0

    def reset(self):
        self.reset_idx()

    def set_object_state(self, root_pos, root_quat, joint_qpos, env_idxs=None):
        assert self.initialized, "Object not initialized"
        if root_pos.shape == (3,):
            root_pos = root_pos[None]

        if root_quat.shape == (4,):
            root_quat = root_quat[None]

        if len(joint_qpos.shape) == 1:
            joint_qpos = joint_qpos[None]
        assert joint_qpos.shape[-1] == self.num_joints, f"joint_qpos.shape={joint_qpos.shape}"
        if getattr(self, "_voc_dof_idxs", None) is not None:
            voc_targets = self._compose_voc_targets(root_pos, root_quat, joint_qpos)
            self.entity.set_dofs_position(
                position=voc_targets,
                dofs_idx_local=self._voc_dof_idxs,
                zero_velocity=True,
                envs_idx=env_idxs,
            )
        else:
            self.entity.set_pos(
                pos=root_pos,
                envs_idx=env_idxs,
            )
            self.entity.set_quat(
                quat=root_quat,
                envs_idx=env_idxs,
            )
            self.entity.set_dofs_position(
                position=joint_qpos,
                dofs_idx_local=self.dof_idxs,
                zero_velocity=True,
                envs_idx=env_idxs,
            )
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)
        self.update_value_buffers()  
    
    def step(self, env_idxs=None):
        assert self.initialized, "Object not initialized" 
        assert self.post_built, "Must call post_scene_build_setup before stepping"
        if self.actuated: 
            demo_goal_t = torch.where(
                self.episode_length_buf >= self.num_demo_frames - 1, self.num_demo_frames - 1, self.episode_length_buf + 1) 
            targets = self.demo_dofs[demo_goal_t]
            self.entity.control_dofs_position(targets)
        if len(self.texture_meshes) > 0:
            for part, mesh in self.texture_meshes.items():
                pose = self.get_part_pose(part)
                mesh.set_pos(pose[:, :3])
                mesh.set_quat(pose[:, 3:7])
        if len(self.link_frames) > 0:
            # visualize the root frame on axis meshes:
            for key, mesh in self.link_frames.items():
                part, _axis = key.rsplit("_", 1)
                pose = self.get_part_pose(part)
                mesh.set_pos(pose[:, :3])
                mesh.set_quat(pose[:, 3:7])

        self.episode_length_buf += 1
        return 
     
    def flush_episode_data(self):
        if len(self.episode_data) == 0:
            return dict()
        _data = dict()
        for k in ['obj_pos', 'obj_quat', 'obj_arti']:
            # _data[k] = np.stack(self.episode_data[k], axis=0)
            _data[k] = torch.stack(self.episode_data[k], dim=0)
        self.episode_data = defaultdict(list)
        return _data

    def collect_data_step(self):
        if self.collect_data: # first env only
            self.update_value_buffers() 
            self.episode_data["obj_pos"].append(self.root_pos[0])
            self.episode_data["obj_quat"].append(self.root_quat[0])
            self.episode_data["obj_arti"].append(self.dof_pos[0])
    
    def transform_part_vertices(self, mesh_verts, part="top"):
        """ 
        mesh_verts: (1, num_verts, 3)
        transform the mesh vertices to the current object pose 
        """
        pose = self.get_part_pose(part)
        quat = pose[:, 3:7]
        matrices = matrix_from_quat(quat)
        offsets = pose[:, :3].unsqueeze(1)
        transformed = torch.einsum("nij,nkj->nki", matrices, mesh_verts) + offsets
        return transformed
    
    def get_part_vertices(self, part="top", num_verts=300): 
        mesh_verts = self.sample_mesh_vertices(num_verts, part)[None].to(self.device)
        transformed = self.transform_part_vertices(mesh_verts, part)    
        return transformed
    
    def get_object_vertices(self, num_verts=300):
        """ return:
        - current object's transformed vertices: (num_envs, num_verts * num_parts, 3)
        - part ids: (num_verts * num_parts,)
        """
        parts = self.get_surface_part_names()
        verts_list = []
        ids_list = []
        for part_id, part in enumerate(parts):
            verts_list.append(self.get_part_vertices(part, num_verts))
            ids_list.append(torch.full((num_verts,), part_id, dtype=torch.long, device=self.device))
        verts = torch.cat(verts_list, dim=1)
        ids = torch.cat(ids_list, dim=0)
        return verts, ids
