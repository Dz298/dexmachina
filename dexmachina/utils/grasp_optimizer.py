import os
import torch
import numpy as np

from graspqp.hands import get_hand_model
from graspqp.core.optimizer import MalaStar
from graspqp.metrics import GraspSpanMetricFactory
from graspqp.core.energy import calculate_energy
from graspqp.core.object_model import ObjectModel, SDF_BACKEND
from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d
import warp as wp
import roma


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Convert Genesis (w,x,y,z) quaternions to roma (x,y,z,w)."""
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert roma (x,y,z,w) quaternions to Genesis (w,x,y,z)."""
    return torch.cat([q[..., -1:], q[..., :-1]], dim=-1)

class DexMachinaObjectModel(ObjectModel):
    def __init__(self, obj_env, batch_size_each, device):
        super().__init__(data_root_path="", batch_size_each=batch_size_each, num_samples=0, device=device)
        self.obj_env = obj_env
        self.scale_choice = torch.tensor([1.0], dtype=torch.float, device=self.device)
        self.object_scale_tensor = torch.ones((1, batch_size_each), dtype=torch.float, device=self.device)
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        self.sdf_library = SDF_BACKEND
        self.initialize_from_env()
    
    def initialize_from_env(self):
        import trimesh
        top_mesh = trimesh.load(self.obj_env.cfg.get('top_mesh_fname'))
        bottom_mesh = trimesh.load(self.obj_env.cfg.get('bottom_mesh_fname'))
        
        mesh = top_mesh + bottom_mesh
        self.object_mesh_list = [mesh]
        
        object_verts = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)
        object_faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
        
        self.surface_points_tensor = object_verts.unsqueeze(0).repeat(self.batch_size_each, 1, 1)
        
        self._init_sdf(object_verts, object_faces)

    def _init_sdf(self, object_verts, object_faces):
        if self.sdf_library == "TORCHSDF":
            from torchsdf import index_vertices_by_faces
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
        elif self.sdf_library == "WARP":
            link_vertices, link_faces = object_verts.cpu().numpy(), object_faces.cpu().numpy()
            verts_wp = wp.from_numpy(np.ascontiguousarray(link_vertices), device=str(self.device), dtype=wp.vec3)
            faces_wp = wp.from_numpy(np.ascontiguousarray(link_faces.flatten()), device=str(self.device), dtype=wp.int32)
            wp_mesh = wp.Mesh(points=verts_wp, indices=faces_wp)
            self.object_face_verts_list.append(wp_mesh)
        elif self.sdf_library == "KAOLIN":
            import kaolin
            link_face_verts = kaolin.ops.mesh.index_vertices_by_faces(object_verts.unsqueeze(0), object_faces)
            self.object_face_verts_list.append((link_face_verts, object_faces, object_verts))


class GraspQPOptimizer:
    def __init__(self, env, hand_name="allegro", device="cuda"):
        self.env = env
        self.device = device
        self.hand_name = hand_name
        self.hand_model = None
        self.object_model = None
        self.initialized = False

    def initialize(self, batch_size):
        from graspqp.hands import AVAILABLE_HANDS
        gn = self.hand_name
        # Map dexmachina hand names to graspqp registry keys.
        # orca_hand_left / orca_hand_right are already exact registry keys.
        # For other hands, strip the side suffix if needed.
        if gn not in AVAILABLE_HANDS:
            if "shadow" in gn.lower():
                gn = "shadow_hand"
            elif "allegro" in gn.lower():
                gn = "allegro"
            # orca_hand_left / orca_hand_right stay as-is; others keep their name
        if gn not in AVAILABLE_HANDS:
            raise ValueError(f"GraspQPOptimizer: hand '{self.hand_name}' (resolved to '{gn}') not in graspqp registry {AVAILABLE_HANDS}")

        self.hand_model = get_hand_model(gn, self.device, grasp_type="all")
            
        self.object_model = DexMachinaObjectModel(
            obj_env=self.env.object, 
            batch_size_each=batch_size,
            device=self.device
        )
        self.initialized = True

    def optimize_grasps(self, robot, env_idxs, iters=50, w_pen=100.0, w_spen=10.0, w_joints=1.0, w_anchor=10.0):
        if not self.initialized or self.object_model.batch_size_each != len(env_idxs):
            self.initialize(len(env_idxs))
            
        B = len(env_idxs)
        idx_tensor = torch.tensor(env_idxs, dtype=torch.long, device=self.device)

        n_robot_dofs = len(robot.actuated_dof_idxs)
        if self.hand_model.n_dofs != n_robot_dofs:
            raise ValueError(
                f"GraspQPOptimizer: hand '{self.hand_name}' has {self.hand_model.n_dofs} DOFs in graspqp "
                f"but robot '{robot.entity.name}' has {n_robot_dofs} actuated DOFs in dexmachina. "
                f"graspqp joints: {self.hand_model._actuated_joints_names}"
            )

        # GraspQP (pytorch_kinematics) and Genesis may traverse the URDF in different
        # orders, producing different DOF orderings.  Build a permutation by matching
        # joint names so we can reorder before feeding into GraspQP and reorder back
        # before writing results into Genesis.
        #
        # dex_to_gqp[gqp_i] = dex_j  means graspqp slot gqp_i = dexmachina slot dex_j
        # gqp_to_dex[dex_j] = gqp_i  (inverse permutation, for writing back)
        robot_dof_names = robot.actuated_dof_names
        gqp_dof_names = self.hand_model._actuated_joints_names
        try:
            dex_to_gqp = torch.tensor(
                [robot_dof_names.index(name) for name in gqp_dof_names],
                dtype=torch.long, device=self.device,
            )
        except ValueError as e:
            raise ValueError(
                f"GraspQPOptimizer: joint name mismatch between graspqp and dexmachina.\n"
                f"  graspqp joints : {gqp_dof_names}\n"
                f"  dexmachina joints: {robot_dof_names}"
            ) from e
        gqp_to_dex = torch.zeros_like(dex_to_gqp)
        gqp_to_dex[dex_to_gqp] = torch.arange(n_robot_dofs, device=self.device)

        # Get world poses
        hand_root_pos = robot.entity.get_pos()[idx_tensor]
        hand_root_quat = robot.entity.get_quat()[idx_tensor]
        # Reorder dexmachina DOFs → graspqp DOF order
        hand_qpos = robot.dof_pos[idx_tensor][:, dex_to_gqp]
        
        obj_root_pos = self.env.object.root_pos[idx_tensor]
        obj_root_quat = self.env.object.root_quat[idx_tensor]
        
        # Transform hand pose to object local frame
        # Obj: w_T_o = (obj_root_pos, obj_root_quat)
        # Hand: w_T_h = (hand_root_pos, hand_root_quat)
        # We need o_T_h = (w_T_o)^-1 * w_T_h
        
        # roma expects (x,y,z,w); Genesis returns (w,x,y,z) — convert before use
        w_R_o = roma.unitquat_to_rotmat(_wxyz_to_xyzw(obj_root_quat))
        o_R_w = w_R_o.transpose(1, 2)
        o_pos_w = -torch.bmm(o_R_w, obj_root_pos.unsqueeze(2)).squeeze(2)

        w_R_h = roma.unitquat_to_rotmat(_wxyz_to_xyzw(hand_root_quat))
        o_R_h = torch.bmm(o_R_w, w_R_h)
        o_t_h = torch.bmm(o_R_w, hand_root_pos.unsqueeze(2)).squeeze(2) + o_pos_w
        
        # Convert o_R_h to ortho6d (flatten first two columns)
        # o_R_h is (B, 3, 3). Columns 0 and 1 are (B, 3), so we take them and concat
        ortho6d = torch.cat([o_R_h[:, :, 0], o_R_h[:, :, 1]], dim=1)
        
        # Shape of graspqp hand_pose: (B, 3 + 6 + n_dofs)
        hand_pose = torch.cat([o_t_h, ortho6d, hand_qpos], dim=1).detach().clone()
        hand_pose.requires_grad_(True)
        anchor_hand_pose = hand_pose.clone().detach()
        
        self.hand_model.set_parameters(hand_pose, contact_point_indices="all")

        optim_config = {
            "switch_possibility": 0.5,
            "starting_temperature": 18,
            "temperature_decay": 0.95,
            "annealing_period": 30,
            "step_size": 0.005,
            "stepsize_period": 50,
            "mu": 0.98,
            "device": self.device,
            "batch_size": B,
            "clip_grad": True,
        }
        optimizer = MalaStar(self.hand_model, **optim_config)
        
        weight_dict = {
            "E_pen": w_pen,
            "E_spen": w_spen,
            "E_joints": w_joints,
        }
        energy_names = list(weight_dict.keys())
        energy_fnc = GraspSpanMetricFactory.create(GraspSpanMetricFactory.MetricType.GRASPQP)
        
        # Initial energy
        losses = calculate_energy(
            self.hand_model,
            self.object_model,
            energy_names=energy_names,
            energy_fnc=energy_fnc,
        )
        
        energy = sum(weight_dict[k] * losses[k] for k in energy_names)
        energy += w_anchor * torch.norm(self.hand_model.hand_pose - anchor_hand_pose, dim=1)

        energy.sum().backward()
        optimizer.zero_grad()

        for step in range(iters):
            s = optimizer.try_step()
            reset_mask = None
            optimizer.zero_grad()

            new_energies = calculate_energy(
                self.hand_model,
                self.object_model,
                energy_names=energy_names,
                energy_fnc=energy_fnc,
            )

            new_energy = sum(weight_dict[k] * new_energies[k] for k in energy_names)
            new_energy += w_anchor * torch.norm(self.hand_model.hand_pose - anchor_hand_pose, dim=1)
            new_energy.sum().backward()

            with torch.no_grad():
                accept, t = optimizer.accept_step(
                    energy,
                    new_energy,
                    reset_mask,
                    z_score=torch.zeros(B, device=self.device),
                    z_score_threshold=1.0, 
                )
                energy[accept] = new_energy[accept]
        
        # After optimization, extract world pose
        final_pose = self.hand_model.hand_pose.detach()
        o_t_h_final = final_pose[:, :3]
        ortho6d_final = final_pose[:, 3:9]
        hand_qpos_final = final_pose[:, 9:]
        
        o_R_h_final = robust_compute_rotation_matrix_from_ortho6d(ortho6d_final)
        
        # Transform back to world
        w_R_h_final = torch.bmm(w_R_o, o_R_h_final)
        w_t_h_final = torch.bmm(w_R_o, o_t_h_final.unsqueeze(2)).squeeze(2) + obj_root_pos

        # roma returns (x,y,z,w); Genesis set_quat expects (w,x,y,z) — convert back
        hand_root_quat_final = _xyzw_to_wxyz(roma.rotmat_to_unitquat(w_R_h_final))

        # Reorder graspqp DOF order → dexmachina DOF order before returning
        hand_qpos_final = hand_qpos_final[:, gqp_to_dex]

        return w_t_h_final, hand_root_quat_final, hand_qpos_final
