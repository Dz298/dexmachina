import os

import numpy as np
import torch

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

    def _infer_contact_count(self) -> int:
        # Match fit.py's default contact budget while remaining safe for simpler hands.
        return min(12, self.hand_model.n_contact_candidates)

    def _get_contact_groups(self):
        groups = []
        prefixes = ("index", "middle", "ring", "thumb")
        for prefix in prefixes:
            link_indices = [
                link_idx
                for link_name, link_idx in self.hand_model.link_name_to_link_index.items()
                if link_name.startswith(prefix)
            ]
            if link_indices:
                groups.append(
                    torch.tensor(link_indices, dtype=torch.long, device=self.device)
                )
        return groups

    def _make_zero_force_closure_metric(self):
        def _zero_metric(contact_pts, *args, **kwargs):
            zeros = torch.zeros(contact_pts.shape[0], dtype=contact_pts.dtype, device=contact_pts.device)
            return zeros, None

        return _zero_metric

    def _calculate_losses(self, weight_dict, energy_fnc, energy_kwargs):
        try:
            losses = calculate_energy(
                self.hand_model,
                self.object_model,
                energy_names=list(weight_dict.keys()),
                energy_fnc=energy_fnc,
                **energy_kwargs,
            )
            return losses, energy_fnc
        except RuntimeError as exc:
            msg = str(exc)
            if weight_dict.get("E_fc", 0.0) <= 0.0 or ("Q is not SPD" not in msg and "positive-definite" not in msg):
                raise
            print("GraspQPOptimizer: force-closure solve was ill-conditioned; disabling E_fc for this optimization run.")
            weight_dict["E_fc"] = 0.0
            zero_metric = self._make_zero_force_closure_metric()
            losses = calculate_energy(
                self.hand_model,
                self.object_model,
                energy_names=list(weight_dict.keys()),
                energy_fnc=zero_metric,
                **energy_kwargs,
            )
            return losses, zero_metric

    def _select_initial_contact_indices(self, hand_pose: torch.Tensor) -> torch.Tensor:
        all_indices = (
            torch.arange(self.hand_model.n_contact_candidates, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(hand_pose.shape[0], -1)
        )
        self.hand_model.set_parameters(hand_pose, contact_point_indices=all_indices)
        distance, _ = self.object_model.cal_distance(self.hand_model.contact_points)
        n_contact = self._infer_contact_count()
        contact_groups = self._get_contact_groups()
        if not contact_groups:
            return distance.abs().topk(k=n_contact, largest=False).indices

        link_indices = self.hand_model.global_index_to_link_index
        selected = []
        n_groups = len(contact_groups)
        for env_idx in range(hand_pose.shape[0]):
            chosen = []
            chosen_mask = torch.zeros(self.hand_model.n_contact_candidates, dtype=torch.bool, device=self.device)
            remaining = n_contact
            for group_idx, group_links in enumerate(contact_groups):
                target = remaining // max(1, (n_groups - group_idx))
                if target == 0:
                    continue
                group_mask = (link_indices.unsqueeze(1) == group_links.unsqueeze(0)).any(dim=1)
                candidate_ids = torch.nonzero(group_mask, as_tuple=False).squeeze(-1)
                if candidate_ids.numel() == 0:
                    continue
                k = min(target, candidate_ids.numel())
                best_local = distance[env_idx, candidate_ids].abs().topk(k=k, largest=False).indices
                best_ids = candidate_ids[best_local]
                chosen.append(best_ids)
                chosen_mask[best_ids] = True
                remaining -= k

            if remaining > 0:
                candidate_ids = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(-1)
                filler_local = distance[env_idx, candidate_ids].abs().topk(k=remaining, largest=False).indices
                chosen.append(candidate_ids[filler_local])

            selected.append(torch.cat(chosen, dim=0)[:n_contact])

        return torch.stack(selected, dim=0)

    def _get_robot_wrist_pose(self, robot, idx_tensor: torch.Tensor):
        if hasattr(robot, "wrist_pose"):
            wrist_pose = robot.wrist_pose[idx_tensor]
            return wrist_pose[:, :3], wrist_pose[:, 3:]
        return robot.entity.get_pos()[idx_tensor], robot.entity.get_quat()[idx_tensor]

    def _get_finger_joint_positions(self, robot):
        robot_dof_names = robot.actuated_dof_names
        name_to_pos = {name: idx for idx, name in enumerate(robot_dof_names)}
        gqp_dof_names = self.hand_model._actuated_joints_names

        if all(name in name_to_pos for name in gqp_dof_names):
            return [name_to_pos[name] for name in gqp_dof_names]

        if "allegro" in self.hand_name.lower():
            allegro_name_map = {
                "index_joint_0": "joint_0.0",
                "index_joint_1": "joint_1.0",
                "index_joint_2": "joint_2.0",
                "index_joint_3": "joint_3.0",
                "middle_joint_0": "joint_4.0",
                "middle_joint_1": "joint_5.0",
                "middle_joint_2": "joint_6.0",
                "middle_joint_3": "joint_7.0",
                "ring_joint_0": "joint_8.0",
                "ring_joint_1": "joint_9.0",
                "ring_joint_2": "joint_10.0",
                "ring_joint_3": "joint_11.0",
                "thumb_joint_0": "joint_12.0",
                "thumb_joint_1": "joint_13.0",
                "thumb_joint_2": "joint_14.0",
                "thumb_joint_3": "joint_15.0",
            }
            if all(allegro_name_map[name] in name_to_pos for name in gqp_dof_names):
                return [name_to_pos[allegro_name_map[name]] for name in gqp_dof_names]

        raise ValueError(
            "GraspQPOptimizer: joint name mismatch between graspqp and dexmachina.\n"
            f"  graspqp joints : {gqp_dof_names}\n"
            f"  dexmachina joints: {robot_dof_names}"
        )

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

    def optimize_grasps(
        self,
        robot,
        env_idxs,
        iters=50,
        w_dis=100.0,
        w_fc=1.0,
        w_pen=100.0,
        w_spen=10.0,
        w_joints=1.0,
        w_anchor=10.0,
        friction=0.2,
        max_lambda_limit=20.0,
        n_friction_cone=4,
        svd_gain=0.1,
        use_gendexgrasp=True,
    ):
        if not self.initialized or self.object_model.batch_size_each != len(env_idxs):
            self.initialize(len(env_idxs))
            
        B = len(env_idxs)
        idx_tensor = torch.tensor(env_idxs, dtype=torch.long, device=self.device)

        dex_to_gqp = torch.tensor(self._get_finger_joint_positions(robot), dtype=torch.long, device=self.device)
        n_robot_dofs = len(dex_to_gqp)
        if self.hand_model.n_dofs != n_robot_dofs:
            robot_name = getattr(robot, "name", getattr(robot.entity, "name", "<unnamed>"))
            raise ValueError(
                f"GraspQPOptimizer: hand '{self.hand_name}' has {self.hand_model.n_dofs} DOFs in graspqp "
                f"but robot '{robot_name}' has {n_robot_dofs} finger DOFs usable by graspqp. "
                f"graspqp joints: {self.hand_model._actuated_joints_names}"
            )

        # Get world poses
        hand_root_pos, hand_root_quat = self._get_robot_wrist_pose(robot, idx_tensor)
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
        
        contact_point_indices = self._select_initial_contact_indices(hand_pose)
        self.hand_model.set_parameters(hand_pose, contact_point_indices=contact_point_indices)

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
            "E_dis": w_dis,
            "E_fc": w_fc,
            "E_pen": w_pen,
            "E_spen": w_spen,
            "E_joints": w_joints,
        }
        energy_names = [name for name, weight in weight_dict.items() if weight > 0.0]
        energy_fnc = GraspSpanMetricFactory.create(
            GraspSpanMetricFactory.MetricType.GRASPQP,
            solver_kwargs={
                "friction": friction,
                "max_limit": max_lambda_limit,
                "n_cone_vecs": n_friction_cone,
            },
        )
        energy_kwargs = {"svd_gain": svd_gain}
        if use_gendexgrasp:
            energy_kwargs["method"] = "gendexgrasp"
        
        # Initial energy
        losses, energy_fnc = self._calculate_losses(weight_dict, energy_fnc, energy_kwargs)
        
        energy = sum(weight_dict[k] * losses[k] for k in energy_names)
        energy += w_anchor * torch.norm(self.hand_model.hand_pose - anchor_hand_pose, dim=1)

        energy.sum().backward()
        optimizer.zero_grad()

        for step in range(iters):
            s = optimizer.try_step()
            reset_mask = None
            optimizer.zero_grad()

            new_energies, energy_fnc = self._calculate_losses(weight_dict, energy_fnc, energy_kwargs)

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
        final_pose = self.hand_model.hand_pose.detach().clone()
        invalid_pose = ~torch.isfinite(final_pose).all(dim=1)
        if invalid_pose.any():
            final_pose[invalid_pose] = anchor_hand_pose[invalid_pose]
        o_t_h_final = final_pose[:, :3]
        ortho6d_final = final_pose[:, 3:9]
        hand_qpos_final = final_pose[:, 9:]
        
        o_R_h_final = robust_compute_rotation_matrix_from_ortho6d(ortho6d_final)
        
        # Transform back to world
        w_R_h_final = torch.bmm(w_R_o, o_R_h_final)
        w_t_h_final = torch.bmm(w_R_o, o_t_h_final.unsqueeze(2)).squeeze(2) + obj_root_pos

        # roma returns (x,y,z,w); Genesis set_quat expects (w,x,y,z) — convert back
        hand_root_quat_final = _xyzw_to_wxyz(roma.rotmat_to_unitquat(w_R_h_final))
        invalid_quat = ~torch.isfinite(hand_root_quat_final).all(dim=1)
        if invalid_quat.any():
            hand_root_quat_final[invalid_quat] = hand_root_quat[invalid_quat]
            w_t_h_final[invalid_quat] = hand_root_pos[invalid_quat]
            hand_qpos_final[invalid_quat] = hand_qpos[invalid_quat]

        # Reorder graspqp DOF order → dexmachina DOF order before returning
        full_qpos_final = robot.dof_pos[idx_tensor].clone()
        full_qpos_final[:, dex_to_gqp] = hand_qpos_final

        return w_t_h_final, hand_root_quat_final, full_qpos_final
