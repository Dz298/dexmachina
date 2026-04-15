import os  
import torch 
import numpy as np
from os.path import join
from dexmachina.envs.reward_utils import position_distance, rotation_distance, chamfer_distance, transform_contact
from dexmachina.envs.math_utils import matrix_from_quat


def get_reward_cfg(last_n_frame=-1):
    reward_cfg = {
        "obj_pos_beta": 20.0,
        "obj_rot_beta": 5.0,
        "obj_arti_beta": 20.0,
        "obj_pos_weight": 2.0,
        "obj_rot_weight": 3.0,
        "obj_arti_weight": 5.0,  
        
        "last_n_frame": last_n_frame,
        "multiply_task_rew": True,
        "multiply_all_rew": False, 
        "task_rew_weight": 1.0,

        "exp_kpt_first": True,
        "imi_rew_weight":  0.0, 
        "imi_wrist_weight": 0.0, # do a weighted avg between fingertip and wrist poses
        "imi_wrist_rot_beta": 3.0,
        "imi_wrist_pos_beta": 10.0,
        "imi_fingertip_beta": 20.0,

        "bc_rew_weight": 0.0,
        "bc_beta": 500.0,

        "contact_rew_weight": 0.0,
        "contact_rew_function": "exp", # exp or sigmoid
        "wrist_frame_contact": True,
        "contact_beta": 30.0,
        "contact_a": 100.0,
        "contact_b": 5.0,
        "multiply_frame_contact": True,
        "mask_zero_contact": True, # if both policy and demo has no contact, reward is 0 (1 if False)
        "contact_phase_penalty": 0,
        "thumb_weight": 1.0,  # Weight multiplier for thumb contact (>1 means thumb is more important)

        "mask_well_track": False, 
        "scale_well_track": 1.0,
        "force_penalty": 0.1,  # ~60 contact pairs in each env
        "action_penalty": 0.0,

        "reach_rew_weight": 0.0,
        "reach_sigma": 0.2,
        "grasp_gate_weight": 0.0,
        "grasp_gate_force_threshold": 1.0,
        "grasp_gate_min_fingers": 2,
        "soft_mask_contact": False,
        "soft_mask_alpha": 1.0,

        "objdex_baseline": False,
        "use_retarget_contact": False,
        "retarget_objframe": True, # if True, the contact is in the object frame, otherwise in the wrist frame
        # Matched retarget contact only: scale per-link reward by exp(kappa * (dot - 1)), dot = clamp((policy-demo)_hat · n_hat, 0, 1)
        # in world frame; 0 disables. Same convention as virtual-force normals (outward object normal).
        "contact_align_kappa": 0.0,

    } 
    return reward_cfg

class RewardModule:
    def __init__(self, reward_cfg, demo_data, retarget_data, device):
        self.cfg = reward_cfg 
        self.last_n_frame = reward_cfg.get("last_n_frame", -1)
        self.exp_kpt_first = reward_cfg.get("exp_kpt_first", False)
        self.imi_rew_weight = reward_cfg["imi_rew_weight"]        
        self.imi_wrist_weight = reward_cfg.get("imi_wrist_weight", 0.0)
        self.task_rew_weight = reward_cfg["task_rew_weight"] 
        self.bc_rew_weight = reward_cfg.get("bc_rew_weight", 0.0)
        self.bc_wrist_weight = reward_cfg.get("bc_wrist_weight", 0.0)  
        self.demo_data = demo_data  
        self.multiply_task_rew = reward_cfg["multiply_task_rew"]
        self.multiply_all_rew = reward_cfg.get("multiply_all_rew", False)
        self.mask_well_track = reward_cfg.get("mask_well_track", False)
        self.scale_well_track = reward_cfg.get("scale_well_track", 1.0)
        self.obj_pos_beta = reward_cfg["obj_pos_beta"]
        self.obj_rot_beta = reward_cfg["obj_rot_beta"]
        self.obj_arti_beta = reward_cfg["obj_arti_beta"]
        if reward_cfg.get("objdex_baseline", False):
            print("Using the lambdas from ObjDex baseline")
            self.obj_pos_beta = 1.0
            self.obj_rot_beta = 20.0
            self.obj_arti_beta = 5.0
            self.imi_rew_weight = 0.0
            self.bc_rew_weight = 0.0
            self.task_rew_weight = 1.0
            self.contact_rew_weight = 0.0
        
        self.use_imi_rew = self.imi_rew_weight > 0.0
        self.use_bc_rew = self.bc_rew_weight > 0.0
        self.obj_pos_weight = reward_cfg["obj_pos_weight"] # not use because we multiply the rewards
        self.obj_rot_weight = reward_cfg["obj_rot_weight"]
        self.obj_arti_weight = reward_cfg["obj_arti_weight"]  

        self.wrist_frame_contact = reward_cfg.get("wrist_frame_contact", False)
        self.use_retarget_contact = reward_cfg.get("use_retarget_contact", False) # if True, the loaded contact is from retargeted hands
        self.retarget_objframe = reward_cfg.get("retarget_objframe", True) # if True, the contact is in the object frame, otherwise in the wrist frame
        self.contact_rew_weight = reward_cfg.get("contact_rew_weight", 0.0)
        self.contact_beta = reward_cfg.get("contact_beta", 10.0)
        self.contact_a = reward_cfg.get("contact_a", 100.0)
        self.contact_b = reward_cfg.get("contact_b", 5.0)
        self.sigmoid_offset = torch.tensor(-1 / (1.0 + np.exp(-self.contact_b)), device=device) + 1.0
        self.contact_phase_penalty = reward_cfg.get("contact_phase_penalty", 0.0)
        self.multiply_frame_contact = reward_cfg.get("multiply_frame_contact", True) 
        self.mask_zero_contact = reward_cfg.get("mask_zero_contact", True)
        self.contact_rew_function = reward_cfg.get("contact_rew_function", "exp")
        self.thumb_weight = reward_cfg.get("thumb_weight", 1.0)
        if self.thumb_weight != 1.0:
            print(f"Using thumb weight: {self.thumb_weight}x")
        self.contact_align_kappa = float(reward_cfg.get("contact_align_kappa", 0.0))
        self._contact_align_enabled = False

        self.reach_rew_weight = reward_cfg.get("reach_rew_weight", 0.0)
        self.reach_sigma = reward_cfg.get("reach_sigma", 0.2)
        self.grasp_gate_weight = reward_cfg.get("grasp_gate_weight", 0.0)
        self.grasp_gate_force_threshold = reward_cfg.get("grasp_gate_force_threshold", 1.0)
        self.grasp_gate_min_fingers = reward_cfg.get("grasp_gate_min_fingers", 2)
        self.soft_mask_contact = reward_cfg.get("soft_mask_contact", False)
        self.soft_mask_alpha = reward_cfg.get("soft_mask_alpha", 1.0)
        if self.soft_mask_contact:
            self.mask_zero_contact = False

        self.load_demo(demo_data, retarget_data, device)
        if self.contact_align_kappa > 0.0:
            if not self.use_retarget_contact:
                print("contact_align_kappa > 0 but use_retarget_contact is False; disabling contact normal alignment.")
            elif not all(
                f"contact_normals_local_{s}" in self.demo_tensors for s in ("left", "right")
            ):
                print(
                    "contact_align_kappa > 0 but demo lacks contact_normals_local_{left,right}; "
                    "disabling contact normal alignment."
                )
            else:
                self._contact_align_enabled = True
                print(f"Contact reward normal alignment enabled (kappa={self.contact_align_kappa})")

    def load_demo(self, demo_data, retarget_data, device):
        self.demo_tensors = dict()
        demo_keys = ["obj_pos", "obj_quat", "obj_arti"]
        self.active_sides = [
            s for s in ("left", "right")
            if f"contact_links_{s}" in demo_data or (s in retarget_data and "kpts_data" in retarget_data.get(s, {}))
        ]
        if not self.active_sides:
            self.active_sides = ["left", "right"]
        if self.contact_rew_weight > 0.0:
            for side in self.active_sides:
                demo_keys.append(f"contact_links_{side}")

        for key in demo_keys:
            assert key in demo_data, f"{key} not in demo_data"
            self.demo_tensors[key] = torch.tensor(
                demo_data[key], dtype=torch.float32, device=device
            )

        optional_demo_keys = [
            "contact_links_valid_left",
            "contact_links_valid_right",
            "contact_links_local_left",
            "contact_links_local_right",
            "contact_normals_local_left",
            "contact_normals_local_right",
        ]
        for key in optional_demo_keys:
            if key not in demo_data:
                continue
            self.demo_tensors[key] = torch.tensor(
                demo_data[key], dtype=torch.float32, device=device
            )

        if self.use_imi_rew:
            for side in self.active_sides:
                key = f"kpts_{side}"
                self.demo_tensors[key] = torch.tensor(
                    retarget_data[side]["kpts_data"]["kpt_pos"],
                    dtype=torch.float32, device=device
                )
        if self.contact_rew_weight > 0.0 or (self.use_imi_rew and self.imi_wrist_weight > 0.0):
            for side in self.active_sides:
                key = f"wrist_pose_{side}"
                if isinstance(retarget_data[side]["wrist_pose"], torch.Tensor):
                    self.demo_tensors[key] = retarget_data[side]["wrist_pose"].clone().to(device)
                else:
                    self.demo_tensors[key] = torch.tensor(
                        retarget_data[side]["wrist_pose"], dtype=torch.float32, device=device
                    )

        self.contact_link_weights = {}
        if self.contact_rew_weight > 0.0 and self.thumb_weight != 1.0:
            for side in self.active_sides:
                if side in demo_data and "collision_link_names" in demo_data[side]:
                    link_names = demo_data[side]['collision_link_names']
                    # Create weight tensor: thumb links get thumb_weight, others get 1.0
                    weights = []
                    for name in link_names:
                        is_thumb = 'thumb' in name.lower()
                        weights.append(self.thumb_weight if is_thumb else 1.0)
                    self.contact_link_weights[side] = torch.tensor(weights, dtype=torch.float32, device=device)
                    thumb_count = sum(1 for n in link_names if 'thumb' in n.lower())
                    print(f"Contact link weights ({side}): {thumb_count} thumb links with {self.thumb_weight}x weight")
        
        # check all the data have the same first dim size 
        assert all(
            [self.demo_tensors[key].shape[0] == self.demo_tensors["obj_pos"].shape[0] for key in self.demo_tensors.keys()]
        ), f"First dim size mismatch: {[self.demo_tensors[key].shape[0] for key in self.demo_tensors.keys()]}"
        self.demo_length = self.demo_tensors["obj_pos"].shape[0] 
        print(f"Loaded demo data with length {self.demo_length}")
    
    def get_demo_length(self):  
        return self.demo_length
    
    def match_demo_state(self, demo_key, episode_length_buf):
        """ returns shape (num_envs, num_features) """
        assert demo_key in self.demo_tensors, f"Key {demo_key} not found in demo tensors"
        demo_t = torch.where(episode_length_buf >= self.demo_length, self.demo_length-1, episode_length_buf)
        # print(demo_t, episode_length_buf)
        return self.demo_tensors[demo_key][demo_t]
    
    def compute_task_reward(self, obj_pos, obj_quat, obj_arti, demo_pos, demo_quat, episode_length_buf):
        if obj_pos is None or obj_quat is None or obj_arti is None or self.task_rew_weight == 0.0:
            # dummy reward
            task_rew = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device)
            return task_rew, dict(task_rew=task_rew)
        
        demo_arti = self.match_demo_state("obj_arti", episode_length_buf)
        pos_dist = position_distance(obj_pos, demo_pos)
        rot_dist = rotation_distance(obj_quat, demo_quat)
        # arti_dist = torch.mean((obj_arti - demo_arti)**2, dim=-1)

        if len(obj_arti.shape) > 1:
            obj_arti = obj_arti.flatten(start_dim=0)
            
        arti_dist = (obj_arti - demo_arti)**2 / 2.0
        
        # these should all be shape (num_envs, )!!
        obj_pos_rew = torch.exp(-self.obj_pos_beta * pos_dist)  
        obj_rot_rew = torch.exp(-self.obj_rot_beta * rot_dist)
        obj_arti_rew = torch.exp(-self.obj_arti_beta * arti_dist)
        assert obj_pos_rew.shape == obj_rot_rew.shape == obj_arti_rew.shape, "Shape mismatch"
        # if joint is closed, don't compute arti reward 
        if self.multiply_task_rew:
            task_rew = self.task_rew_weight * obj_pos_rew * obj_rot_rew * obj_arti_rew # scale is stil [0,1]
            # print(f"========= weighted task rew: {task_rew} | rew weight: {self.task_rew_weight * 10 } obj pos rew: {obj_pos_rew} | obj rot rew: {obj_rot_rew} | obj arti rew: {obj_arti_rew} =========")
        else:
            # obj_arti_rew = torch.where(demo_arti <= 0.0001, torch.zeros_like(obj_arti_rew), obj_arti_rew)
            task_rew = self.task_rew_weight * (
                self.obj_pos_weight * obj_pos_rew + 
                self.obj_rot_weight * obj_rot_rew + 
                self.obj_arti_weight * obj_arti_rew
                )
            
        # give a bonus if obj state is well tracked and articulation joint is open 
        well_track = (pos_dist < 0.005) & (rot_dist < 0.1) & (arti_dist < 0.1)  & (demo_arti > 0.1)
        if self.scale_well_track > 1.0:
            task_rew = torch.where(well_track, task_rew * self.scale_well_track, task_rew)

        rew_dict = dict(
            pos_dist=pos_dist,
            rot_dist=rot_dist,
            arti_dist=arti_dist,
            obj_pos_rew=obj_pos_rew,
            obj_rot_rew=obj_rot_rew,
            obj_arti_rew=obj_arti_rew,
            task_rew=task_rew.clone(), # NOTE: otherwise it's the same tensor as the total reward
            well_track=well_track,
        )  

        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame, 
                torch.zeros(task_rew.shape, device=task_rew.device, dtype=torch.bool),
                torch.ones(task_rew.shape, device=task_rew.device, dtype=torch.bool)
                )
            task_rew[tomask] = 0.0
        return task_rew, rew_dict

    def compute_keypoint_dist(self, keypoint_pos, episode_length_buf, left_hand=True):
        demo_key = "kpts_left" if left_hand else "kpts_right"
        demo_kpts = self.match_demo_state(demo_key, episode_length_buf) 
        return position_distance(keypoint_pos, demo_kpts)
    
    def compute_wrist_reward(self, wrist_pose, episode_length_buf, side='left'):
        demo_wrist = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        wrist_rot_dist = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])
        wrist_pos_dist = position_distance(wrist_pose[:, :3], demo_wrist[:, :3])
        rot_beta = self.cfg["imi_wrist_rot_beta"]
        pos_beta = self.cfg["imi_wrist_pos_beta"]
        wrist_rew = (torch.exp(-rot_beta * wrist_rot_dist) + torch.exp(-pos_beta * wrist_pos_dist)) / 2.0 
        return wrist_rew, wrist_pos_dist, wrist_rot_dist

    def compute_reach_reward(self, kpts_left, kpts_right, obj_pos):
        """Always-on dense reward: fingertip-to-object-center distance."""
        obj_expanded = obj_pos.unsqueeze(1)  # (B, 1, 3)
        dists = []
        if kpts_left is not None:
            dists.append(torch.norm(kpts_left - obj_expanded, dim=-1).mean(dim=-1))
        if kpts_right is not None:
            dists.append(torch.norm(kpts_right - obj_expanded, dim=-1).mean(dim=-1))
        mean_dist = torch.stack(dists, dim=0).mean(dim=0)
        reach_rew = self.reach_rew_weight * (1.0 - torch.tanh(mean_dist / self.reach_sigma))
        return reach_rew, mean_dist

    def compute_grasp_gate(self, contact_forces):
        """Binary grasp quality: 1 if >= min_fingers robot links have force > threshold."""
        if contact_forces is None:
            return None, None
        force_norm = torch.norm(contact_forces, dim=-1)  # (B, n_obj_parts, n_robot_links)
        max_force_per_link = force_norm.max(dim=1).values  # (B, n_robot_links)
        has_contact = max_force_per_link > self.grasp_gate_force_threshold
        n_contacts = has_contact.sum(dim=-1).float()
        grasp_gate = (n_contacts >= self.grasp_gate_min_fingers).float()
        return grasp_gate, n_contacts

    def compute_imitation_reward(
        self,
        wrist_pose_left,
        wrist_pose_right,
        kpts_left: torch.Tensor,
        kpts_right: torch.Tensor,
        episode_length_buf: torch.Tensor,
    ):
        kpt_dists_per_side = {}
        for side, kpts in [("left", kpts_left), ("right", kpts_right)]:
            if kpts is None or side not in self.active_sides:
                continue
            kpt_dists_per_side[side] = self.compute_keypoint_dist(
                kpts, episode_length_buf, left_hand=(side == "left")
            )
        if not kpt_dists_per_side:
            return torch.zeros(episode_length_buf.shape[0], device=episode_length_buf.device), {}
        stacked = torch.stack(list(kpt_dists_per_side.values()))
        fingertip_dist = torch.mean(stacked.mean(dim=0), dim=-1)
        beta = self.cfg["imi_fingertip_beta"]
        if self.exp_kpt_first:
            fingertip_rew = torch.mean(
                torch.stack([torch.exp(-beta * d) for d in kpt_dists_per_side.values()]).mean(dim=0),
                dim=-1,
            )
        else:
            fingertip_rew = torch.exp(-beta * fingertip_dist)

        if self.imi_wrist_weight > 0.0:
            wrist_rews, pos_dists = [], []
            for side, wp in [("left", wrist_pose_left), ("right", wrist_pose_right)]:
                if wp is None or side not in self.active_sides:
                    continue
                wr, pd, _ = self.compute_wrist_reward(wp, episode_length_buf, side=side)
                wrist_rews.append(wr)
                pos_dists.append(pd)
            if wrist_rews:
                wrist_rew = torch.stack(wrist_rews).mean(dim=0)
                wrist_dist = torch.stack(pos_dists).mean(dim=0)
                imi_rew = self.imi_wrist_weight * wrist_rew + (1.0 - self.imi_wrist_weight) * fingertip_rew
                keypoint_dist = self.imi_wrist_weight * wrist_dist + (1.0 - self.imi_wrist_weight) * fingertip_dist
            else:
                imi_rew = fingertip_rew
                keypoint_dist = fingertip_dist
        else:
            imi_rew = fingertip_rew
            keypoint_dist = fingertip_dist

        imi_rew *= self.imi_rew_weight
        rew_dict = dict(
            imi_rew=imi_rew,
            keypoint_dist=keypoint_dist,
        )
        for side, dist in kpt_dists_per_side.items():
            rew_dict[f"kpts_dist_{side}"] = dist
        if self.last_n_frame > 0:
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame,
                torch.zeros(imi_rew.shape, device=imi_rew.device, dtype=torch.bool),
                torch.ones(imi_rew.shape, device=imi_rew.device, dtype=torch.bool),
            )
            imi_rew[tomask] = 0.0
        return imi_rew, rew_dict 

    def contact_dist_to_rew(self, dist, function='exp'):
        if function == 'exp':
            return torch.exp(-self.contact_beta * dist)
        elif function == 'sigmoid':
            axb = self.contact_a * dist + self.contact_b
            rew = 1.0 / (1.0 + torch.exp(-axb)) + self.sigmoid_offset.to(dist.device) 
            return rew
        else:
            raise ValueError(f"Unknown contact reward function: {function}")

    def compute_hand_contact_reward(
        self,
        contact_link_pos, # shape (N, num_obj_links * num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links * num_hand_links, 1)
        wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7)
        episode_length_buf,
        demo_obj_pose, # shape (N, 7)
        side='left',
    ):
        demo_wrist_pose = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf) 
        # N, num_links * 2, 4 (last dim is contact pair ID) -> NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top' 
        demo_positions = demo_contacts[:, :, :3]
        demo_valid_contact = demo_contacts[:, :, -1] > 0.0 # (part id is <= 0 if no contact)
        chamfer_dists = dict()
        contact_rewards = dict()

        positions = contact_link_pos[:, :, :3]
        valid_mask = contact_link_pos[:, :, -1] > 0.0 
        for part_id in [1, 2]:
            # consider contact invalid if part_id is different
            demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
            part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                 
            for frame in ['obj', 'wrist']:
                if not self.wrist_frame_contact and frame == 'wrist':
                    continue
                demo_pose = demo_obj_pose if frame == 'obj' else demo_wrist_pose
                demo_in_frame = transform_contact(
                    demo_positions, demo_pose
                )
                pose = obj_pose if frame == 'obj' else wrist_pose
                in_frame = transform_contact(
                    positions, pose
                )
                dist = chamfer_distance(
                    in_frame, demo_in_frame, part_valid, demo_part_valid
                )
                chamfer_dists[f"CD_{side}_p{part_id}_frame_{frame}"] = dist
                rew = self.contact_dist_to_rew(dist, self.contact_rew_function)
                both_zero = (part_valid.sum(dim=-1) == 0) & (demo_part_valid.sum(dim=-1) == 0)
                if self.mask_zero_contact:
                    rew = torch.where(both_zero, torch.zeros_like(rew), rew)
                else:
                    rew = torch.where(both_zero, torch.ones_like(rew), rew)
                contact_rewards[f"conrew_{side}_p{part_id}_frame_{frame}"] = rew
            
        # per-part contact should be added, because sometimes full contact coverage is not feasible 
        if self.multiply_frame_contact:
            # multiply the rew in obj & wrist frame such that one does not dominate the other
            contact_rews = []
            for part_id in [1, 2]:
                obj_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_obj"]
                if not self.wrist_frame_contact:
                    wrist_rew = 1.0
                else:
                    wrist_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_wrist"] 
                multi_rew = obj_rew * wrist_rew
                contact_rewards[f"mul_conrew_{side}_p{part_id}"] = multi_rew
                contact_rews.append(multi_rew)
            contact_rew = torch.stack(contact_rews, dim=-1).mean(dim=-1)

        else: # sum & average all contact rewards
            contact_rew = sum(contact_rewards.values()) / len(contact_rewards)

        contact_dict = {**chamfer_dists, **contact_rewards, f"contact_rew_{side}": contact_rew}
        # add penalty for scenarios where demo has 0 contact but env has contact
        if self.contact_phase_penalty != 0.0: 
            mismatch = torch.zeros_like(contact_rew).bool()
            num_mismatch = torch.zeros_like(contact_rew)
            for part_id in [1, 2]:
                demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
                part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                part_mismatch = (demo_part_valid.sum(dim=-1) == 0) & (part_valid.sum(dim=-1) > 0)
                mismatch = mismatch | part_mismatch
                num_mismatch += part_mismatch.float()
            contact_rew[mismatch] += self.contact_phase_penalty
            contact_dict['num_mismatch'] = num_mismatch

        return contact_rew, contact_dict
    
    def compute_matched_contact_per_hand(
        self,
        contact_link_pos, # shape (N, num_obj_links, num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links, num_hand_links, 1)
        episode_length_buf,
        # wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7) 
        demo_obj_pose,
        side='left',
        max_distance=1.0,
    ):
        """ 
        Instead of computing point cloud reward, here we assume demo contacts and policy contacts
        are from the same set of dex hand links and hence each contact point has a matched target position in the demo 
        """ 
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf)
        n_parts = contact_link_pos.shape[1]
        # Demo may have 2 parts (ARCTIC top/bottom); YCB has 1 part - slice to match
        if demo_contacts.shape[1] > n_parts:
            demo_contacts = demo_contacts[:, :n_parts, :, :].clone()
        # ARCTIC: retargeted contacts are (N, 2, num_links, 4), first row 'top' second 'bottom'; flip to match policy (bottom first)
        if n_parts == 2:
            demo_contacts = demo_contacts.clone()[:, [1, 0], :, :]
        assert demo_contacts.shape[1] == contact_link_pos.shape[1], f"Shape mismatch: {demo_contacts.shape} vs {contact_link_pos.shape}"
        assert demo_contacts.shape[2] == contact_link_pos.shape[2], f"Shape mismatch: {contact_link_pos.shape} vs {demo_contacts.shape}"
        demo_valids = demo_contacts[:, :, :, -1] > 0.0 # (part id is <= 0 if no contact)
        v = contact_link_valid
        if v.dtype != torch.bool:
            v = v > 0
        if v.dim() == 4 and v.shape[-1] == 1:
            v = v.squeeze(-1)
         
        # need to reshape this to (N, num_obj_links * num_hand_links, 3) to do the transformation first
        if self.retarget_objframe:
            bsize, nparts, nlinks = demo_contacts.shape[:3]
            demo_pos_global = demo_contacts[:,:,:,:3].reshape(
                bsize, nparts * nlinks, 3
            ) # (N, num_obj_links * num_hand_links, 3)
            demo_pos = transform_contact(demo_pos_global, demo_obj_pose)        
            demo_pos = demo_pos.reshape(bsize, nparts, nlinks, 3)  

            policy_pos_global = contact_link_pos[:, :, :, :3].reshape(
                bsize, nparts * nlinks, 3
            ) # (N, num_obj_links, num_hand_links, 3)
            policy_pos = transform_contact(policy_pos_global, obj_pose)
            # then reshape it back!
            policy_pos = policy_pos.reshape(bsize, nparts, nlinks, 3)
        else:
            demo_pos = demo_contacts[:, :, :, :3]
            policy_pos = contact_link_pos[:, :, :, :3]
        
        
        both_invalid_mask = torch.logical_not(demo_valids) & torch.logical_not(v)
        # only one valid 
        one_valid_mask = torch.logical_xor(demo_valids, v)
        # compute distance between demo and policy contact points
        dist = position_distance(demo_pos, policy_pos)
        if self.mask_zero_contact:
            # if both invalid, set distance to 0
            dist = torch.where(both_invalid_mask, torch.ones_like(dist) * max_distance, dist)
        else:
            # if both invalid, set distance to 0
            dist = torch.where(both_invalid_mask, torch.zeros_like(dist), dist)
        # if one valid, set distance to a large value
        dist = torch.where(one_valid_mask, torch.ones_like(dist) * max_distance, dist) # (B, 2, nlinks)
        
        # different ways to compute the distance: 
        # if take mean, encourages all contacts to be close to the targets 
        # if take min, encourages only one contact to be close to the target
        # part_dist = torch.mean(dist, dim=-1) # (N, num_obj_links=2)  

        if not self._contact_align_enabled:
            return dist, torch.ones_like(dist)

        demo_xyz = demo_contacts[:, :, :, :3]
        policy_xyz = contact_link_pos[:, :, :, :3]
        diff = policy_xyz - demo_xyz
        demo_normals = self.match_demo_state(f"contact_normals_local_{side}", episode_length_buf)
        demo_normals = demo_normals.clone()[:, [1, 0], :, :]
        R = matrix_from_quat(demo_obj_pose[:, 3:7])
        # Same as normals_local_to_world_torch: n_w = n @ R.T; avoid (B,2,L,3)@(B,3,3) matmul
        # (PyTorch would treat ... x (L,3) as matrix batch and break).
        n_w = torch.einsum("bpni,bki->bpnk", demo_normals, R)
        d_hat = torch.nn.functional.normalize(diff, dim=-1, eps=1e-8)
        n_hat = torch.nn.functional.normalize(n_w, dim=-1, eps=1e-8)
        dot = (d_hat * n_hat).sum(dim=-1).clamp(min=0.0, max=1.0)
        align_w = torch.exp(self.contact_align_kappa * (dot - 1.0))

        pair_ok = ~(both_invalid_mask | one_valid_mask)
        diff_norm = torch.norm(diff, dim=-1)
        use_align = pair_ok & (diff_norm >= 1e-5)
        align_w = torch.where(use_align, align_w, torch.ones_like(align_w))
        return dist, align_w
    
    def compute_matched_contact_reward(
        self,
        contacts_link_left,
        contacts_link_valid_left,
        contacts_link_right,
        contacts_link_valid_right,
        obj_pose,
        demo_obj_pose,
        episode_length_buf,
    ):
        rews = dict()
        contact_rew = 0
        total_part_terms = 0
        sides_data = [
            ("left", contacts_link_left, contacts_link_valid_left),
            ("right", contacts_link_right, contacts_link_valid_right),
        ]
        n_active = 0
        for side, contacts, valids in sides_data:
            if contacts is None or valids is None or side not in self.active_sides:
                continue
            n_active += 1
            part_dist, part_align = self.compute_matched_contact_per_hand(
                contacts, valids, episode_length_buf, 
                obj_pose, demo_obj_pose, side=side
            )
            # Get link weights (thumb-weighted if enabled)
            link_weights = self.contact_link_weights.get(side, None)
            n_parts = part_dist.shape[1]  # 2 for ARCTIC, 1 for YCB
            part_names = ("bottom", "top") if n_parts == 2 else ("base",)
            for i in range(n_parts):
                part = part_names[i] if i < len(part_names) else f"part{i}"
                con_dist = part_dist[:, i]  # shape (N, num_links)
                a_w = part_align[:, i]
                rews[f"contact_align_{side}_{part}"] = a_w.mean(dim=-1)
                if self.exp_kpt_first:
                    per_link_rew = self.contact_dist_to_rew(con_dist, self.contact_rew_function)  # (N, num_links)
                    per_link_rew = per_link_rew * a_w
                    # Use weighted mean if weights are available
                    if link_weights is not None:
                        # Weighted mean: sum(w * x) / sum(w)
                        con_rew = (per_link_rew * link_weights).sum(dim=-1) / link_weights.sum()
                        # Log thumb vs other finger rewards separately
                        thumb_mask = link_weights > 1.0  # thumb links have weight > 1
                        if thumb_mask.any():
                            thumb_rew = per_link_rew[:, thumb_mask].mean(dim=-1)
                            other_rew = per_link_rew[:, ~thumb_mask].mean(dim=-1)
                            rews[f"thumb_conrew_{side}_{part}"] = thumb_rew
                            rews[f"other_conrew_{side}_{part}"] = other_rew
                    else:
                        con_rew = per_link_rew.mean(dim=-1)
                else:
                    # For non-exp_kpt_first, apply weights to distances before mean
                    if link_weights is not None:
                        weighted_dist = (con_dist * link_weights).sum(dim=-1) / link_weights.sum()
                        weighted_align = (a_w * link_weights).sum(dim=-1) / link_weights.sum()
                        con_rew = self.contact_dist_to_rew(weighted_dist, self.contact_rew_function) * weighted_align
                    else:
                        con_dist_mean = con_dist.mean(dim=-1)
                        a_mean = a_w.mean(dim=-1)
                        con_rew = self.contact_dist_to_rew(con_dist_mean, self.contact_rew_function) * a_mean
                rews[f"conrew_{side}_{part}"] = con_rew
                rews[f"matched_condist_{side}_{part}"] = con_dist
                contact_rew += con_rew
            total_part_terms += n_parts
        if total_part_terms > 0:
            contact_rew /= float(total_part_terms)
        contact_rew *= self.contact_rew_weight
        rews['con_rew'] = contact_rew
        
        # Aggregate thumb vs other reward for easier WandB tracking
        if self.thumb_weight != 1.0:
            thumb_rews = [v for k, v in rews.items() if k.startswith('thumb_conrew')]
            other_rews = [v for k, v in rews.items() if k.startswith('other_conrew')]
            if thumb_rews:
                rews['thumb_con_rew'] = torch.stack(thumb_rews).mean(dim=0)
                rews['other_con_rew'] = torch.stack(other_rews).mean(dim=0)
        
        return contact_rew, rews


    def compute_contact_reward(
        self,
        obj_pose,
        wrist_pose_left,
        wrist_pose_right,
        contacts_link_left,
        contacts_link_valid_left,
        contacts_link_right,
        contacts_link_valid_right,
        episode_length_buf,
    ):
        demo_obj_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_obj_quat = self.match_demo_state("obj_quat", episode_length_buf)
        demo_obj_pose = torch.cat([demo_obj_pos, demo_obj_quat], dim=1)
        contact_rews = []
        contact_dict = {}
        for side, contacts, valids, wrist in [
            ("left", contacts_link_left, contacts_link_valid_left, wrist_pose_left),
            ("right", contacts_link_right, contacts_link_valid_right, wrist_pose_right),
        ]:
            if contacts is None or valids is None or side not in self.active_sides:
                continue
            c_rew, c_dict = self.compute_hand_contact_reward(
                contacts, valids, wrist, obj_pose, episode_length_buf, demo_obj_pose, side=side
            )
            contact_rews.append(c_rew)
            contact_dict.update(c_dict)
        contact_rew = torch.stack(contact_rews).mean(dim=0) if contact_rews else torch.zeros(obj_pose.shape[0], device=obj_pose.device) 
        contact_rew *= self.contact_rew_weight
        contact_dict["con_rew"] = contact_rew
        return contact_rew, contact_dict

    def reshape_contact_with_label(self, contact_link_pos, contact_link_valid):
        """ 
        Reshape the environment contact to better match with demos:
        In: 
            contact_link_pos_left: (N, 2, num_hand_links, 3), 2 is each object part
            contact_link_valid_left: (N, 2, num_hand_links)
        Out: 
            reshaped_contact_link_pos_left: (N, 2*num_hand_links, 4): 4 is (x, y, z, part_id), part_id is 1 or 2 or 0 if no valid
            reshaped contact_link_valid: (N, 2, num_hand_links) -> (N, 2*num_hand_links)
        """
        # first, add a part_id dim so (N, 2, num_links, 4) and last dim is part_id label
        labeled = torch.cat(
            [contact_link_pos, torch.zeros_like(contact_link_pos)[:, :, :, 0:1]], dim=-1
        ) # (N, 2, num_links, 4) 
        # fill in the labels 1 or 2 if valid contact
        valid_reshaped = contact_link_valid[..., None] # (N, 2, num_links, 1)
        valid = torch.cat(
            [valid_reshaped, torch.zeros_like(valid_reshaped)], dim=-1
        ) # -> (N, 2, num_links, 2)
        for part_idx in [0, 1]: # bottom, top (sice obj urdf is flipped for parsing error)
            label_id = part_idx + 1 
            # Need flipping, because in demos, id=2 is bottom
            label_id = 1 if label_id == 2 else 2 
            labeled[:, part_idx, :, -1] = torch.where(
                contact_link_valid[:, part_idx, :],  label_id, 0
            ) 
            valid[:, part_idx, :, 1] = torch.where(
                contact_link_valid[:, part_idx, :], label_id, 0
            )

        # reshape to (N, num_links*2, 4)
        contact_link_reshaped = labeled.view(
            labeled.shape[0], -1, labeled.shape[-1]
        )
        contact_valid_reshaped = valid.view(
            valid.shape[0], -1, valid.shape[-1]
        )
        
        return contact_link_reshaped, contact_valid_reshaped

    def compute_reward(
        self, 
        actions,
        bc_dist, # (num_envs, left+right hand num of joints) 
        obj_pos,
        obj_quat,
        obj_arti,
        kpts_left,
        kpts_right,
        contact_link_pos_left, 
        contact_link_valid_left,  # shape (N, num_obj_links, num_hand_links)
        contact_link_pos_right,
        contact_link_valid_right,
        wrist_pose_left,
        wrist_pose_right,
        contact_forces, # shape (N, num_obj_links, num_hand_links, 3)
        episode_length_buf,
    ):  
        demo_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_quat = self.match_demo_state("obj_quat", episode_length_buf) 
        rew, rew_dict = self.compute_task_reward(
            obj_pos, obj_quat, obj_arti, 
            demo_pos, demo_quat,
            episode_length_buf
        )

        # --- r_reach: always-on dense fingertip-to-object reward ---
        if self.reach_rew_weight > 0.0 and (kpts_left is not None or kpts_right is not None) and obj_pos is not None:
            reach_rew, reach_dist = self.compute_reach_reward(kpts_left, kpts_right, obj_pos)
            rew += reach_rew
            rew_dict["reach_rew"] = reach_rew
            rew_dict["reach_dist"] = reach_dist

        # --- r_grasp_gate: binary grasp quality indicator ---
        grasp_gate = None
        if self.grasp_gate_weight > 0.0 and contact_forces is not None:
            grasp_gate, n_grasp_contacts = self.compute_grasp_gate(contact_forces)
            rew += self.grasp_gate_weight * grasp_gate
            rew_dict["grasp_gate"] = grasp_gate
            rew_dict["n_grasp_contacts"] = n_grasp_contacts

        if self.bc_rew_weight > 0.0:
            bc_rew = self.bc_rew_weight * torch.exp(-self.cfg["bc_beta"] * bc_dist)
            bc_rew = torch.mean(bc_rew, dim=-1)
            rew_dict["bc_dist"] = bc_dist.mean(dim=-1)
            rew_dict["bc_rew"] = bc_rew

        if self.use_imi_rew: 
            imi_rew, imi_rew_dict = self.compute_imitation_reward(
                wrist_pose_left, wrist_pose_right,
                kpts_left, kpts_right, episode_length_buf
            )
            if "well_track" in rew_dict and self.mask_well_track:
                imi_rew = torch.where(rew_dict["well_track"], torch.zeros_like(imi_rew), imi_rew) 
            rew_dict.update(imi_rew_dict)
 
        if self.contact_rew_weight > 0.0: 
            obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
            demo_pose = torch.cat([demo_pos, demo_quat], dim=1)
            if self.use_retarget_contact:
                contact_rew, contact_dict = self.compute_matched_contact_reward(
                    contact_link_pos_left, contact_link_valid_left,
                    contact_link_pos_right, contact_link_valid_right,
                    obj_pose, demo_pose, episode_length_buf
                )
            else:
                left_reshaped, left_valid = self.reshape_contact_with_label(
                    contact_link_pos_left, contact_link_valid_left
                    )
                right_reshaped, right_valid = self.reshape_contact_with_label(
                    contact_link_pos_right, contact_link_valid_right
                    )
                contact_rew, contact_dict = self.compute_contact_reward(
                    obj_pose, 
                    wrist_pose_left, 
                    wrist_pose_right,
                    left_reshaped, left_valid, 
                    right_reshaped, right_valid,
                    episode_length_buf
                ) 
            if "well_track" in rew_dict and self.mask_well_track:
                contact_rew = torch.where(rew_dict["well_track"], torch.zeros_like(contact_rew), contact_rew)   

            # --- soft mask: smooth contact density instead of hard zero ---
            if self.soft_mask_contact and contact_forces is not None:
                cf_norm = torch.norm(contact_forces, dim=-1)  # (B, n_parts, n_links)
                n_active = (cf_norm > 1.0).flatten(start_dim=1).sum(dim=-1).float()
                contact_density = 1.0 - torch.exp(-self.soft_mask_alpha * n_active)
                contact_rew = contact_rew * contact_density
                rew_dict["contact_density"] = contact_density

            rew_dict.update(contact_dict)
            rew_dict["con_rew"] = contact_rew
        
        if self.multiply_all_rew:
            aux_rew = 1.0
            if self.use_imi_rew:
                aux_rew *= imi_rew
            if self.contact_rew_weight > 0.0:
                aux_rew *= contact_rew
            if self.bc_rew_weight > 0.0:
                aux_rew *= bc_rew
            rew += aux_rew * 0.5
        else:
            if self.use_imi_rew:
                rew += imi_rew
            if self.contact_rew_weight > 0.0:
                rew += contact_rew
            if self.bc_rew_weight > 0.0:
                rew += bc_rew
        
        if self.cfg["force_penalty"] > 0.0 and contact_forces is not None:
            force_norm = torch.norm(contact_forces, dim=-1).flatten(start_dim=1)
            high_force = torch.where(force_norm > 500.0, force_norm - 500.0, torch.zeros_like(force_norm))
            high_force = torch.mean(high_force, dim=-1)
            force_penalty = self.cfg["force_penalty"] * high_force 
            rew -= force_penalty
            rew_dict["force_penalty"] = force_penalty
        
        if self.cfg["action_penalty"] > 0.0:
            action_penalty = torch.mean(actions**2, dim=-1) * self.cfg["action_penalty"]
            rew -= action_penalty
            rew_dict["action_penalty"] = action_penalty  
        return rew, rew_dict
 
    def get_reward_keys(self):
        keys = []
        if self.task_rew_weight > 0:
            keys.append('task')
        if self.imi_rew_weight > 0:
            keys.append('imi')
        if self.contact_rew_weight > 0:
            keys.append('con')
        if self.bc_rew_weight > 0:
            keys.append('bc')
        return keys 
