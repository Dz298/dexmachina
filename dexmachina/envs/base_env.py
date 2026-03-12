import os  
import torch
import numpy as np
import genesis as gs 
from dexmachina.envs.robot import BaseRobot
from dexmachina.envs.object import ArticulatedObject
from dexmachina.envs.rewards import RewardModule
from dexmachina.envs.math_utils import matrix_from_quat, quat_mul, quat_conjugate
from dexmachina.envs.contacts import get_filtered_contacts
from dexmachina.envs.randomizations import RandomizationModule
from dexmachina.envs.curriculum import Curriculum 
from dexmachina.envs.maniptrans_curr import ManipTransCurriculum 
from typing import Dict, List, Tuple, Union
from collections import deque
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from dexmachina.utils.grasp_optimizer import GraspQPOptimizer

TABLE_HEIGHT = 0.6
OBJ_DEFAULT_POS = (0.0597, -0.2476,  1.0354)
OBJ_DEFAULT_ROT = (-0.6413,  0.2875,  0.6467, -0.2964)
CARDBOARD_POS = (0, -0.08, 0.90) 
CAMERA_RES=(160, 160)
ENV_SPACING=(1.0, 1.0)


def get_scene_cfg(
    dt=1/60, 
    zero_gravity=False, 
    show_viewer=False, 
    show_fps=False, 
    batch_dofs_info=False, 
    use_visualizer=False,
    n_rendered_envs=None,
    raytrace=False,
    visualize_contact=False,
    enable_joint_limit=True,
):
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=(0, 0, -9.81) if not zero_gravity else (0, 0, 0),
            #  gravity=(0, 0, 0),
        ), 
        vis_options=gs.options.VisOptions(
            n_rendered_envs=n_rendered_envs,
            show_world_frame=False,
            visualize_contact=visualize_contact,
            segmentation_level="entity", # "link" or "entity" or 'geom'
            # NOTE set to false to speed up rendering!!
            ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=enable_joint_limit,
            max_collision_pairs=100, # default was 100
            batch_dofs_info=batch_dofs_info,
        ),
        viewer_options=gs.options.ViewerOptions( 
            camera_pos=(0.5, 1.5, 1.8), # looking from behind
            camera_lookat=(0.0, -0.15, 1.0),
            camera_fov=30,
        ),
        use_visualizer=use_visualizer,
        show_viewer=show_viewer,
        show_FPS=show_fps, 
    )
    
    if raytrace:
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
    return scene_cfg

def get_env_cfg(
    dt=1/60, 
    use_visualizer=False, 
    show_viewer=False, 
    show_fps=False, 
    zero_gravity=False, 
):  
    scene_kwargs = dict(
        dt=dt, 
        zero_gravity=zero_gravity, 
        use_visualizer=use_visualizer,
        show_viewer=show_viewer, 
        show_fps=show_fps,
        batch_dofs_info=False, 
        # NOTE: this will be slower but allows different gains per env,turn this on for actuated object
        n_rendered_envs=None,
        raytrace=False,
        visualize_contact=False,
        enable_joint_limit=True, # enable joint limits for robots
        )
    camera_kwargs = dict(
        front=dict(
            res=(160, 160),
            # pos=(0.5, -1.5, 1.2),
            # lookat=(0.0, -0.15, 1.0),
            pos=( 0, -1.6,  2.2),
            lookat=(0.0, -0.1, 1.2),
            fov=30,
        ),
        back=dict(
            res=(160, 160),
            pos=(0.4, 1.5, 1.8),
            lookat=(0.0, -0.15, 1.0),
            fov=25,
        ),
    )
    env_cfg = {
        "num_envs": 1, 
        "episode_length": 10,
        "action_clip": 1.0,
        "action_scale": 1.0,
        "obs_clip": 5.0,
        "scene_kwargs": scene_kwargs,   
        "dt": dt,
        "early_reset_threshold": 0.0,
        "early_reset_interval": 5,
        "early_reset_aux_thres": dict(con=0, imi=0, bc=0),
        "record_video": False,
        "render_segmentation": False,
        "texture_cardbox": False,
        "max_video_frames": 0,
        "observe_tip_dist": False,
        "observe_contact_force": False,
        "observe_hand_demo_diff": False,
        "traj_lookahead_frames": 0,
        "use_contact_reward": False,
        'use_rl_games': True,
        "is_eval": False, 
        "rand_init_ratio": 0.0, # randomize initial states  
        "enforce_valid_rand_init_grasp": False,
        "valid_grasp_min_contacts": 1,
        "valid_grasp_contact_thresh": 0.01,
        "valid_grasp_max_contact_err_per_link": 0.08,  # guardrail: reject if mean contact error (m) per link exceeds this
        "valid_grasp_require_both_hands": False,
        "valid_grasp_sample_iters": 2,
        "valid_grasp_sample_count": 32,
        "valid_grasp_sample_std": 0.02,
        "valid_grasp_close_bias": 0.01,
        "valid_grasp_guided_gain": 0.08,
        "valid_grasp_guided_abd_gain": 0.03,
        "valid_grasp_settle_steps": 8,
        "valid_grasp_slip_weight": 2.0,
        "valid_grasp_vel_weight": 0.5,
        "valid_grasp_ang_vel_weight": 0.25,
        "valid_grasp_opposition_weight": 0.0,  # reward weight on thumb-vs-fingers opposition score (higher is better)
        "valid_grasp_functional_opposition_min": 0.15,  # minimum thumb-vs-fingers opposition score; <=0 disables
        "valid_grasp_hold_max_drop": -1.0,  # hard gate (m) on free-hold drop; <=0 disables
        "valid_grasp_hold_max_vel": -1.0,   # hard gate (m/s) on free-hold max linear vel; <=0 disables
        "valid_grasp_hold_max_ang_vel": -1.0,  # hard gate (rad/s) on free-hold max angular vel; <=0 disables
        "valid_grasp_hold_ignore_steps": 0,  # ignore first N unpinned hold steps when measuring stability
        "valid_grasp_contact_persistence_min": 0.0,  # require final contact persistence ratio during hold; <=0 disables
        "valid_grasp_compliance_steps": 50,  # virtual 6D spring steps after unpinning (stiffness decays 1->0)
        "valid_grasp_contact_bonus": 0.01,
        "valid_grasp_refine_mode": "sampling",  # one of: sampling, ik, virtual_force
        "valid_grasp_use_ik": False,
        "valid_grasp_ik_iters": 4,
        "valid_grasp_ik_damping": 1e-3,
        "valid_grasp_ik_eps": 2e-3,
        "valid_grasp_ik_step_clip": 0.12,
        "valid_grasp_ik_close_bias": 0.02,
        "valid_grasp_vf_attract_steps": 20,   # steps of attract+settle (object pinned)
        "valid_grasp_vf_hold_steps": 10,      # steps of free-physics validation (slip measure)
        "valid_grasp_vf_anneal_probe_hold_steps": 5,  # short hold probe per anneal step for stability-aware selection
        "valid_grasp_vf_close_bias": 0.02,
        "valid_grasp_vf_squeeze_torque": 0.0,  # constant close-direction torque bias during pinned close-settle
        "valid_grasp_vf_attract_gain": 0.5,   # J^T attraction gain during pre-tension phase
        "valid_grasp_vf_attract_dq_clip": 0.03,  # per-step joint delta clip for J^T attraction
        "valid_grasp_vf_soft_unpin_steps": 50,  # steps with decreasing gravity compensation after pre-tension
        "valid_grasp_gravity_ramp_steps": 0,  # ramp down gravity compensation over N unpinned hold steps
        "valid_grasp_fallback_to_zero": True,
        "valid_grasp_debug": False,
        "env_spacing": ENV_SPACING,
        "n_envs_per_row": None, # this will default to grid layout 
        "chunk_ep_length": -1,#chunk the episode length
        "plane_urdf_path": 'urdf/plane/plane.urdf',
        'camera_kwargs': camera_kwargs,
        'render_camera': 'front',  
    } 
    return env_cfg 

class BaseEnv:
    """
    Support multiple different embodiments, and either bimanual or single hand
    """
    def __init__(
        self, 
        env_cfg,
        robot_cfgs, # dict of robots 
        object_cfgs, # dict of objects
        reward_cfg, 
        demo_data,
        retarget_data=dict(), # dict of joint_pos after retargeting
        rand_cfg=dict(),
        curriculum_cfg=dict(), 
        device=torch.device('cuda'),
        visualize_contact=False, 
        contact_marker_cfgs=dict(),
        group_collisions=False,
        render_figure=False,
        hide_cardbox=False, 
        postpone_build=False,
    ):
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.demo_data = demo_data
        self.curr_cfg = curriculum_cfg
        self.group_collisions = group_collisions

        # Grasp optimization controls must be defined before post_scene_build_setup
        self.rand_init_ratio = env_cfg.get('rand_init_ratio', 0.0)
        self.enforce_valid_rand_init_grasp = env_cfg.get('enforce_valid_rand_init_grasp', False)
        self.valid_grasp_min_contacts = env_cfg.get('valid_grasp_min_contacts', 1)
        self.valid_grasp_contact_thresh = env_cfg.get('valid_grasp_contact_thresh', 0.01)
        self.valid_grasp_max_contact_err_per_link = env_cfg.get('valid_grasp_max_contact_err_per_link', 0.08)
        self.valid_grasp_require_both_hands = env_cfg.get('valid_grasp_require_both_hands', False)
        self.valid_grasp_sample_iters = env_cfg.get('valid_grasp_sample_iters', 2)
        self.valid_grasp_sample_count = env_cfg.get('valid_grasp_sample_count', 32)
        self.valid_grasp_sample_std = env_cfg.get('valid_grasp_sample_std', 0.02)
        self.valid_grasp_close_bias = env_cfg.get('valid_grasp_close_bias', 0.01)
        self.valid_grasp_guided_gain = env_cfg.get('valid_grasp_guided_gain', 0.08)
        self.valid_grasp_guided_abd_gain = env_cfg.get('valid_grasp_guided_abd_gain', 0.03)
        self.valid_grasp_settle_steps = env_cfg.get('valid_grasp_settle_steps', 8)
        self.valid_grasp_slip_weight = env_cfg.get('valid_grasp_slip_weight', 2.0)
        self.valid_grasp_vel_weight = env_cfg.get('valid_grasp_vel_weight', 0.5)
        self.valid_grasp_ang_vel_weight = env_cfg.get('valid_grasp_ang_vel_weight', 0.25)
        self.valid_grasp_opposition_weight = env_cfg.get('valid_grasp_opposition_weight', 0.0)
        self.valid_grasp_functional_opposition_min = env_cfg.get('valid_grasp_functional_opposition_min', 0.15)
        self.valid_grasp_hold_max_drop = env_cfg.get('valid_grasp_hold_max_drop', -1.0)
        self.valid_grasp_hold_max_vel = env_cfg.get('valid_grasp_hold_max_vel', -1.0)
        self.valid_grasp_hold_max_ang_vel = env_cfg.get('valid_grasp_hold_max_ang_vel', -1.0)
        self.valid_grasp_hold_ignore_steps = env_cfg.get('valid_grasp_hold_ignore_steps', 0)
        self.valid_grasp_contact_persistence_min = env_cfg.get('valid_grasp_contact_persistence_min', 0.0)
        self.valid_grasp_compliance_steps = env_cfg.get('valid_grasp_compliance_steps', 50)
        self.valid_grasp_contact_bonus = env_cfg.get('valid_grasp_contact_bonus', 0.01)
        self.valid_grasp_refine_mode = env_cfg.get('valid_grasp_refine_mode', None)
        self.valid_grasp_use_ik = env_cfg.get('valid_grasp_use_ik', False)
        self.valid_grasp_ik_iters = env_cfg.get('valid_grasp_ik_iters', 4)
        self.valid_grasp_ik_damping = env_cfg.get('valid_grasp_ik_damping', 1e-3)
        self.valid_grasp_ik_eps = env_cfg.get('valid_grasp_ik_eps', 2e-3)
        self.valid_grasp_ik_step_clip = env_cfg.get('valid_grasp_ik_step_clip', 0.12)
        self.valid_grasp_ik_close_bias = env_cfg.get('valid_grasp_ik_close_bias', 0.02)
        self.valid_grasp_vf_close_bias = env_cfg.get('valid_grasp_vf_close_bias', 0.02)
        self.valid_grasp_vf_attract_steps = env_cfg.get('valid_grasp_vf_attract_steps', 20)
        self.valid_grasp_vf_hold_steps = env_cfg.get('valid_grasp_vf_hold_steps', 10)
        self.valid_grasp_vf_anneal_probe_hold_steps = env_cfg.get('valid_grasp_vf_anneal_probe_hold_steps', 5)
        self.valid_grasp_vf_squeeze_torque = env_cfg.get('valid_grasp_vf_squeeze_torque', 0.0)
        self.valid_grasp_vf_attract_gain = env_cfg.get('valid_grasp_vf_attract_gain', 0.5)
        self.valid_grasp_vf_attract_dq_clip = env_cfg.get('valid_grasp_vf_attract_dq_clip', 0.03)
        self.valid_grasp_vf_soft_unpin_steps = env_cfg.get('valid_grasp_vf_soft_unpin_steps', 50)
        self.valid_grasp_gravity_ramp_steps = env_cfg.get('valid_grasp_gravity_ramp_steps', 0)
        self.valid_grasp_fallback_to_zero = env_cfg.get('valid_grasp_fallback_to_zero', True)
        self.valid_grasp_debug = env_cfg.get('valid_grasp_debug', False)
        if self.valid_grasp_refine_mode is None:
            self.valid_grasp_refine_mode = 'ik' if self.valid_grasp_use_ik else 'sampling'
        self._warned_no_active_contact_finger = False
        self._last_settle_stats = dict()

        self.grasp_optimizer = None   # kept for backward compat checks
        self.grasp_optimizers = {}     # per-side optimizers
        if self.enforce_valid_rand_init_grasp:
            from graspqp.hands import AVAILABLE_HANDS
            for side in ['left', 'right']:
                if side not in robot_cfgs:
                    continue
                raw_hand_name = robot_cfgs[side].get('name', 'allegro')
                hand_name_with_side = f"{raw_hand_name}_{side}"
                if hand_name_with_side in AVAILABLE_HANDS:
                    hand_name = hand_name_with_side
                else:
                    hand_name = raw_hand_name
                self.grasp_optimizers[side] = GraspQPOptimizer(env=self, hand_name=hand_name, device=device)
            # Provide a backwards-compat single reference (first available side)
            if self.grasp_optimizers:
                self.grasp_optimizer = next(iter(self.grasp_optimizers.values()))

        # Latent world model config
        self.use_latent_world_model = env_cfg.get('use_latent_world_model', False)
        self.wm_latent_dim = env_cfg.get('wm_latent_dim', 32)

        self.num_envs = env_cfg['num_envs']
        self.max_video_frames = env_cfg['max_video_frames']
        self.record_video = env_cfg['record_video']
        self.render_segmentation = env_cfg.get('render_segmentation', False)
        self.max_episode_length = int(env_cfg['episode_length'])
        self.chunk_ep_length = env_cfg['chunk_ep_length']
        if self.chunk_ep_length > 0:
            print("Chunking episode length to ", self.chunk_ep_length)
            self.max_episode_length = self.chunk_ep_length

        self.action_clip = env_cfg['action_clip']
        self.action_scale = env_cfg['action_scale']
        self.obs_clip = env_cfg['obs_clip']
        self.dt = env_cfg['dt'] 
        self.early_reset_threshold = env_cfg['early_reset_threshold']
        self.early_reset_interval = int(env_cfg['early_reset_interval']) 
        self.early_reset_aux_thres = env_cfg.get('early_reset_aux_thres', dict())
        # if true, return obs dict insteaf of obs
        self.use_rl_games = env_cfg['use_rl_games']
        self.reward_module = RewardModule(
            reward_cfg, demo_data, retarget_data, device
            )

        self.reward_keys = self.reward_module.get_reward_keys()
        if self.num_envs > 3 and self.record_video:
            print("Warning: setting render env to 1 when there's more than 3 envs")
            env_cfg['scene_kwargs']['n_rendered_envs'] = 1
            self.max_video_frames = int(self.max_episode_length * 2)

        self.scene_cfg = get_scene_cfg(**env_cfg['scene_kwargs'])
        
        if self.group_collisions:
            print('Setting the SAME collision grouping to both hands')
            self.scene_cfg['rigid_options'].enable_self_collision = True
            self.scene_cfg['rigid_options'].self_collision_group_filter = True
            collision_groups = robot_cfgs['left'].get('collision_groups', dict())
            self.scene_cfg['rigid_options'].link_group_mapping = collision_groups
        if render_figure:
            print("Disabling gravity for figure rendering")
            self.scene_cfg['sim_options'].gravity = (0, 0, 0)
        self.demo_length = self.reward_module.get_demo_length()
        if self.chunk_ep_length <= 0:
            assert self.max_episode_length >= self.demo_length, f"Episode length {self.max_episode_length} != demo length {self.reward_module.demo_length}"
        
        self.is_finite_horizon = True # need this for rlgames
        
        self.table_height = TABLE_HEIGHT
        self.device = device 
        self.scene = gs.Scene(**self.scene_cfg)
        
        self.rigid_solver = None 
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver 
        
        # Determine global gravity scale for computing compensation force
        self.global_gravity = float(self.scene.sim.gravity[2])


        self.robot_cfgs = robot_cfgs
        self.robots = dict()
        self.retarget_data = retarget_data
        for k, cfg in robot_cfgs.items():
            self.robots[k] = BaseRobot(
                robot_cfg=cfg, 
                scene=self.scene,
                num_envs=self.num_envs,
                device=device,
                retarget_data=retarget_data.get(k, dict()),
                visualize_contact=visualize_contact,
                is_eval=env_cfg['is_eval'], 
                disable_collision=cfg.get('disable_collision', False),
                ) 
        self.robot_names = list(self.robots.keys())
        
        self.object_cfgs = object_cfgs
        # use retarget data to set base_init_pos, base_init_quat in obj_cfg
        self.objects = dict()
        for k, cfg in object_cfgs.items():
            # cfg['base_init_pos'] =  demo_data['obj_pos'][0]
            # cfg['base_init_quat'] = demo_data['obj_quat'][0]
            self.objects[k] = ArticulatedObject(
                cfg, 
                device=device,
                scene=self.scene,
                num_envs=self.num_envs,
                demo_data=demo_data,
                visualize_contact=visualize_contact, 
                disable_collision=cfg.get('disable_collision', False), # or render_figure,
                ) 
        
        self.object_names = list(self.objects.keys())
        self.object = None 
        if len(self.object_names) > 0: 
            self.object = self.objects[self.object_names[0]] # only support one object for now
        self.obj_verts = dict()
        self._setup_contact_link_metadata()
        self.n_objects = len(self.object_names) # might be 0!!
       
        self.use_curriculum = False 
        self.curriculum = None
        if self.n_objects == 1 and self.object.actuated:
            self.use_curriculum = True 
            self.curriculum = Curriculum(
                self.curr_cfg, 
                task_object=self.object,
                reward_keys=self.reward_keys,
                num_envs=self.num_envs,
                achieved_length=0,
                max_episode_length=self.max_episode_length,
            )        
        elif self.curr_cfg.get('type', None) == 'maniptrans':
            print("Using ManipTrans curriculum for ablation")
            self.use_curriculum = True
            self.curriculum = ManipTransCurriculum(
                self.curr_cfg, 
                task_object=self.object,
                reward_keys=self.reward_keys,
                num_envs=self.num_envs,
                achieved_length=0,
                max_episode_length=self.max_episode_length,
                sim=self.scene.sim,
                rigid_solver=self.rigid_solver,
            )
            
        # Object mass buffer for gravity compensation
        self.object_mass_buffer = None

        cardbox_size = (0.2,0.2,0.1)
        if self.n_objects == 1 and 'notebook' in self.object_names[0]:
            print("Adding a SMALLER cardboard box for notebook") 
            cardbox_size = (0.15, 0.15, 0.1) # wider: cardbox_size = (0.25, 0.2, 0.1)
        
        cardbox_surface = gs.surfaces.Rough(roughness=0.1, color=(167/255, 134/255, 103/255, 1.0))
        if env_cfg.get('texture_cardbox', False):
            cardbox_surface = gs.surfaces.Default( 
                diffuse_texture=gs.textures.ImageTexture(
                    image_path='/home/mandi/chiral/assets/wood.jpg',
                ),
            ) 
        self.cardboard_box = self.scene.add_entity(
            gs.morphs.Box(
                pos=CARDBOARD_POS,
                size=cardbox_size,
                fixed=True,
                visualization=(not hide_cardbox),
            ),
            surface=cardbox_surface,
        ) 
        
        self.contact_markers = dict()
        for name, marker_cfg in contact_marker_cfgs.items():
            num_vis_contacts = marker_cfg.get('num_vis_contacts', 0)
            color = marker_cfg.get('color', (1.0, 0.0, 0.0, 0.8))
            radius = marker_cfg.get('radius', 0.007)
            markers = [] 
            for _ in range(num_vis_contacts):
                marker = self.scene.add_entity(
                    gs.morphs.Sphere(
                        radius=radius, fixed=False, collision=False, # no collision
                        ),
                    surface=gs.surfaces.Smooth(color=color),
                )
                markers.append(marker)
            self.contact_markers[name] = markers
        
        plane_urdf_path = env_cfg.get('plane_urdf_path', 'urdf/plane/plane.urdf')
        self.ground = self.scene.add_entity(
            gs.morphs.URDF(file=plane_urdf_path, fixed=True)
        )

        self._recording = False
        self._recorded_frames = []
        
        self._floating_camera = None
        if self.record_video:
            assert gs.platform != 'macos', "Cannot render on macos"           
            self.render_camera = env_cfg.get('render_camera', 'front')
            self._add_camera(camera_kwargs=env_cfg['camera_kwargs'])

        self.env_cfg = env_cfg
        self.rand_cfg = rand_cfg
        if not postpone_build:
            self.build_scene()
            self.post_scene_build_setup()
        else:
            print("Scene created but not built yet") 
    
    def _setup_contact_link_metadata(self):
        """Map demo contact link ordering to robot link indices for each hand."""
        self.contact_link_meta = dict()
        self.thumb_indices = dict()
        self.finger_indices = dict()
        if len(self.robots) == 0:
            return
        for side, robot in self.robots.items():
            demo_side = self.demo_data.get(side, dict())
            link_names = demo_side.get('collision_link_names', [])
            if len(link_names) == 0:
                continue
            link_idxs = []
            thumb_link_idxs = []
            finger_link_idxs = []
            missing = []
            for name in link_names:
                idx = robot.link_name_to_local_idx.get(name)
                if idx is None:
                    missing.append(name)
                else:
                    link_idxs.append(int(idx))
                    lname = name.lower()
                    if "thumb" in lname:
                        thumb_link_idxs.append(int(idx))
                    else:
                        finger_link_idxs.append(int(idx))
            if len(link_idxs) == 0 or len(missing) > 0:
                if len(missing) > 0:
                    print(f"[valid-grasp] Missing collision links for {side}: {missing}")
                continue
            self.contact_link_meta[side] = dict(
                link_names=link_names,
                link_local_idxs=torch.tensor(link_idxs, dtype=torch.long, device=self.device),
                num_links=len(link_idxs),
            )
            if len(thumb_link_idxs) > 0:
                self.thumb_indices[side] = torch.tensor(
                    sorted(list(set(thumb_link_idxs))), dtype=torch.long, device=self.device
                )
            if len(finger_link_idxs) > 0:
                self.finger_indices[side] = torch.tensor(
                    sorted(list(set(finger_link_idxs))), dtype=torch.long, device=self.device
                )
            
    def build_scene(self):
        env_cfg = self.env_cfg
        self.scene.build(
            n_envs=self.num_envs, 
            env_spacing=env_cfg.get('env_spacing', ENV_SPACING),
            n_envs_per_row=env_cfg.get('n_envs_per_row', None),
            )

    def post_scene_build_setup(self):
        """ call this separately to customize the scene after env.init()"""
        env_cfg = self.env_cfg  
        rand_cfg = self.rand_cfg
        self.setup_actions(self.robots) 
        self.observe_tip_dist = env_cfg['observe_tip_dist']
        need_obj_surface_samples = self.observe_tip_dist or self.enforce_valid_rand_init_grasp
        if need_obj_surface_samples:
            assert self.n_objects == 1, "Only support one object for now"
            obj = self.objects[self.object_names[0]]
            self.obj_verts = {part: obj.sample_mesh_vertices(300, part) for part in ['top', 'bottom']}
            
        if self.n_objects > 0:
            link_masses = [link.get_mass() for link in self.object.entity.links]
            self.object_mass_buffer = torch.tensor(link_masses, device=self.device, dtype=torch.float32)
        
        self.observe_contact_force = env_cfg.get('observe_contact_force', False)
        if self.n_objects == 0:
            self.observe_contact_force = False
            print("Disabling contact force observation because no object")
        
        # Hand demo diff observation: measures deviation from demo trajectory
        self.observe_hand_demo_diff = env_cfg.get('observe_hand_demo_diff', False)
        if self.observe_hand_demo_diff:
            # Check that robots have residual_qpos set (demo trajectory)
            has_demo = all(robot.residual_qpos is not None for robot in self.robots.values())
            if not has_demo:
                print("Warning: observe_hand_demo_diff requires residual_qpos, disabling")
                self.observe_hand_demo_diff = False
            else:
                self.hand_demo_diff_dim = sum(robot.ndof for robot in self.robots.values())
                print(f"Enabling hand demo diff observation, dim={self.hand_demo_diff_dim}")
        
        # Trajectory lookahead observation: future K frames of demo trajectory
        self.traj_lookahead_frames = env_cfg.get('traj_lookahead_frames', 0)
        if self.traj_lookahead_frames > 0:
            has_demo = all(robot.residual_qpos is not None for robot in self.robots.values())
            if not has_demo:
                print("Warning: traj_lookahead requires residual_qpos, disabling")
                self.traj_lookahead_frames = 0
            else:
                # For each future frame, we observe the delta from current demo qpos
                # This gives relative motion rather than absolute positions
                ndof_total = sum(robot.ndof for robot in self.robots.values())
                self.traj_lookahead_dim = self.traj_lookahead_frames * ndof_total
                print(f"Enabling traj lookahead observation, K={self.traj_lookahead_frames}, dim={self.traj_lookahead_dim}")
        
        self.use_contact_reward = env_cfg.get('use_contact_reward', False) 
        if self.observe_contact_force or self.use_contact_reward:
            self.num_obj_links = len(self.object.coll_idxs_global)
            self.num_robot_links = sum([len(robot.coll_idxs_global) for robot in self.robots.values()])

            self.filter_links_a = torch.tensor(self.object.coll_idxs_global, device=self.device)
            self.filter_links_b = torch.tensor(
                self.robots['left'].coll_idxs_global + self.robots['right'].coll_idxs_global, 
                device=self.device
                )
            self.num_left_contact_links = len(self.robots['left'].coll_idxs_global)

            print("num_obj_links", self.num_obj_links) 
        

        self.obs_dim, self.obs_idxs = self.compute_obs_dim() 
        self.num_obs = self.obs_dim
        self.num_privileged_obs = None
        self.num_actions = self.action_dim # need for rsl
        self._demo_contact_cache = dict()
        self.initialize_value_buffers()
        
        # # NOTE! seems like scene must be built first
        for name, robot in self.robots.items():
            robot.post_scene_build_setup()
        
        for name, obj in self.objects.items():
            obj.post_scene_build_setup()
        
        if self.use_curriculum:
            self.curriculum.post_scene_build_setup()

        self.scene.step()
        self.extras = dict(log=dict())
        self.rew_dict = dict()
        self.obs_dict = dict()
        self.is_eval = env_cfg.get('is_eval', False) # if eval, have the option of skipping some env idxs for vis
        self._step_env_idxs = list(range(self.num_envs))
        if self.is_eval and self.num_envs > 1:
            self._step_env_idxs = self._step_env_idxs[:-1] # skip the LAST env for eval
        
        self.randomization = RandomizationModule(rand_cfg, self.rigid_solver, self.object, self.num_envs)
        if self.record_video: #  and self.num_envs > 2: 
            # NOTE: set the camera to only record the first env, must do this after the scene.build call
            offset = self.scene.rigid_solver.envs_offset.to_numpy()[0] # (3,)
            lookat_pos = offset + np.array([0, -0.1, 1.0])
            cam_pos = lookat_pos + np.array([0.0, -1.5, 1.2])
            self._set_camera(pos=cam_pos, lookat=lookat_pos, fov=30, name='front')


    def setup_actions(self, robots: Dict[str, BaseRobot]):
        action_dim = 0 
        idxs_to_robot = dict()
        for k, robot in robots.items():
            dim = robot.get_action_dim()
            idxs_to_robot[k] = [i for i in range(action_dim, action_dim+dim)]
            action_dim += dim
        self.action_dim = action_dim
        self.action_idxs_to_robot = idxs_to_robot
        # TODO: sim action latency as in Go2?
        # self.action_latency = 0 
    
    def update_contact_markers(self, contact_dict: Dict[str, torch.Tensor]):
        if len(self.contact_markers) == 0:
            return
        for name, contacts in contact_dict.items():
            if name not in self.contact_markers:
                continue
            markers = self.contact_markers[name]
            contacts = contacts.reshape((contacts.shape[0], -1, 3)) # squeeze middle dims
            # contacts shaped (num_envs, num_contacts, 6)
            num_vis = min(len(markers), contacts.shape[1])
            for i in range(num_vis):
                markers[i].set_pos(contacts[:, i, :3])
        
    def compute_obs_dim(self):
        
        obs_dim = 0
        obs_dim_info = dict()
        obs_idxs = dict()
        for k, robot in self.robots.items():
            dim, dim_info = robot.compute_obs_dim()
            obs_dim_info[k] = dim_info
            obs_idxs[k] = (obs_dim, obs_dim + dim)
            obs_dim += dim
        for k, obj in self.objects.items():
            dim, dim_info = obj.compute_obs_dim()
            obs_dim_info[k] = dim_info
            obs_idxs[k] = (obs_dim, obs_dim + dim)
            obs_dim += dim
        if self.observe_tip_dist:
            n_kpts = self.robots['left'].n_kpts + self.robots['right'].n_kpts
            obs_dim += n_kpts * 2 # because two obj parts!
        
        if self.observe_contact_force:
            obs_dim += self.num_obj_links * self.num_robot_links * 1 # 3 for force vec

        if self.observe_hand_demo_diff:
            obs_dim += self.hand_demo_diff_dim

        if self.traj_lookahead_frames > 0:
            obs_dim += self.traj_lookahead_dim

        obs_idxs['episode_length'] = (obs_dim, obs_dim + 1) 
        ep_len_dim = 1 #* 10
        obs_dim += ep_len_dim

        # Latent world model adds latent_dim to observation
        if self.use_latent_world_model:
            obs_idxs['latent'] = (obs_dim, obs_dim + self.wm_latent_dim)
            obs_dim += self.wm_latent_dim

        # return 20, obs_idxs
        return obs_dim, obs_idxs
    
    def initialize_value_buffers(self):
        # assume robots and objects have been initialized with buffers
        self.obs_buf = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_task_rew = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_con_rew = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_imi_rew = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_bc_rew = torch.zeros(self.num_envs, device=self.device)
        self.obs_sum = torch.zeros(self.num_envs, self.obs_dim, device=self.device)

        # match isaaclab
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.nan_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # NOTE this must be int dtype for indexing in match_demo_state
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.episode_start_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # this should be demo timestep

        if self.chunk_ep_length > 0:
            # each env idx has a different init timestep
            for i in range(self.num_envs):
                self.episode_start_buf[i] = i % self.chunk_ep_length
                self.episode_length_buf[i] = i % self.chunk_ep_length

        self.max_achieved_length = 0
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)

        # approximate dist from hand kpt to object surface
        num_obj_parts = 2 
        if self.observe_tip_dist:
            self.kpt_dists_left = torch.zeros((self.num_envs, self.robots['left'].n_kpts, num_obj_parts), device=self.device)
            self.kpt_dists_right = torch.zeros((self.num_envs, self.robots['right'].n_kpts, num_obj_parts), device=self.device)

        if self.observe_contact_force:
            self.contact_forces = torch.zeros((self.num_envs, self.num_obj_links,  self.num_robot_links, 3), device=self.device) 

        if self.use_contact_reward:
            self.contact_link_pos = torch.zeros((self.num_envs, self.num_obj_links, self.num_robot_links, 3), device=self.device)
            self.contact_link_valid = torch.zeros((self.num_envs, self.num_obj_links, self.num_robot_links), device=self.device, dtype=torch.bool)
        
        # Latent world model buffer
        if self.use_latent_world_model:
            self.latent_buf = torch.zeros((self.num_envs, self.wm_latent_dim), device=self.device)
        
        self.extras = dict() 
    # def progress_episode_length(self):
    #     self.episode_length_buf += 1
    #     for k, robot in self.robots.items():
    #         robot.episode_length_buf += 1
    #     for k, obj in self.objects.items():
    #         obj.episode_length_buf += 1
    def set_retarget_states(self, step, env_idxs=None):
        if env_idxs is None:
            env_idxs = list(range(self.num_envs))
        """directly set the state of the object and robots from retargeting data, use for visualization"""
        for k, robot in self.robots.items():
            if robot.residual_qpos is not None:
                robot.set_joint_position(
                    joint_targets=robot.residual_qpos[step][None].repeat(len(env_idxs), 1),
                    env_idxs=env_idxs,
                )

        if self.n_objects == 1:
            if self.object.demo_dofs is not None:
                targets = self.object.demo_dofs[step][None].repeat(len(env_idxs), 1) 
                self.object.entity.set_dofs_position(targets) 
        self.scene.step()
        self.episode_length_buf += 1 
        self._compute_intermediate_values()
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones() 
        self.reset_buf[:] = self.reset_terminated | self.reset_time_outs
        self._get_rewards()
        if self.record_video:
            self._render_headless()
        return  
    
    def pre_scene_step(self, actions: torch.Tensor):
        """ call this for only stepping the robot/objects"""
        self.last_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions, -self.action_clip, self.action_clip) * self.action_scale
        
        for k, robot in self.robots.items():
            idxs = self.action_idxs_to_robot[k]
            robot.step(self.actions[:, idxs], self._step_env_idxs)
        for k, obj in self.objects.items():
            obj.step() 
            
    def step(self, actions: torch.Tensor):
        """
        actions: torch.Tensor of shape (num_envs, action_dim)
        """
        assert actions.shape[1] == self.action_dim
        assert actions.shape[0] == self.num_envs
        self.last_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions, -self.action_clip, self.action_clip) * self.action_scale
        
        # For policy_residual mode, run base policy ONCE for all robots
        base_policy_actions = None
        for k, robot in self.robots.items():
            if robot.action_mode == "policy_residual":
                if base_policy_actions is None:
                    # Run base policy once and cache the result (exclude base_action from obs)
                    base_policy_obs = self.get_observations(include_base_action=False)
                    if self.use_rl_games:
                        base_policy_obs = base_policy_obs["policy"]
                    base_policy_actions = robot.get_base_policy_actions(base_policy_obs)
                break
        
        # Step each robot with appropriate actions
        for k, robot in self.robots.items():
            idxs = self.action_idxs_to_robot[k]
            if robot.action_mode == "policy_residual":
                # Extract this robot's portion of base policy actions
                robot_base_actions = base_policy_actions[:, idxs]
                robot.step(self.actions[:, idxs], self._step_env_idxs, obs=self.obs_buf, 
                          base_actions=robot_base_actions)
            else:
                robot.step(self.actions[:, idxs], self._step_env_idxs)
        for k, obj in self.objects.items():
            obj.step()
            
        # Apply object gravity compensation if curriculum is active
        if self.use_curriculum and self.curriculum is not None:
            curr_gains = self.curriculum.get_current_gains()
            if 'gravity' in curr_gains and curr_gains['gravity'] > 0.0 and self.n_objects == 1 and self.object_mass_buffer is not None:
                gravity_gain = curr_gains['gravity']
                # Create an upward force corresponding to the link masses and gravity gain
                comp_force = torch.zeros((self.num_envs, self.object.n_links, 3), device=self.device, dtype=torch.float32)
                # Note: global_gravity is negative (e.g., -9.81), so we subtract it
                comp_force[..., 2] = -self.global_gravity * self.object_mass_buffer * gravity_gain
                global_link_idxs = [link.idx for link in self.object.entity.links]
                self.rigid_solver.apply_links_external_force(comp_force, global_link_idxs)

        self.randomization.on_step(self.episode_length_buf)
        self.scene.step()  
        self.episode_length_buf += 1
        # self.progress_episode_length() 
        self._compute_intermediate_values()
        
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones() 
        self.reset_buf[:] = self.reset_terminated | self.reset_time_outs
 
        rew_dict = self._get_rewards()
        if "log" not in self.extras:
            self.extras["log"] = dict() 

        # maniptrans curriculum checks additional early resets
        if isinstance(self.curriculum, ManipTransCurriculum) and self.n_objects == 1 and "keypoint_dist" in rew_dict:
            # get the object position and rotation error from rew_dict   
            curriculm_reset = self.curriculum.determine_early_term(
                obj_pos_err=rew_dict['pos_dist'],
                obj_rot_err=rew_dict['rot_dist'],
                finger_pos_err=rew_dict['keypoint_dist'],
            ) 
            self.reset_buf[:] = self.reset_buf[:] | curriculm_reset

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # only log cum. episode reward if that env_idx is DONE 
            rew_dict['episode_rew'] = self.cumulative_task_rew[reset_env_ids] 
            self.reset_idx(reset_env_ids)  # rew and obs will be resetted   
        
        self.extras["log"].update(rew_dict) 
        if self.use_curriculum:
            rew_grads = self.curriculum.get_reward_grads()
            self.extras["log"].update(
                {f"grad/{k}": v for k, v in rew_grads.items()}
            )
            self.extras["log"].update(
                self.curriculum.get_current_gains()
                )
        # log contact forces
        if self.observe_contact_force:
            self.extras["log"]["contact_force"] = torch.norm(self.contact_forces, dim=-1).max().item()
        
        # get control_force 
        for side in ['left', 'right']:
            robot = self.robots[side]
            control_force = robot.get_control_force()
            self.extras["log"][f"{side}_control_force"] = control_force.mean().item()

        if self.record_video:
            self._render_headless()
        
        # update obs after potential reset_idx: 
        if self.use_rl_games:
            observations = self.get_observations()
            self.obs_buf[:] = observations['policy']
            return observations, self.rew_buf, self.reset_terminated, self.reset_time_outs, self.extras 
        else:
            self.obs_buf[:] = self.get_observations() # if resetted, the obs should be the initial ones
        self.obs_sum[:] += torch.abs(self.obs_buf)  
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
    
    def _get_rewards(self):
        if self.n_objects == 1:
            obj = self.objects[self.object_names[0]]
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
        else:
            obj_pos, obj_quat, obj_arti = None, None, None
        bc_dist = torch.cat([robot.get_bc_dist() for robot in self.robots.values()], dim=-1)
        reward_kwargs = dict(
            actions=self.actions,
            bc_dist=bc_dist,
            obj_pos=obj_pos,
            obj_quat=obj_quat,
            obj_arti=obj_arti,
            kpts_left=self.robots['left'].kpt_pos,
            kpts_right=self.robots['right'].kpt_pos,
            episode_length_buf=self.episode_length_buf,
            contact_link_pos_left=None,
            contact_link_valid_left=None,
            contact_link_pos_right=None,
            contact_link_valid_right=None,
            wrist_pose_left=self.robots['left'].wrist_pose,
            wrist_pose_right=self.robots['right'].wrist_pose,
            contact_forces=None,
            )
        if self.use_contact_reward:
            reward_kwargs.update(
                contact_link_pos_left=self.contact_link_pos[:, :, :self.num_left_contact_links], # N, 2, 13, 3
                contact_link_valid_left=self.contact_link_valid[:, :, :self.num_left_contact_links],
                contact_link_pos_right=self.contact_link_pos[:, :, self.num_left_contact_links:],
                contact_link_valid_right=self.contact_link_valid[:, :, self.num_left_contact_links:],
            )
        if self.observe_contact_force:
            reward_kwargs.update(
                contact_forces=self.contact_forces
            )
        rewards, rew_dict = self.reward_module.compute_reward(
            **reward_kwargs
        )
        
        if not self.use_rl_games:
            # scale the reward by 0.1 manually to match the scale in rl_games
            rewards *= 0.1

        self.rew_dict = rew_dict 
        self.rew_buf[:] = rewards
        # there's potentially nan values 
        self.rew_buf[self.nan_envs] = -1.0
        task_rewards = rew_dict['task_rew'] # use task_rew for reset
        task_rewards[self.nan_envs] = -1.0
        rew_dict['task_rew'] = task_rewards
        self.cumulative_task_rew[:] += task_rewards if self.n_objects == 1 else rewards 
        
        if 'con_rew' in rew_dict:
            con_rew = rew_dict['con_rew']
            con_rew[self.nan_envs] = -1.0
            self.cumulative_con_rew[:] += con_rew
        if 'imi_rew' in rew_dict:
            imi_rew = rew_dict['imi_rew']
            imi_rew[self.nan_envs] = -1.0
            self.cumulative_imi_rew[:] += imi_rew
        if 'bc_rew' in rew_dict:
            bc_rew = rew_dict['bc_rew']
            bc_rew[self.nan_envs] = -1.0
            self.cumulative_bc_rew[:] += bc_rew
        return rew_dict 

    def _get_dones(self):
        stepped_length = self.episode_length_buf - self.episode_start_buf
        if self.chunk_ep_length > 0:
            timeout = stepped_length >= self.chunk_ep_length
        else:
            timeout = self.episode_length_buf >= self.max_episode_length  # returns true for end of episode
 
        timeout = timeout | self.nan_envs
        if self.is_eval:
            return timeout, timeout # always let it run to last frame for eval
        
        object_fell_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        first_obj = None
        if self.n_objects == 1:
            first_obj = self.objects[self.object_names[0]]
            object_fell_off = first_obj.root_pos[:, 2] < self.table_height
        
        need_reset = timeout | object_fell_off | self.nan_envs
        
        if self.early_reset_threshold > 0.0:
            # early reset curriculum  
            for interval in range(0, self.max_episode_length, self.early_reset_interval):
                need_reset = torch.where(
                    (stepped_length > interval),
                    self.cumulative_task_rew < self.early_reset_threshold * interval,
                    need_reset
                ) 
        
        for key, cum_rew in zip(
            ['con', 'imi', 'bc'],
            [self.cumulative_con_rew, self.cumulative_imi_rew, self.cumulative_bc_rew]
        ):  
            thres = self.early_reset_aux_thres.get(key, 0.0)
            if thres > 0.0:
                for interval in range(0, self.max_episode_length, 20):   
                    need_reset = torch.where(
                        (stepped_length > interval), cum_rew < thres * interval, need_reset
                    )
        return need_reset, timeout
    
    def prepare_sliced_contact(self, source='policy', part='top', side='left'):
        """ this does not have to always return the same shape """
        if source == 'policy' and self.use_contact_reward:
            _pos = self.contact_link_pos 
            # set the invalid positions all to 0
            _pos[~self.contact_link_valid] = 0
            part_idx = 0 if part == 'top' else 1
            _pos = _pos[:, part_idx, :]
            if side == 'left':
                _pos = _pos[:, :self.num_left_contact_links]    
            else:
                _pos = _pos[:, self.num_left_contact_links:]

        else:        
            _pos = self.reward_module.match_demo_state(f'contact_links_{side}', self.episode_length_buf) # (N, 2*nlinks, 4) -> last dim is part id
            part_id = 2 if part == 'bottom' else 1
            if len(_pos.shape) == 4:
               # for retargeted contact, the shape is (N, 2, nlinks, 4) 
               _pos = _pos.reshape(self.num_envs, -1, 4)
            # only take positions that has the correct part id
            demo_part_valid = (_pos[:, :, -1] == part_id)# skip valid mask& demo_valid_contact
            # set invalid positions to 0
            _pos[~demo_part_valid] = 0
            _pos = _pos[:, :, :3]
        return _pos 
        
    def _compute_intermediate_values(self):
        for name, robot in self.robots.items():
            robot.update_value_buffers()
            self.nan_envs[:] = self.nan_envs | robot.get_nan_envs()
        for name, obj in self.objects.items():
            obj.update_value_buffers() 
            self.nan_envs[:] = self.nan_envs | obj.get_nan_envs()

        if self.observe_contact_force or self.use_contact_reward:
            entity_a = self.object.entity
            contact_info = get_filtered_contacts(
                    entity_a=entity_a, 
                    entity_b=None,
                    filter_links_a=self.filter_links_a,
                    filter_links_b=self.filter_links_b,
                    return_link_force=self.observe_contact_force,
                    return_link_pos=self.use_contact_reward,
                    device=self.device ,
                )
                
            if self.observe_contact_force:
                contact_force = contact_info["contact_force_link_a"] # shape (n_env, n_obj_links, n_robot_links, 3)
                self.contact_forces[:] = contact_force
            if self.use_contact_reward and self.num_obj_links > 0:
                self.contact_link_pos[:] = contact_info['contact_pos_link_a']
                self.contact_link_valid[:] = contact_info['contact_pos_link_a_valid']
                # vis left hand contact for now:
            
        contact_dict = dict()
        for key in self.contact_markers.keys():
            source, part, side = key.split('_')
            contact_dict[key] = self.prepare_sliced_contact(source, part, side) 
        self.update_contact_markers(contact_dict)
            
    def get_observations(self, include_base_action=True):
        value_list = []
        all_obs_dict = dict()
        for name, robot in self.robots.items():
            obs_dict = robot.get_observations(include_base_action=include_base_action)
            value_list.extend(list(obs_dict.values()))
            all_obs_dict[name] = obs_dict
        for name, obj in self.objects.items():
            obs_dict = obj.get_observations()
            value_list.extend(list(obs_dict.values()))
            all_obs_dict[name] = obs_dict
        left = self.robots['left']
        right = self.robots['right'] 
        # contact_info = left.entity.get_contacts(obj.entity)
        # force, mask = contact_info['force_a'], contact_info['valid_mask']
        # print(force[mask].shape)
        if self.chunk_ep_length > 0:
            normalize_ep_len = 2.0 * self.episode_length_buf[:, None].float() / self.demo_length - 1.0
        else:
            normalize_ep_len = 2.0 * self.episode_length_buf[:, None].float() / self.max_episode_length - 1.0
        value_list.append(normalize_ep_len)

        # value_list = []
        # value_list.append(normalize_ep_len.repeat(1, 20))

        if self.observe_tip_dist:
            assert self.n_objects == 1, "Only support one object for now"
            obj = self.objects[self.object_names[0]]
            # compute the kpt distances to the object surface
            for side, dists_tensor in zip(['left', 'right'], [self.kpt_dists_left, self.kpt_dists_right]):
                robot = self.robots[side]
                for i, part in enumerate(['top', 'bottom']):
                    part_pose = obj.get_part_pose(part)
                    dists_tensor[:, :, i] = self.compute_closest_vertice_dist_single(
                        self.obj_verts[part], robot.kpt_pos, part_pose
                    )
                    name = f"{side}_kpt_dist_{part}"
                    # print(name, np.round(dists_tensor[:, :, i].cpu().numpy(), 2))
            value_list.extend([
                self.kpt_dists_left.flatten(start_dim=1),
                self.kpt_dists_right.flatten(start_dim=1),
                ])

        if self.observe_contact_force:
            force_norm = torch.norm(self.contact_forces, dim=-1) * 0.01 # scale down! max contact force can go to 1000+
            value_list.append(force_norm.flatten(start_dim=1))

        if self.observe_hand_demo_diff:
            # Compute hand deviation from demo trajectory: demo_qpos - curr_qpos
            # This tells the policy "how far am I from where I should be"
            hand_demo_diffs = []
            for name, robot in self.robots.items():
                # curr_res_qpos is the demo qpos at current timestep (already computed in translate_actions)
                # dof_pos is the actual current joint positions
                demo_diff = robot.curr_res_qpos - robot.dof_pos  # shape (num_envs, ndof)
                hand_demo_diffs.append(demo_diff)
            value_list.append(torch.cat(hand_demo_diffs, dim=-1))

        if self.traj_lookahead_frames > 0:
            # Compute trajectory lookahead: future K frames of demo trajectory (as deltas from current)
            # This tells the policy "where should I be going next"
            lookahead_list = []
            for k in range(1, self.traj_lookahead_frames + 1):
                future_qpos_list = []
                for name, robot in self.robots.items():
                    # Compute future timestep, clamped to max frames
                    future_t = torch.clamp(
                        self.episode_length_buf + k,
                        max=robot.residual_num_frames - 1
                    )
                    future_qpos = robot.residual_qpos[future_t]  # shape (num_envs, ndof)
                    # Express as delta from current demo qpos (relative motion)
                    future_delta = future_qpos - robot.curr_res_qpos  # shape (num_envs, ndof)
                    future_qpos_list.append(future_delta)
                lookahead_list.append(torch.cat(future_qpos_list, dim=-1))
            value_list.append(torch.cat(lookahead_list, dim=-1))

        obs = torch.cat(value_list, dim=-1)
        self.obs_dict = all_obs_dict
        # if sum(torch.isnan(obs).flatten()) > 0:
        #     print("NAN OBSERVATIONS")
        #     breakpoint()
        nan_mask = torch.isnan(obs) 
        obs[nan_mask] = -self.obs_clip
        # if there's nan, need immediately reset 
        all_obs_dict['nan_mask'] = nan_mask
        obs = torch.clamp(obs, -self.obs_clip, self.obs_clip)

        # Append latent from world model if enabled
        if self.use_latent_world_model:
            obs = torch.cat([obs, self.latent_buf], dim=-1)
        
        if self.use_rl_games:
            return dict(policy=obs, itemized=all_obs_dict, critic=obs) # critic for sil
        return obs 
    
    def get_privileged_observations(self): 
        return None
    
    def get_object_state(self) -> torch.Tensor:
        """
        Get object state for world model decoder target.
        Returns: (num_envs, 14) tensor with [pos(3), quat(4), dof(1), lin_vel(3), ang_vel(3)]
        """
        if self.n_objects == 0:
            # Return zeros if no object
            return torch.zeros((self.num_envs, 14), device=self.device)
        
        obj = self.objects[self.object_names[0]]
        return torch.cat([
            obj.root_pos,      # 3D
            obj.root_quat,     # 4D
            obj.dof_pos,       # 1D
            obj.root_lin_vel,  # 3D
            obj.root_ang_vel,  # 3D
        ], dim=-1)
    
    def get_obs_without_latent(self) -> torch.Tensor:
        """
        Get observation without the latent component (for world model encoder input).
        Returns the full observation minus the latent_dim trailing dimensions.
        """
        if not self.use_latent_world_model:
            return self.obs_buf
        # Return obs without the trailing latent dimensions
        return self.obs_buf[:, :-self.wm_latent_dim]
    
    def update_latent(self, latent: torch.Tensor):
        """
        Update the latent buffer with new latent encoding.
        
        Args:
            latent: (num_envs, latent_dim) tensor from world model encoder
        """
        if self.use_latent_world_model:
            self.latent_buf[:] = latent

    def normalize_episode_rew(self, rewards: torch.Tensor):
        # rewards should be shape (num_envs,)
        avg_rew = rewards / self.max_episode_length
        return avg_rew.mean().item()

    def reset_idx(self, env_idxs=[]):
        if len(env_idxs) == 0:
            return  
        if isinstance(env_idxs, torch.Tensor):
            env_idxs_list = env_idxs.tolist()
        else:
            env_idxs_list = list(env_idxs)
        env_idxs_list = [int(idx) for idx in env_idxs_list]
        self.randomization.on_reset_idx(env_idxs)
        progressed = self.episode_length_buf[env_idxs] - self.episode_start_buf[env_idxs]
        progressed_avg = torch.mean(progressed.float()).item()
        self.max_achieved_length = int(self.max_achieved_length * 0.5 + progressed_avg * 0.5)

        if self.use_curriculum:
            episode_rewards = dict()
            if 'task' in self.reward_keys:
                episode_rewards['task'] = self.normalize_episode_rew(self.cumulative_task_rew[env_idxs])
            if 'con' in self.reward_keys:
                episode_rewards['con'] = self.normalize_episode_rew(self.cumulative_con_rew[env_idxs])
            if 'imi' in self.reward_keys:
                episode_rewards['imi'] = self.normalize_episode_rew(self.cumulative_imi_rew[env_idxs])
            if 'bc' in self.reward_keys:
                episode_rewards['bc'] = self.normalize_episode_rew(self.cumulative_bc_rew[env_idxs])
            self.curriculum.update_progress(episode_rewards, self.max_achieved_length)

        if self.chunk_ep_length > 0:
            self.episode_start_buf[env_idxs] = (env_idxs % self.chunk_ep_length).to(torch.int32).to(self.device)
            self.episode_length_buf[env_idxs] = self.episode_start_buf[env_idxs]
        else:
            self.episode_length_buf[env_idxs] = 0
            self.episode_start_buf[env_idxs] = 0 
        
        rand_init_mask = None
        if self.rand_init_ratio > 0.0:
            # randomly sample non-zero initial t 
            # num_rand = int(self.rand_init_ratio * len(env_idxs)) + 1
            # treat this as probability 
            torand = torch.rand(len(env_idxs)) <= self.rand_init_ratio
            # randomly sample from any t within max_episode_length
            # end_t = min(self.max_achieved_length + 1, self.max_episode_length - 1)
            end_t = min(100, self.max_episode_length - 1) # TODO: hardcoded for now
            rand_t = torch.randint(0, end_t, (len(env_idxs),), dtype=torch.int32, device=self.device)
            ep_starts = torch.zeros(len(env_idxs), dtype=torch.int32, device=self.device)
            ep_starts[torand] = rand_t[torand] 
            self.episode_length_buf[env_idxs] = ep_starts
            self.episode_start_buf[env_idxs] = ep_starts
            rand_init_mask = torand.clone()
            
        for k, robot in self.robots.items():
            # need to rand init too
            robot.reset_idx(env_idxs, self.episode_start_buf[env_idxs])
        
        for k, obj in self.objects.items():
            obj.reset_idx(env_idxs, self.episode_start_buf[env_idxs])

        if rand_init_mask is not None:
            self._ensure_valid_rand_init_grasp(env_idxs_list, rand_init_mask)
            self._zero_object_velocity_envs(env_idxs_list)

        self.last_actions[env_idxs] = 0.0
        
        self.reset_buf[env_idxs] = True
        self.reset_terminated[env_idxs] = True
        self.reset_time_outs[env_idxs] = True

        # self.rew_buf[env_idxs] = 0.0
        self.cumulative_task_rew[env_idxs] = 0.0
        self.cumulative_con_rew[env_idxs] = 0.0
        self.cumulative_imi_rew[env_idxs] = 0.0
        self.cumulative_bc_rew[env_idxs] = 0.0
        # self.obs_buf[env_idxs] = 0.0
        self.obs_sum[env_idxs] = 0.0   

        # self._compute_intermediate_values()
        if self.observe_tip_dist:
            self.kpt_dists_left[env_idxs] = 0.0
            self.kpt_dists_right[env_idxs] = 0.0
        
        if self.observe_contact_force:
            self.contact_forces[env_idxs] = 0.0
        
        if self.use_contact_reward:
            self.contact_link_pos[env_idxs] = 0.0
            self.contact_link_valid[env_idxs] = False
        
        # Reset latent buffer for world model
        if self.use_latent_world_model:
            self.latent_buf[env_idxs] = 0.0  

    def _zero_object_velocity_envs(self, env_idxs):
        """Hard reset object velocities for selected envs."""
        if self.object is None:
            return
        if isinstance(env_idxs, torch.Tensor):
            env_idxs = env_idxs.tolist()
        env_idxs = [int(i) for i in env_idxs]
        if len(env_idxs) == 0:
            return
        self.object.entity.zero_all_dofs_velocity(envs_idx=env_idxs)
        self.object.root_lin_vel[env_idxs, :] = 0.0
        self.object.root_ang_vel[env_idxs, :] = 0.0
        self.object.dof_vel[env_idxs, :] = 0.0

    def _ensure_valid_rand_init_grasp(self, env_idxs_list, rand_mask):
        """Adjust invalid random initializations via contact-aware sampling."""
        if (
            not self.enforce_valid_rand_init_grasp
            or self.rand_init_ratio <= 0.0
            or self.n_objects == 0
            or len(env_idxs_list) == 0
            or rand_mask is None
        ):
            return
        if isinstance(rand_mask, torch.Tensor):
            rand_flags = rand_mask.to(device='cpu').tolist()
        else:
            rand_flags = list(rand_mask)
        if len(rand_flags) == 0 or not any(rand_flags):
            return
        candidate_envs = [
            int(env_idxs_list[i])
            for i in range(min(len(env_idxs_list), len(rand_flags)))
            if rand_flags[i]
        ]
        if len(candidate_envs) == 0:
            return
        targets_map = {
            env_idx: self._get_demo_contact_targets_for_env(env_idx)
            for env_idx in candidate_envs
        }
        invalid_envs = []
        for env_idx in candidate_envs:
            avg_err = self._evaluate_grasp_quality(env_idx, targets_map[env_idx])
            if avg_err is None:
                continue
            if avg_err > self.valid_grasp_contact_thresh:
                invalid_envs.append(env_idx)
        if not invalid_envs:
            self._zero_object_velocity_envs(candidate_envs)
            return
        if self.valid_grasp_debug:
            print(f"[valid-grasp] invalid envs={invalid_envs} from candidates={candidate_envs}")
        
        
        
        for env_idx in invalid_envs:
            targets = targets_map.get(env_idx, dict())
            success = self._optimize_grasp_state(env_idx, targets)
            if not success and self.valid_grasp_fallback_to_zero:
                if self.valid_grasp_debug:
                    print(f"[valid-grasp] env={env_idx} optimize failed, fallback_to_zero=True")
                self._reset_env_to_demo_anchor(env_idx)
        # if self.grasp_optimizers:
        #     import random as _random
        #     # Identify which sides have contact for each invalid env.
        #     # If both sides are in contact in a given env, pick one at random.
        #     # The idle side is reset to its demo pose.
        #     for env_idx in invalid_envs:
        #         active_sides = list(self._get_active_contact_fingers(targets_map[env_idx]).keys())
        #         if not active_sides:
        #             # No contact detected – skip this env
        #             continue
        #         if len(active_sides) == 1:
        #             optimize_side = active_sides[0]
        #         else:
        #             # Both hands in contact – GraspQP cannot do multi-hand; pick one at random
        #             optimize_side = _random.choice(active_sides)
        #             idle_side = [s for s in active_sides if s != optimize_side][0]
        #             idle_robot = self.robots.get(idle_side)
        #             if idle_robot is not None:
        #                 # Reset idle hand to demo pose
        #                 idle_tensor = torch.tensor([env_idx], dtype=torch.long, device=self.device)
        #                 idle_robot.reset_idx(idle_tensor, self.episode_start_buf[idle_tensor])

        #         optimizer = self.grasp_optimizers.get(optimize_side)
        #         robot = self.robots.get(optimize_side)
        #         if optimizer is not None and robot is not None:
        #             iters = self.env_cfg.get('valid_grasp_ik_iters', 50)
        #             with torch.enable_grad():
        #                 new_pos, new_quat, new_qpos = optimizer.optimize_grasps(
        #                     robot, [env_idx], iters=iters
        #                 )
        #             env_tensor = torch.tensor([env_idx], dtype=torch.long, device=self.device)
        #             robot.entity.set_pos(new_pos, envs_idx=env_tensor)
        #             robot.entity.set_quat(new_quat, envs_idx=env_tensor)
        #             robot.entity.set_dofs_position(
        #                 new_qpos,
        #                 dofs_idx_local=robot.actuated_dof_idxs,
        #                 envs_idx=env_tensor,
        #             )
        self._zero_object_velocity_envs(candidate_envs)

    def _get_demo_contact_targets_for_env(self, env_idx: int):
        targets = dict()
        if len(self.contact_link_meta) == 0:
            return targets
        env_idx = int(env_idx)
        if env_idx < 0 or env_idx >= self.num_envs:
            return targets
        env_episode = self.episode_start_buf[env_idx:env_idx+1]
        for side, meta in self.contact_link_meta.items():
            key = f"contact_links_{side}"
            if key not in self.reward_module.demo_tensors:
                continue
            demo_state = self.reward_module.match_demo_state(key, env_episode)
            if demo_state.shape[1] == meta['num_links'] * 2:
                reshaped = demo_state[0].reshape(2, meta['num_links'], -1)
            elif demo_state.dim() == 4 and demo_state.shape[2] == meta['num_links']:
                reshaped = demo_state[0]
            else:
                continue
            positions = reshaped[:, :, :3]
            part_ids = reshaped[:, :, 3]
            valid = part_ids > 0
            targets[side] = dict(
                positions=positions,
                valid=valid,
            )
        return targets

    def _evaluate_grasp_quality(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        if len(targets) == 0:
            return self._keypoint_grasp_distance(env_idx)
        error_sum, count, _ = self._contact_alignment_error(env_idx, targets)
        if count == 0:
            return self._keypoint_grasp_distance(env_idx)
        if count < self.valid_grasp_min_contacts:
            return float('inf')
        return error_sum / max(count, 1)

    def _keypoint_grasp_distance(self, env_idx: int):
        if self.n_objects == 0 or len(self.obj_verts) == 0:
            return None
        obj = self.object
        env_idx = int(env_idx)
        part_poses = dict()
        for part in ['top', 'bottom']:
            if part in self.obj_verts:
                part_poses[part] = obj.get_part_pose(part)
        if len(part_poses) == 0:
            return None
        total = 0.0
        total_count = 0
        for side, robot in self.robots.items():
            dists = []
            for part, pose in part_poses.items():
                verts = self.obj_verts.get(part, None)
                if verts is None:
                    continue
                part_dist = self.compute_closest_vertice_dist_single(verts, robot.kpt_pos, pose)
                dists.append(part_dist[env_idx])
            if len(dists) == 0:
                continue
            stacked = torch.stack(dists, dim=0)
            min_dists = torch.min(stacked, dim=0).values
            total += min_dists.sum().item()
            total_count += min_dists.numel()
        if total_count == 0:
            return None
        return total / total_count

    def _contact_alignment_error(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        total_error = 0.0
        total_count = 0
        side_counts = dict()
        for side, target in targets.items():
            meta = self.contact_link_meta.get(side, None)
            robot = self.robots.get(side, None)
            if meta is None or robot is None:
                continue
            link_idxs = meta['link_local_idxs']
            link_pos = robot.entity.get_links_pos()
            env_link_pos = link_pos[env_idx, link_idxs]
            pos_expanded = env_link_pos.unsqueeze(0) # (1, num_links, 3)
            diff = target['positions'] - pos_expanded
            dists = torch.norm(diff, dim=-1)
            mask = target['valid']
            count = int(mask.sum().item())
            side_counts[side] = count
            if count > 0:
                total_error += dists[mask].sum().item()
                total_count += count
        return total_error, total_count, side_counts

    def _get_active_contact_fingers(self, targets: Dict[str, Dict[str, torch.Tensor]]):
        active = dict()
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for side, target in targets.items():
            meta = self.contact_link_meta.get(side, None)
            robot = self.robots.get(side, None)
            if meta is None or robot is None:
                continue
            link_names = meta.get('link_names', [])
            link_idxs = meta.get('link_local_idxs', None)
            if len(link_names) == 0:
                continue
            valid_any = target['valid']
            active_fingers = set()
            collision_groups = robot.get_collision_groups()
            group_to_finger = {
                1: 'thumb',
                2: 'index',
                3: 'middle',
                4: 'ring',
                5: 'pinky',
            }
            for link_i, link_name in enumerate(link_names):
                if not valid_any[:, link_i].any().item():
                    continue
                matched = False
                for finger in finger_names:
                    if finger in link_name:
                        active_fingers.add(finger)
                        matched = True
                        break
                if matched:
                    continue
                if link_idxs is not None and link_i < len(link_idxs):
                    local_idx = int(link_idxs[link_i].item())
                    group_id = collision_groups.get(local_idx, None)
                    finger = group_to_finger.get(group_id, None)
                    if finger is not None:
                        active_fingers.add(finger)
            if len(active_fingers) > 0:
                active[side] = list(sorted(active_fingers))
        return active

    def _compute_finger_contact_dirs(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        dirs = dict()
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for side, target in targets.items():
            meta = self.contact_link_meta.get(side, None)
            robot = self.robots.get(side, None)
            if meta is None or robot is None:
                continue
            link_names = meta.get('link_names', [])
            link_idxs = meta.get('link_local_idxs', None)
            if len(link_names) == 0 or link_idxs is None or len(link_names) != len(link_idxs):
                continue
            env_link_pos = robot.entity.get_links_pos()[env_idx, link_idxs]
            side_dirs = {name: torch.zeros(3, device=self.device) for name in finger_names}
            side_counts = {name: 0 for name in finger_names}
            for part_idx in range(target['positions'].shape[0]):
                valid_mask = target['valid'][part_idx]
                if valid_mask.sum() == 0:
                    continue
                diffs = target['positions'][part_idx] - env_link_pos
                for link_i, link_name in enumerate(link_names):
                    if not valid_mask[link_i]:
                        continue
                    finger = None
                    for f in finger_names:
                        if f in link_name:
                            finger = f
                            break
                    if finger is None:
                        continue
                    side_dirs[finger] += diffs[link_i]
                    side_counts[finger] += 1
            for finger in finger_names:
                if side_counts[finger] > 0:
                    side_dirs[finger] = side_dirs[finger] / float(side_counts[finger])
            dirs[side] = side_dirs
        return dirs

    def _map_link_to_finger(self, robot, link_name: str, local_idx: int):
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger in link_name:
                return finger
        group_to_finger = {1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'}
        group_id = robot.get_collision_groups().get(int(local_idx), None)
        return group_to_finger.get(group_id, None)

    def _build_finger_contact_problem(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        problem = dict()
        for side, target in targets.items():
            meta = self.contact_link_meta.get(side, None)
            robot = self.robots.get(side, None)
            if meta is None or robot is None:
                continue
            link_names = meta.get('link_names', [])
            link_idxs = meta.get('link_local_idxs', None)
            if len(link_names) == 0 or link_idxs is None:
                continue
            finger_groups = robot.get_finger_joint_groups()
            side_problem = dict()
            for link_i, link_name in enumerate(link_names):
                valid = target['valid'][:, link_i]
                if not valid.any().item():
                    continue
                tgt_pos = target['positions'][:, link_i, :][valid].mean(dim=0)
                local_idx = int(link_idxs[link_i].item())
                finger = self._map_link_to_finger(robot, link_name, local_idx)
                if finger is None:
                    continue
                joint_idxs = finger_groups.get(finger, [])
                if len(joint_idxs) == 0:
                    continue
                if finger not in side_problem:
                    side_problem[finger] = dict(
                        joint_idxs=joint_idxs,
                        link_local_idxs=[],
                        target_positions=[],
                    )
                side_problem[finger]['link_local_idxs'].append(local_idx)
                side_problem[finger]['target_positions'].append(tgt_pos)
            if len(side_problem) > 0:
                problem[side] = side_problem
        return problem

    def _build_side_contact_problem(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        problem = dict()
        for side, target in targets.items():
            meta = self.contact_link_meta.get(side, None)
            robot = self.robots.get(side, None)
            if meta is None or robot is None:
                continue
            link_names = meta.get('link_names', [])
            link_idxs = meta.get('link_local_idxs', None)
            if len(link_names) == 0 or link_idxs is None:
                continue
            finger_groups = robot.get_finger_joint_groups()
            link_local_idxs = []
            target_positions = []
            active_fingers = set()
            for link_i, link_name in enumerate(link_names):
                valid = target['valid'][:, link_i]
                if not valid.any().item():
                    continue
                local_idx = int(link_idxs[link_i].item())
                tgt_pos = target['positions'][:, link_i, :][valid].mean(dim=0)
                link_local_idxs.append(local_idx)
                target_positions.append(tgt_pos)
                # finger mapping only needed for joint selection, not for IK targets
                finger = self._map_link_to_finger(robot, link_name, local_idx)
                if finger is not None:
                    active_fingers.add(finger)
            if len(link_local_idxs) == 0:
                continue
            joint_set = set()
            for finger in active_fingers:
                for j in finger_groups.get(finger, []):
                    joint_set.add(int(j))
            # Include wrist joints (forearm tx/ty/tz + roll/pitch/yaw) so IK can
            # reposition the hand globally, not just flex individual fingers.
            for j in robot.wrist_dof_idxs:
                joint_set.add(int(j))
            if len(joint_set) == 0:
                continue
            problem[side] = dict(
                joint_idxs=sorted(list(joint_set)),
                link_local_idxs=link_local_idxs,
                target_positions=torch.stack(target_positions, dim=0),
                active_fingers=active_fingers,
            )
        return problem

    def _numeric_finger_jacobian(self, env_idx: int, robot, joint_targets: torch.Tensor, joint_idxs, link_local_idxs):
        env_idx = int(env_idx)
        eps = float(self.valid_grasp_ik_eps)
        robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
        base = robot.entity.get_links_pos()[env_idx, link_local_idxs]
        n_res = base.numel()
        n_dof = len(joint_idxs)
        J = torch.zeros((n_res, n_dof), device=self.device)
        for col, dof_idx in enumerate(joint_idxs):
            perturbed = joint_targets.clone()
            perturbed[dof_idx] += eps
            lower = robot.dof_limits[:, 0]
            upper = robot.dof_limits[:, 1]
            perturbed = torch.clamp(perturbed, lower, upper)
            robot.set_joint_position(perturbed[None], env_idxs=[env_idx])
            pos = robot.entity.get_links_pos()[env_idx, link_local_idxs]
            J[:, col] = ((pos - base) / eps).reshape(-1)
        robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
        return J, base

    def _numeric_contact_jacobian(self, env_idx: int, robot, joint_targets: torch.Tensor, joint_idxs, link_local_idxs):
        env_idx = int(env_idx)
        eps = float(self.valid_grasp_ik_eps)
        robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
        base = robot.entity.get_links_pos()[env_idx, link_local_idxs]
        n_res = base.numel()
        n_dof = len(joint_idxs)
        J = torch.zeros((n_res, n_dof), device=self.device)
        for col, dof_idx in enumerate(joint_idxs):
            perturbed = joint_targets.clone()
            perturbed[dof_idx] += eps
            lower = robot.dof_limits[:, 0]
            upper = robot.dof_limits[:, 1]
            perturbed = torch.clamp(perturbed, lower, upper)
            robot.set_joint_position(perturbed[None], env_idxs=[env_idx])
            pos = robot.entity.get_links_pos()[env_idx, link_local_idxs]
            J[:, col] = ((pos - base) / eps).reshape(-1)
        robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
        return J, base

    def _solve_contact_ik_lm_for_env(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        env_idx = int(env_idx)
        problem = self._build_side_contact_problem(env_idx, targets)
        if len(problem) == 0:
            if self.valid_grasp_debug:
                print(f"[valid-grasp] env={env_idx} LM IK skipped: no side contact problem")
            return False
        changed = False
        for side, side_problem in problem.items():
            robot = self.robots.get(side, None)
            if robot is None:
                continue
            joint_idxs = side_problem['joint_idxs']
            link_local_idxs = side_problem['link_local_idxs']
            target_positions = side_problem['target_positions']
            q = robot.dof_pos[env_idx].clone()
            init_res = None
            for it in range(max(1, int(self.valid_grasp_ik_iters))):
                J, curr_pos = self._numeric_contact_jacobian(env_idx, robot, q, joint_idxs, link_local_idxs)
                residual = (target_positions - curr_pos).reshape(-1)
                res_norm = torch.norm(residual).item()
                if init_res is None:
                    init_res = res_norm
                if res_norm < 1e-4:
                    break
                damping = float(self.valid_grasp_ik_damping)
                A = J.T @ J + damping * torch.eye(J.shape[1], device=self.device)
                b = J.T @ residual
                dq = torch.linalg.solve(A, b)
                dq = torch.clamp(dq, -self.valid_grasp_ik_step_clip, self.valid_grasp_ik_step_clip)
                if torch.norm(dq).item() < 1e-6:
                    break
                q[joint_idxs] += dq
                lower = robot.dof_limits[:, 0]
                upper = robot.dof_limits[:, 1]
                q = torch.clamp(q, lower, upper)
                changed = True
            if self.valid_grasp_debug:
                J, curr_pos = self._numeric_contact_jacobian(env_idx, robot, q, joint_idxs, link_local_idxs)
                final_res = torch.norm((target_positions - curr_pos).reshape(-1)).item()
                print(
                    f"[LM-IK] env={env_idx} {side}: iters={it+1} "
                    f"res {init_res:.4f}->{final_res:.4f} n_joints={len(joint_idxs)}"
                )
            if self.valid_grasp_ik_close_bias > 0:
                name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
                for dof_idx in joint_idxs:
                    lname = name_by_idx.get(dof_idx, "").lower()
                    if "abd" in lname or "spread" in lname or "forearm" in lname:
                        continue
                    q[dof_idx] += self.valid_grasp_ik_close_bias
                lower = robot.dof_limits[:, 0]
                upper = robot.dof_limits[:, 1]
                q = torch.clamp(q, lower, upper)
            robot.set_joint_position(q[None], env_idxs=[env_idx])
        return changed

    def _solve_contact_ik_for_env(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        env_idx = int(env_idx)
        problem = self._build_finger_contact_problem(env_idx, targets)
        if len(problem) == 0:
            if self.valid_grasp_debug:
                print(f"[valid-grasp] env={env_idx} IK skipped: no finger contact problem")
            return False
        changed = False
        for side, side_problem in problem.items():
            robot = self.robots.get(side, None)
            if robot is None:
                continue
            for finger, data in side_problem.items():
                joint_idxs = data['joint_idxs']
                link_local_idxs = data['link_local_idxs']
                target_positions = torch.stack(data['target_positions'], dim=0)
                q = robot.dof_pos[env_idx].clone()
                for _ in range(max(1, int(self.valid_grasp_ik_iters))):
                    J, curr_pos = self._numeric_finger_jacobian(env_idx, robot, q, joint_idxs, link_local_idxs)
                    residual = (target_positions - curr_pos).reshape(-1)
                    if torch.norm(residual).item() < 1e-4:
                        break
                    damping = float(self.valid_grasp_ik_damping)
                    A = J.T @ J + damping * torch.eye(J.shape[1], device=self.device)
                    b = J.T @ residual
                    dq = torch.linalg.solve(A, b)
                    dq = torch.clamp(dq, -self.valid_grasp_ik_step_clip, self.valid_grasp_ik_step_clip)
                    if torch.norm(dq).item() < 1e-6:
                        break
                    q[joint_idxs] += dq
                    lower = robot.dof_limits[:, 0]
                    upper = robot.dof_limits[:, 1]
                    q = torch.clamp(q, lower, upper)
                    changed = True
                if self.valid_grasp_ik_close_bias > 0:
                    name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
                    for dof_idx in joint_idxs:
                        lname = name_by_idx.get(dof_idx, "").lower()
                        if "abd" in lname or "spread" in lname:
                            continue
                        q[dof_idx] += self.valid_grasp_ik_close_bias
                    lower = robot.dof_limits[:, 0]
                    upper = robot.dof_limits[:, 1]
                    q = torch.clamp(q, lower, upper)
                robot.set_joint_position(q[None], env_idxs=[env_idx])
        return changed

    def _apply_sampled_finger_perturbation(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        env_idx = int(env_idx)
        active = self._get_active_contact_fingers(targets)
        if len(active) == 0:
            if not self._warned_no_active_contact_finger:
                self._warned_no_active_contact_finger = True
                print("[valid-grasp] no active contact fingers identified; skipping refinement sample")
            return False
        finger_dirs = self._compute_finger_contact_dirs(env_idx, targets)
        for side, fingers in active.items():
            robot = self.robots.get(side, None)
            if robot is None:
                continue
            finger_groups = robot.get_finger_joint_groups()
            name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
            joint_targets = robot.dof_pos[env_idx].clone()
            for finger in fingers:
                joint_idxs = finger_groups.get(finger, [])
                if len(joint_idxs) == 0:
                    continue
                noise = torch.randn(len(joint_idxs), device=self.device) * self.valid_grasp_sample_std
                bias = torch.zeros_like(noise)
                contact_dir = finger_dirs.get(side, {}).get(finger, torch.zeros(3, device=self.device))
                guided_flex = self.valid_grasp_guided_gain * float(contact_dir[2].item())
                guided_abd = self.valid_grasp_guided_abd_gain * float(contact_dir[0].item())
                if self.valid_grasp_close_bias > 0:
                    for i, joint_idx in enumerate(joint_idxs):
                        jname = name_by_idx.get(joint_idx, "")
                        lname = jname.lower()
                        if "abd" in lname or "spread" in lname:
                            bias[i] = guided_abd
                            continue
                        bias[i] = self.valid_grasp_close_bias + guided_flex
                joint_targets[joint_idxs] += noise + bias
            lower = robot.dof_limits[:, 0]
            upper = robot.dof_limits[:, 1]
            joint_targets = torch.clamp(joint_targets, lower, upper)
            robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
        return True

    def _apply_two_sided_finger_probing(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]], init_obj_z: float):
        """Try +/- guided perturbations per active finger and keep local improvements."""
        env_idx = int(env_idx)
        active = self._get_active_contact_fingers(targets)
        if len(active) == 0:
            return False
        finger_dirs = self._compute_finger_contact_dirs(env_idx, targets)
        working_state = self._snapshot_env_state(env_idx)
        working_score, _, _ = self._score_candidate(env_idx, targets, init_obj_z)
        changed = False

        for side, fingers in active.items():
            robot = self.robots.get(side, None)
            if robot is None:
                continue
            finger_groups = robot.get_finger_joint_groups()
            name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
            side_dirs = finger_dirs.get(side, {})
            for finger in fingers:
                joint_idxs = finger_groups.get(finger, [])
                if len(joint_idxs) == 0:
                    continue
                direction = side_dirs.get(finger, torch.zeros(3, device=self.device))
                probe_step = torch.zeros(len(joint_idxs), device=self.device)
                for i, joint_idx in enumerate(joint_idxs):
                    lname = name_by_idx.get(joint_idx, "").lower()
                    if "abd" in lname or "spread" in lname:
                        base = self.valid_grasp_guided_abd_gain * float(direction[0].item())
                    else:
                        base = self.valid_grasp_close_bias + self.valid_grasp_guided_gain * float(direction[2].item())
                    probe_step[i] = max(abs(base), self.valid_grasp_sample_std)

                best_local_score = working_score
                best_local_state = working_state
                for sign in (1.0, -1.0):
                    self._restore_env_state(env_idx, working_state)
                    joint_targets = robot.dof_pos[env_idx].clone()
                    noise = torch.randn(len(joint_idxs), device=self.device) * (0.25 * self.valid_grasp_sample_std)
                    delta = sign * probe_step + noise
                    joint_targets[joint_idxs] += delta
                    lower = robot.dof_limits[:, 0]
                    upper = robot.dof_limits[:, 1]
                    joint_targets = torch.clamp(joint_targets, lower, upper)
                    robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
                    candidate_score, _, _ = self._score_candidate(env_idx, targets, init_obj_z)
                    if candidate_score < best_local_score:
                        best_local_score = candidate_score
                        best_local_state = self._snapshot_env_state(env_idx)

                if best_local_score < working_score:
                    working_score = best_local_score
                    working_state = best_local_state
                    self._restore_env_state(env_idx, working_state)
                    changed = True

        self._restore_env_state(env_idx, working_state)
        return changed

    def _object_root_link_global_idx(self):
        if self.object is None:
            return None
        if len(self.object.entity.links) == 0:
            return None
        return int(self.object.entity.links[0].idx)

    def _apply_gravity_compensation_force(self, env_idx: int, alpha: float):
        """Apply upward external force to smooth object unpinning hand-off."""
        if self.object is None or self.rigid_solver is None:
            return
        if not hasattr(self.rigid_solver, "apply_links_external_force"):
            return
        gravity_vec = torch.tensor(self.scene_cfg["sim_options"].gravity, dtype=torch.float32, device=self.device)
        gravity_norm = float(torch.norm(gravity_vec).item())
        if gravity_norm <= 1e-6:
            return
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha >= 1.0:
            return
        root_link = self._object_root_link_global_idx()
        if root_link is None:
            return
        mass = float(self.object.entity.get_mass())
        # Apply (1-alpha) * (-m*g) as an external force in world frame.
        comp_force = (-gravity_vec * mass * (1.0 - alpha)).reshape(1, 1, 3)
        links_idx = torch.tensor([root_link], dtype=torch.long, device=self.device)
        envs_idx = torch.tensor([int(env_idx)], dtype=torch.long, device=self.device)
        self.rigid_solver.apply_links_external_force(comp_force, links_idx, envs_idx=envs_idx)

    def _quat_error_rotvec(self, target_quat: torch.Tensor, curr_quat: torch.Tensor):
        """Quaternion error as axis-angle rotation vector in world frame."""
        q_err = quat_mul(target_quat[None], quat_conjugate(curr_quat[None]))[0]
        if q_err[0].item() < 0.0:
            q_err = -q_err
        vec = q_err[1:]
        sin_half = torch.norm(vec)
        if sin_half.item() < 1e-6:
            return 2.0 * vec
        w = torch.clamp(q_err[0], -1.0, 1.0)
        angle = 2.0 * torch.atan2(sin_half, w)
        axis = vec / sin_half
        return axis * angle

    def _apply_compliance_wrench(self, env_idx: int, anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, stiffness_scale: float):
        """Apply a decaying 6D virtual spring to soften the unpin transition."""
        if self.object is None or self.rigid_solver is None:
            return
        if stiffness_scale <= 0.0:
            return
        if not hasattr(self.rigid_solver, "apply_links_external_force"):
            return
        if not hasattr(self.rigid_solver, "apply_links_external_torque"):
            return
        root_link = self._object_root_link_global_idx()
        if root_link is None:
            return
        mass = float(self.object.entity.get_mass())
        pos = self.object.root_pos[env_idx]
        quat = self.object.root_quat[env_idx]
        lin_vel = self.object.root_lin_vel[env_idx]
        ang_vel = self.object.root_ang_vel[env_idx]
        pos_err = anchor_pos - pos
        rot_err = self._quat_error_rotvec(anchor_quat, quat)

        k_pos = float(stiffness_scale) * (150.0 * mass)
        d_pos = float(stiffness_scale) * (2.0 * np.sqrt(max(k_pos * mass, 1e-6)))
        k_rot = float(stiffness_scale) * (35.0 * mass)
        d_rot = float(stiffness_scale) * (2.0 * np.sqrt(max(k_rot * mass, 1e-6)))

        force = k_pos * pos_err - d_pos * lin_vel
        torque = k_rot * rot_err - d_rot * ang_vel

        max_force = 200.0 * mass
        max_torque = 50.0 * mass
        force_norm = torch.norm(force).item()
        torque_norm = torch.norm(torque).item()
        if force_norm > max_force and force_norm > 1e-6:
            force = force * (max_force / force_norm)
        if torque_norm > max_torque and torque_norm > 1e-6:
            torque = torque * (max_torque / torque_norm)

        links_idx = torch.tensor([root_link], dtype=torch.long, device=self.device)
        envs_idx = torch.tensor([int(env_idx)], dtype=torch.long, device=self.device)
        self.rigid_solver.apply_links_external_force(
            force.reshape(1, 1, 3), links_idx, envs_idx=envs_idx
        )
        self.rigid_solver.apply_links_external_torque(
            torque.reshape(1, 1, 3), links_idx, envs_idx=envs_idx
        )

    def _simulate_settle(
        self,
        env_idx: int,
        hold_targets: Dict[str, torch.Tensor],
        steps: int,
        expected_contact_links_by_side: Dict[str, set] = None,
    ):
        if steps <= 0:
            return None, None, None
        env_idx = int(env_idx)
        # Ensure stability probe starts from zero object velocity.
        self._zero_object_velocity_envs([env_idx])
        ignore_steps = max(0, int(self.valid_grasp_hold_ignore_steps))
        gravity_ramp_steps = max(0, int(self.valid_grasp_gravity_ramp_steps))
        compliance_steps = max(0, int(self.valid_grasp_compliance_steps))
        expected_contact_links_by_side = expected_contact_links_by_side if isinstance(expected_contact_links_by_side, dict) else None
        compliance_anchor_pos = None
        compliance_anchor_quat = None
        if self.object is not None and compliance_steps > 0:
            compliance_anchor_pos = self.object.root_pos[env_idx].clone()
            compliance_anchor_quat = self.object.root_quat[env_idx].clone()
        min_z = None
        max_vel = None
        max_ang_vel = None
        persistence_ratio = 1.0
        has_persistence_measurement = False
        for step_i in range(steps):
            for side, robot in self.robots.items():
                target = hold_targets.get(side, None)
                if target is None:
                    continue
                robot.control_joint_position(target[None], env_idxs=[env_idx])
            for _, obj in self.objects.items():
                obj.step()
            if compliance_steps > 0 and step_i < compliance_steps and compliance_anchor_pos is not None:
                if compliance_steps == 1:
                    stiffness_scale = 1.0
                else:
                    stiffness_scale = max(0.0, 1.0 - float(step_i) / float(compliance_steps - 1))
                self._apply_compliance_wrench(
                    env_idx, compliance_anchor_pos, compliance_anchor_quat, stiffness_scale
                )
            elif gravity_ramp_steps > 0:
                if gravity_ramp_steps == 1:
                    alpha = 0.0
                else:
                    alpha = min(float(step_i) / float(gravity_ramp_steps - 1), 1.0)
                self._apply_gravity_compensation_force(env_idx, alpha=alpha)
            self.scene.step()
            if self.object is not None:
                self.object.update_value_buffers()
                curr_z = float(self.object.root_pos[env_idx, 2].item())
                vel = float(torch.norm(self.object.root_lin_vel[env_idx]).item())
                ang_vel = float(torch.norm(self.object.root_ang_vel[env_idx]).item())
                if step_i >= ignore_steps:
                    if min_z is None or curr_z < min_z:
                        min_z = curr_z
                    if max_vel is None or vel > max_vel:
                        max_vel = vel
                    if max_ang_vel is None or ang_vel > max_ang_vel:
                        max_ang_vel = ang_vel
                    if expected_contact_links_by_side is not None and len(expected_contact_links_by_side) > 0:
                        touched = self._get_contact_link_touches_by_side(env_idx)
                        curr_ratio = self._contact_persistence_ratio(touched, expected_contact_links_by_side)
                        persistence_ratio = min(persistence_ratio, curr_ratio)
                        has_persistence_measurement = True
        self._compute_intermediate_values()
        if min_z is None and self.object is not None:
            min_z = float(self.object.root_pos[env_idx, 2].item())
        if max_vel is None and self.object is not None:
            max_vel = float(torch.norm(self.object.root_lin_vel[env_idx]).item())
        if max_ang_vel is None and self.object is not None:
            max_ang_vel = float(torch.norm(self.object.root_ang_vel[env_idx]).item())
        if expected_contact_links_by_side is not None and len(expected_contact_links_by_side) > 0 and not has_persistence_measurement:
            touched = self._get_contact_link_touches_by_side(env_idx)
            persistence_ratio = self._contact_persistence_ratio(touched, expected_contact_links_by_side)
        persistence_ok = (
            self.valid_grasp_contact_persistence_min <= 0.0
            or persistence_ratio >= float(self.valid_grasp_contact_persistence_min)
        )
        self._last_settle_stats[env_idx] = dict(
            ignore_steps=int(ignore_steps),
            gravity_ramp_steps=int(gravity_ramp_steps),
            compliance_steps=int(compliance_steps),
            persistence_ratio=float(persistence_ratio),
            persistence_ok=bool(persistence_ok),
        )
        return min_z, max_vel, max_ang_vel

    def _get_raw_world_contact_data(self, env_idx: int):
        """Return raw world contact tensors for one env."""
        env_idx = int(env_idx)
        try:
            n_contacts = int(self.rigid_solver.collider.n_contacts.to_torch(device=self.device)[env_idx].item())
        except Exception:
            return None
        if n_contacts <= 0:
            return None
        contact_data = self.rigid_solver.collider.contact_data
        return dict(
            n_contacts=n_contacts,
            link_a=contact_data.link_a.to_torch(device=self.device)[:n_contacts, env_idx].to(dtype=torch.long),
            link_b=contact_data.link_b.to_torch(device=self.device)[:n_contacts, env_idx].to(dtype=torch.long),
            normal=contact_data.normal.to_torch(device=self.device)[:n_contacts, env_idx],
            force=contact_data.force.to_torch(device=self.device)[:n_contacts, env_idx],
            penetration=contact_data.penetration.to_torch(device=self.device)[:n_contacts, env_idx],
        )

    def _object_global_link_idxs(self):
        """Global link ids for the tracked object (use all object links, not only coll_idxs)."""
        if self.object is None:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        ids = [int(link.idx) for link in self.object.entity.links]
        if len(ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _local_to_global_link_idxs(self, robot, local_idxs: torch.Tensor):
        """Convert robot-local link indices to solver-global link indices robustly."""
        if local_idxs is None or len(local_idxs) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        ids = []
        for li in local_idxs.to(dtype=torch.long, device='cpu').tolist():
            li = int(li)
            if li < 0 or li >= len(robot.entity.links):
                continue
            ids.append(int(robot.entity.links[li].idx))
        if len(ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _meta_to_global_link_idxs(self, side: str, meta: Dict, valid_mask: torch.Tensor = None):
        """Resolve contact meta links to solver-global ids, preferring link names over saved idxs."""
        robot = self.robots.get(side, None)
        if robot is None or meta is None:
            return torch.zeros((0,), dtype=torch.long, device=self.device)

        ids = []
        link_names = meta.get("link_names", [])
        if len(link_names) > 0:
            for i, name in enumerate(link_names):
                if valid_mask is not None:
                    if i >= valid_mask.shape[0] or not bool(valid_mask[i].item()):
                        continue
                local_idx = robot.link_name_to_local_idx.get(name, None)
                if local_idx is None:
                    continue
                if local_idx < 0 or local_idx >= len(robot.entity.links):
                    continue
                ids.append(int(robot.entity.links[int(local_idx)].idx))
        else:
            local_idxs = meta.get("link_local_idxs", None)
            if local_idxs is not None and len(local_idxs) > 0:
                if valid_mask is not None and valid_mask.shape[0] == local_idxs.shape[0]:
                    local_idxs = local_idxs[valid_mask]
                return self._local_to_global_link_idxs(robot, local_idxs)

        if len(ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _raw_contact_stats_for_sets(self, raw_contact, obj_link_idxs: torch.Tensor, hand_link_idxs: torch.Tensor):
        """Count unique object-hand link pairs and touched hand links from raw contact_data."""
        if raw_contact is None or hand_link_idxs is None or hand_link_idxs.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=self.device)
            return 0, 0, empty
        link_a = raw_contact["link_a"]
        link_b = raw_contact["link_b"]
        obj_link_idxs = obj_link_idxs.to(dtype=torch.long, device=self.device)
        hand_link_idxs = hand_link_idxs.to(dtype=torch.long, device=self.device)
        mask_ab = torch.isin(link_a, obj_link_idxs) & torch.isin(link_b, hand_link_idxs)
        mask_ba = torch.isin(link_b, obj_link_idxs) & torch.isin(link_a, hand_link_idxs)
        obj_ids = torch.cat([link_a[mask_ab], link_b[mask_ba]], dim=0)
        hand_ids = torch.cat([link_b[mask_ab], link_a[mask_ba]], dim=0)
        if obj_ids.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=self.device)
            return 0, 0, empty
        uniq_pairs = torch.unique(torch.stack([obj_ids, hand_ids], dim=-1), dim=0)
        touched_hand = torch.unique(hand_ids)
        return int(uniq_pairs.shape[0]), int(touched_hand.shape[0]), touched_hand

    def _raw_contact_normals_for_sets(self, raw_contact, obj_link_idxs: torch.Tensor, hand_link_idxs: torch.Tensor):
        """Return object->hand oriented contact normals for selected object/hand links."""
        if raw_contact is None or hand_link_idxs is None or hand_link_idxs.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        link_a = raw_contact["link_a"]
        link_b = raw_contact["link_b"]
        normal = raw_contact["normal"]
        obj_link_idxs = obj_link_idxs.to(dtype=torch.long, device=self.device)
        hand_link_idxs = hand_link_idxs.to(dtype=torch.long, device=self.device)
        mask_ab = torch.isin(link_a, obj_link_idxs) & torch.isin(link_b, hand_link_idxs)
        mask_ba = torch.isin(link_b, obj_link_idxs) & torch.isin(link_a, hand_link_idxs)
        if not mask_ab.any() and not mask_ba.any():
            return torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        # Collider normal is geom_a->geom_b. Re-orient to object->hand for both mask directions.
        normals_ab = normal[mask_ab]
        normals_ba = -normal[mask_ba]
        normals = torch.cat([normals_ab, normals_ba], dim=0)
        return normals

    def _count_actual_contact_pairs(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]] = None):
        """Count actual object-hand contact pairs from raw world contacts for tracked links."""
        if self.object is None or len(self.contact_link_meta) == 0:
            return 0
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        if raw_contact is None:
            return 0
        obj_link_idxs = self._object_global_link_idxs()
        total = 0
        for side, meta in self.contact_link_meta.items():
            valid_mask = None
            if targets is not None and side in targets:
                valid_mask = targets[side]["valid"].any(dim=0)
            global_link_idxs = self._meta_to_global_link_idxs(side, meta, valid_mask=valid_mask)
            n_pairs, _, _ = self._raw_contact_stats_for_sets(raw_contact, obj_link_idxs, global_link_idxs)
            total += n_pairs
        return int(total)

    def _count_actual_contact_pairs_by_side(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]] = None):
        """Count actual object-hand contact pairs per side from raw world contacts."""
        if self.object is None or len(self.contact_link_meta) == 0:
            return {}
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        if raw_contact is None:
            return {side: 0 for side in self.contact_link_meta.keys()}
        obj_link_idxs = self._object_global_link_idxs()
        side_pairs = {}
        for side, meta in self.contact_link_meta.items():
            valid_mask = None
            if targets is not None and side in targets:
                valid_mask = targets[side]["valid"].any(dim=0)
            global_link_idxs = self._meta_to_global_link_idxs(side, meta, valid_mask=valid_mask)
            n_pairs, _, _ = self._raw_contact_stats_for_sets(raw_contact, obj_link_idxs, global_link_idxs)
            side_pairs[side] = int(n_pairs)
        return side_pairs

    def _get_contact_link_touches_by_side(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]] = None):
        """Return touched hand link ids per side from raw object-hand contacts."""
        if self.object is None or len(self.contact_link_meta) == 0:
            return {}
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        if raw_contact is None:
            return {side: set() for side in self.contact_link_meta.keys()}
        obj_link_idxs = self._object_global_link_idxs()
        touches = {}
        for side, meta in self.contact_link_meta.items():
            valid_mask = None
            if targets is not None and side in targets:
                valid_mask = targets[side]["valid"].any(dim=0)
            global_link_idxs = self._meta_to_global_link_idxs(side, meta, valid_mask=valid_mask)
            _, _, touched = self._raw_contact_stats_for_sets(raw_contact, obj_link_idxs, global_link_idxs)
            touches[side] = set([int(v) for v in touched.detach().cpu().tolist()])
        return touches

    def _contact_persistence_ratio(self, touched_by_side: Dict[str, set], expected_by_side: Dict[str, set]):
        """Ratio of expected touched links that remain touched (min over sides with expectations)."""
        if not isinstance(expected_by_side, dict) or len(expected_by_side) == 0:
            return 1.0
        ratios = []
        for side, expected in expected_by_side.items():
            if not isinstance(expected, set) or len(expected) == 0:
                continue
            touched = touched_by_side.get(side, set()) if isinstance(touched_by_side, dict) else set()
            if len(touched) == 0:
                ratios.append(0.0)
                continue
            overlap = len(expected.intersection(touched))
            ratios.append(float(overlap) / float(max(len(expected), 1)))
        if len(ratios) == 0:
            return 1.0
        return float(min(ratios))

    def _compute_thumb_finger_opposition(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]] = None):
        """Functional opposition from contact normals: thumb mean vs non-thumb mean."""
        if self.object is None or len(self.contact_link_meta) == 0:
            return dict(score=0.0, dot=1.0, thumb_contacts=0, finger_contacts=0)
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        if raw_contact is None:
            return dict(score=0.0, dot=1.0, thumb_contacts=0, finger_contacts=0)
        obj_link_idxs = self._object_global_link_idxs()
        thumb_normals = []
        finger_normals = []
        for side, robot in self.robots.items():
            thumb_local = self.thumb_indices.get(
                side, torch.zeros((0,), dtype=torch.long, device=self.device)
            )
            finger_local = self.finger_indices.get(
                side, torch.zeros((0,), dtype=torch.long, device=self.device)
            )
            if targets is not None and side in targets and side in self.contact_link_meta:
                valid_mask = targets[side]["valid"].any(dim=0)
                meta_local = self.contact_link_meta[side]["link_local_idxs"]
                if valid_mask.shape[0] == meta_local.shape[0]:
                    valid_local = set(meta_local[valid_mask].detach().cpu().tolist())
                    if len(valid_local) > 0:
                        thumb_local = torch.tensor(
                            [int(i) for i in thumb_local.detach().cpu().tolist() if int(i) in valid_local],
                            dtype=torch.long,
                            device=self.device,
                        )
                        finger_local = torch.tensor(
                            [int(i) for i in finger_local.detach().cpu().tolist() if int(i) in valid_local],
                            dtype=torch.long,
                            device=self.device,
                        )
            thumb_global = self._local_to_global_link_idxs(robot, thumb_local)
            finger_global = self._local_to_global_link_idxs(robot, finger_local)
            n_thumb = self._raw_contact_normals_for_sets(raw_contact, obj_link_idxs, thumb_global)
            n_finger = self._raw_contact_normals_for_sets(raw_contact, obj_link_idxs, finger_global)
            if n_thumb.numel() > 0:
                thumb_normals.append(n_thumb)
            if n_finger.numel() > 0:
                finger_normals.append(n_finger)
        if len(thumb_normals) == 0 or len(finger_normals) == 0:
            thumb_cnt = int(sum([v.shape[0] for v in thumb_normals])) if len(thumb_normals) > 0 else 0
            finger_cnt = int(sum([v.shape[0] for v in finger_normals])) if len(finger_normals) > 0 else 0
            return dict(score=0.0, dot=1.0, thumb_contacts=thumb_cnt, finger_contacts=finger_cnt)
        thumb = torch.cat(thumb_normals, dim=0)
        finger = torch.cat(finger_normals, dim=0)
        thumb_unit = thumb / torch.norm(thumb, dim=-1, keepdim=True).clamp(min=1e-6)
        finger_unit = finger / torch.norm(finger, dim=-1, keepdim=True).clamp(min=1e-6)
        thumb_mean = thumb_unit.mean(dim=0)
        finger_mean = finger_unit.mean(dim=0)
        thumb_mean = thumb_mean / torch.norm(thumb_mean).clamp(min=1e-6)
        finger_mean = finger_mean / torch.norm(finger_mean).clamp(min=1e-6)
        dot = float(torch.clamp(torch.dot(thumb_mean, finger_mean), -1.0, 1.0).item())
        score = 0.5 * (1.0 - dot)  # 0 when aligned, 1 when perfectly opposing.
        return dict(
            score=float(score),
            dot=dot,
            thumb_contacts=int(thumb.shape[0]),
            finger_contacts=int(finger.shape[0]),
        )

    def _passes_functional_opposition(self, opposition_metrics: Dict):
        """Functional opposition gate: require thumb and finger contacts with sufficient opposition."""
        min_score = float(self.valid_grasp_functional_opposition_min)
        if min_score <= 0.0:
            return True
        if not isinstance(opposition_metrics, dict):
            return False
        if int(opposition_metrics.get("thumb_contacts", 0)) <= 0:
            return False
        if int(opposition_metrics.get("finger_contacts", 0)) <= 0:
            return False
        return float(opposition_metrics.get("score", 0.0)) >= min_score

    def _compute_contact_normal_metrics(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]] = None):
        """Compute force-closure proxies from contact normals: opposition and directional diversity."""
        if self.object is None or len(self.contact_link_meta) == 0:
            return dict(opposition_norm=1.0, diversity=0.0, n_contacts=0)
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        if raw_contact is None:
            return dict(opposition_norm=1.0, diversity=0.0, n_contacts=0)
        obj_link_idxs = self._object_global_link_idxs()
        normals = []
        for side, meta in self.contact_link_meta.items():
            valid_mask = None
            if targets is not None and side in targets:
                valid_mask = targets[side]["valid"].any(dim=0)
            global_link_idxs = self._meta_to_global_link_idxs(side, meta, valid_mask=valid_mask)
            side_normals = self._raw_contact_normals_for_sets(raw_contact, obj_link_idxs, global_link_idxs)
            if side_normals.numel() > 0:
                normals.append(side_normals)
        if len(normals) == 0:
            return dict(opposition_norm=1.0, diversity=0.0, n_contacts=0)
        normals = torch.cat(normals, dim=0)
        n_contacts = int(normals.shape[0])
        norms = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-6)
        unit = normals / norms
        mean_vec = unit.mean(dim=0)
        opposition_norm = float(torch.norm(mean_vec).item())
        cov = unit.T @ unit / max(float(n_contacts), 1.0)
        eigvals = torch.linalg.eigvalsh(cov).real
        lambda_max = float(torch.max(eigvals).item())
        diversity = float(max(0.0, 1.0 - lambda_max))
        return dict(opposition_norm=opposition_norm, diversity=diversity, n_contacts=n_contacts)

    def _passes_contact_gate(self, side_pairs: Dict[str, int]):
        """Contact-count gate: allow single-hand grasps if one side has enough contacts."""
        return True
        if not isinstance(side_pairs, dict):
            return True
        min_contacts = int(self.valid_grasp_min_contacts)
        if min_contacts <= 0:
            return True
        return max([int(v) for v in side_pairs.values()] + [0]) >= min_contacts

    def _debug_world_contact_pairs(self, env_idx: int, prefix: str = "[VF-close-diag]", max_pairs: int = 10):
        """Print top world contact pairs (by force norm) with link names."""
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        if raw_contact is None:
            print(f"{prefix} env={env_idx}: cannot read world contacts")
            return
        n_contacts = int(raw_contact["n_contacts"])
        link_a = raw_contact["link_a"]
        link_b = raw_contact["link_b"]
        force = raw_contact["force"]
        penetration = raw_contact["penetration"]
        force_norm = torch.norm(force, dim=-1)
        topk = min(int(max_pairs), n_contacts)
        order = torch.argsort(force_norm, descending=True)[:topk]

        link_name_map = {}
        for obj_name, obj in self.objects.items():
            for link in obj.entity.links:
                link_name_map[int(link.idx)] = f"obj:{obj_name}/{link.name}"
        for side, robot in self.robots.items():
            for link in robot.entity.links:
                link_name_map[int(link.idx)] = f"robot:{side}/{link.name}"

        def _name(link_idx: int):
            return link_name_map.get(link_idx, f"other/link#{link_idx}")

        print(f"{prefix} env={env_idx}: top world contact pairs by |force| (showing {topk}/{n_contacts})")
        for rank, idx in enumerate(order.tolist()):
            la = int(link_a[idx].item())
            lb = int(link_b[idx].item())
            fn = float(force_norm[idx].item())
            pen = float(penetration[idx].item())
            print(
                f"{prefix} env={env_idx} pair[{rank}] "
                f"{_name(la)}(id={la}) <-> {_name(lb)}(id={lb}) |force|={fn:.4f} penetration={pen:.6f}"
            )

    def _debug_contact_diagnostics(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]] = None, prefix: str = "[VF-close-diag]"):
        """Print why tracked contact pairs are missing (debug-only utility)."""
        if self.object is None or len(self.contact_link_meta) == 0:
            print(f"{prefix} env={env_idx}: no object or contact_link_meta")
            return
        env_idx = int(env_idx)
        raw_contact = self._get_raw_world_contact_data(env_idx)
        world_n_contacts = int(raw_contact["n_contacts"]) if raw_contact is not None else 0
        if world_n_contacts > 0:
            self._debug_world_contact_pairs(env_idx, prefix=prefix, max_pairs=8)
        obj_link_idxs = self._object_global_link_idxs()
        for side, meta in self.contact_link_meta.items():
            robot = self.robots.get(side, None)
            if robot is None:
                continue
            local_idxs = meta.get("link_local_idxs", None)
            if local_idxs is None or len(local_idxs) == 0:
                print(f"{prefix} env={env_idx} side={side}: no tracked links")
                continue
            valid_mask = None
            if targets is not None and side in targets:
                valid_mask = targets[side]["valid"].any(dim=0)
            tracked_global_all = self._meta_to_global_link_idxs(side, meta, valid_mask=None)
            tracked_global = self._meta_to_global_link_idxs(side, meta, valid_mask=valid_mask)
            all_global = torch.tensor(robot.coll_idxs_global, device=self.device, dtype=torch.long)
            tracked_pairs, _, tracked_touched_ids = self._raw_contact_stats_for_sets(
                raw_contact, obj_link_idxs, tracked_global
            )
            all_pairs, _, all_touched_ids = self._raw_contact_stats_for_sets(
                raw_contact, obj_link_idxs, all_global
            )
            tracked_touched = int(torch.isin(tracked_global, tracked_touched_ids).sum().item()) if tracked_global.numel() > 0 else 0
            all_touched = int(torch.isin(all_global, all_touched_ids).sum().item()) if all_global.numel() > 0 else 0

            target_valid_links = 0
            missed_names = []
            tracked_surface_mean_mm = None
            tracked_surface_min_mm = None
            all_surface_mean_mm = None
            all_surface_min_mm = None
            if len(self.obj_verts) > 0:
                # Approximate signed distance proxy: min Euclidean distance to sampled object surface verts.
                tracked_link_pos = robot.entity.get_links_pos()[env_idx:env_idx+1, local_idxs]
                all_local_idxs = torch.tensor(robot.coll_idxs_local, device=self.device, dtype=torch.long)
                all_link_pos = robot.entity.get_links_pos()[env_idx:env_idx+1, all_local_idxs]
                tracked_part_dists = []
                all_part_dists = []
                for part in ["top", "bottom"]:
                    verts = self.obj_verts.get(part, None)
                    if verts is None:
                        continue
                    part_pose = self.object.get_part_pose(part)[env_idx:env_idx+1]
                    tracked_d = self.compute_closest_vertice_dist_single(verts, tracked_link_pos, part_pose)[0]
                    all_d = self.compute_closest_vertice_dist_single(verts, all_link_pos, part_pose)[0]
                    tracked_part_dists.append(tracked_d)
                    all_part_dists.append(all_d)
                if len(tracked_part_dists) > 0:
                    tracked_min_d = torch.stack(tracked_part_dists, dim=0).min(dim=0).values
                    all_min_d = torch.stack(all_part_dists, dim=0).min(dim=0).values
                    tracked_surface_mean_mm = float(tracked_min_d.mean().item() * 1000.0)
                    tracked_surface_min_mm = float(tracked_min_d.min().item() * 1000.0)
                    all_surface_mean_mm = float(all_min_d.mean().item() * 1000.0)
                    all_surface_min_mm = float(all_min_d.min().item() * 1000.0)
            if targets is not None and side in targets:
                valid_mask = targets[side]["valid"].any(dim=0)
                target_valid_links = int(valid_mask.sum().item())
                tracked_link_touch_all = torch.isin(tracked_global_all, tracked_touched_ids)
                if tracked_link_touch_all.shape[0] == valid_mask.shape[0]:
                    tracked_link_touch = tracked_link_touch_all
                    missed = torch.logical_and(valid_mask, torch.logical_not(tracked_link_touch))
                else:
                    missed = torch.zeros_like(valid_mask, dtype=torch.bool)
                link_names = meta.get("link_names", [])
                missed_idxs = missed.nonzero(as_tuple=False).flatten().tolist()
                missed_names = [link_names[i] for i in missed_idxs[:6] if i < len(link_names)]

            print(
                f"{prefix} env={env_idx} side={side}: "
                f"world_contacts={world_n_contacts} "
                f"tracked_pairs={tracked_pairs} all_pairs={all_pairs} "
                f"tracked_links_touch={tracked_touched}/{tracked_global.shape[0]} "
                f"all_links_touch={all_touched}/{all_global.shape[0]} "
                f"target_valid_links={target_valid_links} "
                f"missed_target_links={missed_names} "
                f"tracked_surface_mm(mean/min)="
                f"{tracked_surface_mean_mm if tracked_surface_mean_mm is not None else 'na'}/"
                f"{tracked_surface_min_mm if tracked_surface_min_mm is not None else 'na'} "
                f"all_surface_mm(mean/min)="
                f"{all_surface_mean_mm if all_surface_mean_mm is not None else 'na'}/"
                f"{all_surface_min_mm if all_surface_min_mm is not None else 'na'}"
            )

    def _simulate_close_settle(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]], steps: int):
        """Pinned annealed close from IK pose, keeping the best contact-aligned close state.

        Instead of one-shot close bias, we linearly ramp close bias over `steps` and
        evaluate contact error after each pinned physics step. The best state along
        this annealing trajectory is restored at the end to avoid over-closing.
        """
        if steps <= 0 or self.object is None:
            return {}
        env_idx = int(env_idx)
        close_bias = float(self.valid_grasp_vf_close_bias)
        squeeze_torque = float(self.valid_grasp_vf_squeeze_torque)
        probe_hold_steps = max(0, int(self.valid_grasp_vf_anneal_probe_hold_steps))
        pin_pos = self.object.root_pos[env_idx].clone()
        pin_quat = self.object.root_quat[env_idx].clone()
        pin_dof = self.object.dof_pos[env_idx].clone()

        active_fingers = self._get_active_contact_fingers(targets)

        # Base IK pose and active flex joints only (wrist/abduction stay fixed)
        problem = self._build_side_contact_problem(env_idx, targets)
        side_meta = {}
        for side, robot in self.robots.items():
            side_prob = problem.get(side, None)
            if side_prob is None:
                continue
            
            joint_idxs = side_prob['joint_idxs']
            link_local_idxs = side_prob['link_local_idxs']
            target_positions = side_prob['target_positions']

            q = robot.dof_pos[env_idx].clone()
            name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
            flex_idxs = [
                int(dof_idx) for dof_idx in joint_idxs
                if "abd" not in name_by_idx.get(dof_idx, "").lower()
                and "spread" not in name_by_idx.get(dof_idx, "").lower()
                and "forearm" not in name_by_idx.get(dof_idx, "").lower()
            ]
            if len(flex_idxs) == 0:
                continue
            
            # Calibrate close direction per joint from local contact-error probe.
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
                robot.set_joint_position(plus[None], env_idxs=[env_idx])
                plus_err, plus_cnt, _ = self._contact_alignment_error(env_idx, targets)
                plus_metric = plus_err / max(plus_cnt, 1) if plus_cnt > 0 else float("inf")
                robot.set_joint_position(minus[None], env_idxs=[env_idx])
                minus_err, minus_cnt, _ = self._contact_alignment_error(env_idx, targets)
                minus_metric = minus_err / max(minus_cnt, 1) if minus_cnt > 0 else float("inf")
                close_signs[dof_idx] = 1.0 if plus_metric <= minus_metric else -1.0
                robot.set_joint_position(base[None], env_idxs=[env_idx])

            side_meta[side] = dict(
                robot=robot,
                base_q=q,
                flex_idxs=flex_idxs,
                close_signs=close_signs,
                joint_idxs=joint_idxs,
                link_local_idxs=link_local_idxs,
                target_positions=target_positions,
            )

        if len(side_meta) == 0:
            return {}

        def _candidate_metrics():
            err, cnt, _ = self._contact_alignment_error(env_idx, targets)
            per_link = err / max(cnt, 1) if cnt > 0 else float("inf")
            side_contact_pairs = self._count_actual_contact_pairs_by_side(env_idx, targets=targets)
            contact_pairs = int(sum(side_contact_pairs.values()))
            contacts_ok = self._passes_contact_gate(side_contact_pairs)
            opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
            opposition_ok = self._passes_functional_opposition(opposition)
            stable_ok = True
            drop = 0.0
            vel = 0.0
            ang_vel = 0.0
            persistence_ratio = 1.0
            persistence_ok = True
            if probe_hold_steps > 0:
                probe_state = self._snapshot_env_state(env_idx)
                hold_targets = {side: robot.dof_pos[env_idx].clone() for side, robot in self.robots.items()}
                min_z, max_vel, max_ang_vel = self._simulate_settle(env_idx, hold_targets, probe_hold_steps)
                settle_stats = self._last_settle_stats.get(env_idx, dict())
                stable_ok, drop, vel, ang_vel = self._passes_stability_gate(
                    env_idx,
                    init_obj_z=pin_pos[2].item(),
                    min_z=min_z,
                    max_vel=max_vel,
                    max_ang_vel=max_ang_vel,
                    settle_stats=settle_stats,
                )
                persistence_ratio = float(settle_stats.get("persistence_ratio", 1.0))
                persistence_ok = bool(settle_stats.get("persistence_ok", True))
                self._restore_env_state(env_idx, probe_state)
                self._compute_intermediate_values()
            return dict(
                per_link=per_link,
                contact_pairs=contact_pairs,
                side_contact_pairs=side_contact_pairs,
                contacts_ok=contacts_ok,
                opposition_ok=opposition_ok,
                stable=stable_ok,
                drop=drop,
                vel=vel,
                ang_vel=ang_vel,
                persistence_ratio=persistence_ratio,
                persistence_ok=persistence_ok,
                opposition_score=float(opposition["score"]),
                opposition_dot=float(opposition["dot"]),
                opposition_thumb_contacts=int(opposition["thumb_contacts"]),
                opposition_finger_contacts=int(opposition["finger_contacts"]),
            )

        def _is_better(curr, best):
            # Prioritize compliance-based stability and functional opposition first.
            def _rank(v):
                if v["stable"] and v["contacts_ok"] and v["opposition_ok"]:
                    return (
                        3,
                        float(v["opposition_score"]),
                        int(v["contact_pairs"]),
                        -float(v["per_link"]),
                        -float(v["drop"]),
                        -float(v["vel"]),
                        -float(v["ang_vel"]),
                    )
                if v["stable"] and v["contacts_ok"]:
                    return (
                        2,
                        float(v["opposition_score"]),
                        int(v["contact_pairs"]),
                        -float(v["per_link"]),
                        -float(v["drop"]),
                        -float(v["vel"]),
                        -float(v["ang_vel"]),
                    )
                return (
                    0,
                    float(v["opposition_score"]),
                    int(v["contact_pairs"]),
                    -float(v["per_link"]),
                    -float(v["drop"]),
                    -float(v["vel"]),
                    -float(v["ang_vel"]),
                )
            key = _rank(curr)
            best_key = _rank(best)
            return key > best_key

        # Keep best close state by hold-stability first, then contact and geometric tie-breaks.
        self._compute_intermediate_values()
        best = _candidate_metrics()
        best_state = self._snapshot_env_state(env_idx)
        best_step = 0
        printed_no_contact_diag = False

        for step_i in range(steps):
            self._compute_intermediate_values()
            alpha = float(step_i + 1) / float(steps)
            self.object.set_object_state(
                pin_pos[None], pin_quat[None], pin_dof[None], env_idxs=[env_idx]
            )
            for side, sm in side_meta.items():
                robot = sm["robot"]
                target = sm["base_q"].clone()
                
                # J^T virtual attraction
                attract_gain = float(self.valid_grasp_vf_attract_gain)
                attract_clip = float(self.valid_grasp_vf_attract_dq_clip)
                if attract_gain > 0 and len(sm['joint_idxs']) > 0:
                    J, curr_pos = self._numeric_contact_jacobian(
                        env_idx, robot, target, sm['joint_idxs'], sm['link_local_idxs']
                    )
                    residual = (sm['target_positions'] - curr_pos).reshape(-1)
                    dq_attract = J.T @ residual * attract_gain
                    dq_attract = torch.clamp(dq_attract, -attract_clip, attract_clip)
                    target[sm['joint_idxs']] = torch.clamp(
                        target[sm['joint_idxs']] + dq_attract,
                        robot.dof_limits[sm['joint_idxs'], 0],
                        robot.dof_limits[sm['joint_idxs'], 1],
                    )

                squeeze = []
                for dof_idx in sm["flex_idxs"]:
                    sign = sm["close_signs"].get(dof_idx, 1.0)
                    target[dof_idx] += sign * alpha * close_bias
                    squeeze.append(sign * squeeze_torque)
                target = torch.clamp(target, robot.dof_limits[:, 0], robot.dof_limits[:, 1])
                robot.control_joint_position(target[None], env_idxs=[env_idx])
                if squeeze_torque > 0.0 and len(squeeze) > 0:
                    squeeze_t = torch.tensor(squeeze, dtype=torch.float32, device=self.device)
                    robot.entity.control_dofs_force(
                        squeeze_t,
                        dofs_idx_local=sm["flex_idxs"],
                        envs_idx=[env_idx],
                    )
            for _, obj in self.objects.items():
                obj.step()
            self.scene.step()
            self.object.set_object_state(
                pin_pos[None], pin_quat[None], pin_dof[None], env_idxs=[env_idx]
            )
            self._compute_intermediate_values()
            curr = _candidate_metrics()
            if self.valid_grasp_debug and curr["contact_pairs"] == 0 and not printed_no_contact_diag:
                self._debug_contact_diagnostics(
                    env_idx,
                    targets=targets,
                    prefix=f"[VF-close-diag] step={step_i}",
                )
                printed_no_contact_diag = True
            if np.isfinite(curr["per_link"]) and _is_better(curr, best):
                best = curr
                best_state = self._snapshot_env_state(env_idx)
                best_step = step_i + 1
            if self.valid_grasp_debug and (step_i % 5 == 0 or step_i == steps - 1):
                print(
                    f"[VF-close] env={env_idx} step={step_i} alpha={alpha:.2f} "
                    f"per_link_err={curr['per_link']:.6f} contacts={curr['contact_pairs']} "
                    f"side_contacts={curr['side_contact_pairs']} contacts_ok={curr['contacts_ok']} "
                    f"opp_ok={curr['opposition_ok']} "
                    f"stable={curr['stable']} drop={curr['drop']:.4f} vel={curr['vel']:.4f} ang_vel={curr['ang_vel']:.4f} "
                    f"persist={curr['persistence_ratio']:.2f}/{curr['persistence_ok']} "
                    f"opp_score={curr['opposition_score']:.4f} opp_dot={curr['opposition_dot']:.4f} "
                    f"best_per_link={best['per_link']:.6f} best_contacts={best['contact_pairs']} "
                    f"best_stable={best['stable']}"
                )

        self._restore_env_state(env_idx, best_state)
        self._compute_intermediate_values()
        # Clear explicit squeeze torques after restoring selected state.
        if squeeze_torque > 0.0:
            for _, sm in side_meta.items():
                if len(sm["flex_idxs"]) == 0:
                    continue
                sm["robot"].entity.control_dofs_force(
                    torch.zeros((len(sm["flex_idxs"]),), dtype=torch.float32, device=self.device),
                    dofs_idx_local=sm["flex_idxs"],
                    envs_idx=[env_idx],
                )
        pinned_contact_links = self._get_contact_link_touches_by_side(env_idx, targets=targets)
        if self.valid_grasp_debug and best["contact_pairs"] == 0:
            self._debug_contact_diagnostics(
                env_idx,
                targets=targets,
                prefix="[VF-close-diag] final",
            )
        if self.valid_grasp_debug:
            print(
                f"[VF-close] env={env_idx} selected_step={best_step}/{steps} "
                f"best_per_link_err={best['per_link']:.6f} best_contacts={best['contact_pairs']} "
                f"best_side_contacts={best['side_contact_pairs']} best_contacts_ok={best['contacts_ok']} "
                f"best_opp_ok={best['opposition_ok']} best_stable={best['stable']} "
                f"best_drop={best['drop']:.4f} best_vel={best['vel']:.4f} "
                f"best_ang_vel={best['ang_vel']:.4f} best_persist={best['persistence_ratio']:.2f}/{best['persistence_ok']} "
                f"best_opp_score={best['opposition_score']:.4f} best_opp_dot={best['opposition_dot']:.4f}"
            )
        return pinned_contact_links

    def _simulate_pre_tension(
        self,
        env_idx: int,
        targets: Dict[str, Dict[str, torch.Tensor]],
        steps: int = 20,
    ):
        """Pinned physics: simultaneously apply squeeze bias and J^T virtual attraction.

        Object is pinned before and after every physics step.  Each step the PD target
        is updated by (a) a small incremental close bias on flex joints and (b) a J^T
        attraction that pulls contact joints toward their target positions on the object.
        An optional squeeze torque is also injected on flex joints.
        """
        if steps <= 0 or self.object is None:
            return
        env_idx = int(env_idx)
        self._compute_intermediate_values()
        pin_pos = self.object.root_pos[env_idx].clone()
        pin_quat = self.object.root_quat[env_idx].clone()
        pin_dof = self.object.dof_pos[env_idx].clone()

        attract_gain = float(self.valid_grasp_vf_attract_gain)
        attract_clip = float(self.valid_grasp_vf_attract_dq_clip)
        close_bias = float(self.valid_grasp_vf_close_bias)
        squeeze_torque = float(self.valid_grasp_vf_squeeze_torque)

        problem = self._build_side_contact_problem(env_idx, targets)
        side_meta = {}
        for side, robot in self.robots.items():
            side_prob = problem.get(side, None)
            if side_prob is None:
                continue
            joint_idxs = side_prob['joint_idxs']        # list[int]: wrist + finger joints
            link_local_idxs = side_prob['link_local_idxs']
            target_positions = side_prob['target_positions']  # (N, 3)
            name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
            flex_idxs = [
                int(dof_idx) for dof_idx in joint_idxs
                if "abd" not in name_by_idx.get(dof_idx, "").lower()
                and "spread" not in name_by_idx.get(dof_idx, "").lower()
                and "forearm" not in name_by_idx.get(dof_idx, "").lower()
            ]
            # Probe ±delta per flex joint to determine correct closing direction.
            q = robot.dof_pos[env_idx].clone()
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
                robot.set_joint_position(plus[None], env_idxs=[env_idx])
                plus_err, plus_cnt, _ = self._contact_alignment_error(env_idx, targets)
                plus_metric = plus_err / max(plus_cnt, 1) if plus_cnt > 0 else float("inf")
                robot.set_joint_position(minus[None], env_idxs=[env_idx])
                minus_err, minus_cnt, _ = self._contact_alignment_error(env_idx, targets)
                minus_metric = minus_err / max(minus_cnt, 1) if minus_cnt > 0 else float("inf")
                close_signs[dof_idx] = 1.0 if plus_metric <= minus_metric else -1.0
                robot.set_joint_position(base[None], env_idxs=[env_idx])
            side_meta[side] = dict(
                robot=robot,
                joint_idxs=joint_idxs,
                link_local_idxs=link_local_idxs,
                target_positions=target_positions,
                flex_idxs=flex_idxs,
                close_signs=close_signs,
            )

        if len(side_meta) == 0:
            return

        step_close = close_bias / max(steps, 1)

        for step_i in range(steps):
            self.object.set_object_state(pin_pos[None], pin_quat[None], pin_dof[None], env_idxs=[env_idx])
            self._compute_intermediate_values()

            for side, sm in side_meta.items():
                robot = sm['robot']
                q = robot.dof_pos[env_idx].clone()

                # J^T virtual attraction: pull joints toward demo contact positions.
                if attract_gain > 0 and len(sm['joint_idxs']) > 0:
                    J, curr_pos = self._numeric_contact_jacobian(
                        env_idx, robot, q, sm['joint_idxs'], sm['link_local_idxs']
                    )
                    residual = (sm['target_positions'] - curr_pos).reshape(-1)
                    dq_attract = J.T @ residual * attract_gain
                    dq_attract = torch.clamp(dq_attract, -attract_clip, attract_clip)
                    q[sm['joint_idxs']] = torch.clamp(
                        q[sm['joint_idxs']] + dq_attract,
                        robot.dof_limits[sm['joint_idxs'], 0],
                        robot.dof_limits[sm['joint_idxs'], 1],
                    )

                # Incremental squeeze bias on flex joints only (direction-aware).
                if step_close > 0 and len(sm['flex_idxs']) > 0:
                    for dof_idx in sm['flex_idxs']:
                        sign = sm['close_signs'].get(dof_idx, 1.0)
                        q[dof_idx] = torch.clamp(
                            q[dof_idx] + sign * step_close,
                            robot.dof_limits[dof_idx, 0],
                            robot.dof_limits[dof_idx, 1],
                        )

                robot.control_joint_position(q[None], env_idxs=[env_idx])

                if squeeze_torque > 0.0 and len(sm['flex_idxs']) > 0:
                    sq_t = torch.full(
                        (len(sm['flex_idxs']),), squeeze_torque,
                        dtype=torch.float32, device=self.device,
                    )
                    robot.entity.control_dofs_force(
                        sq_t, dofs_idx_local=sm['flex_idxs'], envs_idx=[env_idx]
                    )

            for _, obj in self.objects.items():
                obj.step()
            self.scene.step()
            self.object.set_object_state(pin_pos[None], pin_quat[None], pin_dof[None], env_idxs=[env_idx])
            self._compute_intermediate_values()

        if self.valid_grasp_debug:
            err, cnt, _ = self._contact_alignment_error(env_idx, targets)
            side_contacts = self._count_actual_contact_pairs_by_side(env_idx, targets=targets)
            print(
                f"[VF-pretension] env={env_idx} after {steps} steps: "
                f"error={err:.6f} count={cnt} contacts={side_contacts}"
            )

    def _simulate_soft_unpin(
        self,
        env_idx: int,
        hold_targets: Dict[str, torch.Tensor],
        steps_assist: int = 50,
        steps_free: int = 20,
    ):
        """Release object from pin while applying decreasing gravity compensation.

        Phase 1 (steps_assist steps): upward force = mass * g * (1 - step/steps_assist),
          so the hand gets full gravity support at step 0 and none at step steps_assist-1.
        Phase 2 (steps_free steps): pure free physics with PD hold.

        Returns (min_z, max_vel, max_ang_vel) measured over the combined phases.
        """
        if self.object is None:
            return None, None, None
        env_idx = int(env_idx)
        self._zero_object_velocity_envs([env_idx])

        min_z = None
        max_vel = None
        max_ang_vel = None
        total_steps = steps_assist + steps_free
        squeeze_torque = float(self.valid_grasp_vf_squeeze_torque)

        # Pre-compute flex joints per side for squeeze torque.
        side_flex_idxs = {}
        for side, robot in self.robots.items():
            name_by_idx = {idx: name for idx, name in zip(robot.actuated_dof_idxs, robot.actuated_dof_names)}
            flex_idxs = []
            for dof_idx in robot.actuated_dof_idxs:
                lname = name_by_idx.get(dof_idx, "").lower()
                if "abd" not in lname and "spread" not in lname and "forearm" not in lname:
                    flex_idxs.append(int(dof_idx))
            if len(flex_idxs) > 0:
                side_flex_idxs[side] = flex_idxs

        for step_i in range(total_steps):
            for side, robot in self.robots.items():
                target = hold_targets.get(side, None)
                if target is None:
                    continue
                robot.control_joint_position(target[None], env_idxs=[env_idx])
                
                flex_idxs = side_flex_idxs.get(side, [])
                if squeeze_torque > 0.0 and len(flex_idxs) > 0:
                    sq_t = torch.full(
                        (len(flex_idxs),), squeeze_torque,
                        dtype=torch.float32, device=self.device,
                    )
                    robot.entity.control_dofs_force(
                        sq_t, dofs_idx_local=flex_idxs, envs_idx=[env_idx]
                    )

            for _, obj in self.objects.items():
                obj.step()
            if step_i < steps_assist:
                # alpha: 0 = full gravity comp, 1 = no comp
                alpha = float(step_i) / float(max(steps_assist - 1, 1))
                self._apply_gravity_compensation_force(env_idx, alpha=alpha)
            self.scene.step()
            if self.object is not None:
                self.object.update_value_buffers()
                curr_z = float(self.object.root_pos[env_idx, 2].item())
                vel = float(torch.norm(self.object.root_lin_vel[env_idx]).item())
                ang_vel = float(torch.norm(self.object.root_ang_vel[env_idx]).item())
                if np.isfinite(curr_z) and (min_z is None or curr_z < min_z):
                    min_z = curr_z
                if np.isfinite(vel) and (max_vel is None or vel > max_vel):
                    max_vel = vel
                if np.isfinite(ang_vel) and (max_ang_vel is None or ang_vel > max_ang_vel):
                    max_ang_vel = ang_vel

        self._compute_intermediate_values()
        if min_z is None and self.object is not None:
            min_z = float(self.object.root_pos[env_idx, 2].item())
        if max_vel is None and self.object is not None:
            max_vel = float(torch.norm(self.object.root_lin_vel[env_idx]).item())
        if max_ang_vel is None and self.object is not None:
            max_ang_vel = float(torch.norm(self.object.root_ang_vel[env_idx]).item())
        return min_z, max_vel, max_ang_vel

    def _score_candidate(
        self,
        env_idx: int,
        targets: Dict[str, Dict[str, torch.Tensor]],
        init_obj_z: float,
        min_z: float = None,
        max_vel: float = None,
        max_ang_vel: float = None,
    ):
        error, count, _ = self._contact_alignment_error(env_idx, targets)
        if count == 0:
            return float("inf"), error, count
        if not np.isfinite(error):
            return float("inf"), error, count
        score = error / max(count, 1)
        actual_contact_pairs = self._count_actual_contact_pairs(env_idx, targets=targets)
        if self.valid_grasp_opposition_weight > 0.0:
            opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
            score -= self.valid_grasp_opposition_weight * float(opposition["score"])
        if self.object is not None:
            curr_z = float(self.object.root_pos[env_idx, 2].item())
            vel = float(torch.norm(self.object.root_lin_vel[env_idx]).item())
            ang_vel = float(torch.norm(self.object.root_ang_vel[env_idx]).item())
            use_z = min_z if min_z is not None else curr_z
            use_vel = max_vel if max_vel is not None else vel
            use_ang_vel = max_ang_vel if max_ang_vel is not None else ang_vel
            if np.isfinite(use_z) and np.isfinite(use_vel) and np.isfinite(use_ang_vel) and np.isfinite(init_obj_z):
                drop = max(0.0, init_obj_z - use_z)
                score += self.valid_grasp_slip_weight * drop
                score += self.valid_grasp_vel_weight * use_vel
                score += self.valid_grasp_ang_vel_weight * use_ang_vel
        if actual_contact_pairs > 0:
            score -= self.valid_grasp_contact_bonus * float(actual_contact_pairs)
        return score, error, count

    def _passes_stability_gate(
        self,
        env_idx: int,
        init_obj_z: float,
        min_z: float = None,
        max_vel: float = None,
        max_ang_vel: float = None,
        settle_stats: Dict = None,
    ):
        """Hard pass/fail gate for free-hold stability."""
        if self.object is None:
            return True, 0.0, 0.0, 0.0
        curr_z = float(self.object.root_pos[env_idx, 2].item())
        curr_vel = float(torch.norm(self.object.root_lin_vel[env_idx]).item())
        curr_ang_vel = float(torch.norm(self.object.root_ang_vel[env_idx]).item())
        use_z = min_z if min_z is not None else curr_z
        use_vel = max_vel if max_vel is not None else curr_vel
        use_ang_vel = max_ang_vel if max_ang_vel is not None else curr_ang_vel
        drop = max(0.0, init_obj_z - use_z) if np.isfinite(use_z) and np.isfinite(init_obj_z) else float("inf")
        vel = use_vel if np.isfinite(use_vel) else float("inf")
        ang_vel = use_ang_vel if np.isfinite(use_ang_vel) else float("inf")
        pass_drop = (self.valid_grasp_hold_max_drop <= 0.0) or (drop <= self.valid_grasp_hold_max_drop)
        pass_vel = (self.valid_grasp_hold_max_vel <= 0.0) or (vel <= self.valid_grasp_hold_max_vel)
        pass_ang = (self.valid_grasp_hold_max_ang_vel <= 0.0) or (ang_vel <= self.valid_grasp_hold_max_ang_vel)
        if settle_stats is None:
            settle_stats = self._last_settle_stats.get(int(env_idx), dict())
        persistence_ok = bool(settle_stats.get("persistence_ok", True))
        if self.valid_grasp_contact_persistence_min > 0.0:
            return bool(pass_drop and pass_vel and pass_ang and persistence_ok), float(drop), float(vel), float(ang_vel)
        return bool(pass_drop and pass_vel and pass_ang), float(drop), float(vel), float(ang_vel)

    def _snapshot_env_state(self, env_idx: int):
        state = dict(robots=dict(), obj=None)
        env_idx = int(env_idx)
        for side, robot in self.robots.items():
            state['robots'][side] = robot.dof_pos[env_idx].clone()
        if self.object is not None:
            state['obj'] = dict(
                pos=self.object.root_pos[env_idx].clone(),
                quat=self.object.root_quat[env_idx].clone(),
                dof=self.object.dof_pos[env_idx].clone(),
            )
        return state

    def _restore_env_state(self, env_idx: int, state: Dict):
        env_idx = int(env_idx)
        for side, robot in self.robots.items():
            if side not in state['robots']:
                continue
            target = state['robots'][side]
            robot.set_joint_position(target[None], env_idxs=[env_idx])
        if self.object is not None and state.get('obj', None) is not None:
            obj_state = state['obj']
            self.object.set_object_state(
                obj_state['pos'][None],
                obj_state['quat'][None],
                obj_state['dof'][None],
                env_idxs=[env_idx],
            )

    def _optimize_grasp_state(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        if len(targets) == 0:
            return False
        env_idx = int(env_idx)
        snapshot = self._snapshot_env_state(env_idx)
        best_state = snapshot
        self._compute_intermediate_values()
        init_obj_z = float(self.object.root_pos[env_idx, 2].item()) if self.object is not None else 0.0
        best_score, best_error, best_count = self._score_candidate(env_idx, targets, init_obj_z)
        if self.valid_grasp_debug:
            print(
                f"[valid-grasp] env={env_idx} init score={best_score:.6f} "
                f"error={best_error:.6f} count={best_count}"
            )
        if best_count == 0 or not np.isfinite(best_score):
            self._restore_env_state(env_idx, snapshot)
            if self.valid_grasp_debug:
                print(f"[valid-grasp] env={env_idx} early exit: invalid initial score/count")
            return False
        init_opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
        init_opposition_ok = self._passes_functional_opposition(init_opposition)
        if (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh and init_opposition_ok:
            if self.valid_grasp_debug:
                print(f"[valid-grasp] env={env_idx} early success: already below threshold")
            return True
        if self.valid_grasp_debug and (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh and not init_opposition_ok:
            print(
                f"[valid-grasp] env={env_idx} early threshold met but opposition failed: "
                f"score={init_opposition['score']:.4f} dot={init_opposition['dot']:.4f}"
            )
        if self.valid_grasp_refine_mode == 'virtual_force':
            # Step 1: Kinematic IK snap — bring wrist + finger joints near demo contact positions.
            # Do NOT run free-physics on the raw snapshot; kinematic state is often non-physical
            # and causes immediate NaN when released under gravity.
            changed = self._solve_contact_ik_lm_for_env(env_idx, targets)
            if not changed:
                changed = self._solve_contact_ik_for_env(env_idx, targets)
            if not changed:
                if self.valid_grasp_debug:
                    print(f"[VF] env={env_idx} IK produced no change, restoring snapshot")
                self._restore_env_state(env_idx, snapshot)
                self._compute_intermediate_values()
                return False
            if self.valid_grasp_debug:
                ik_err, ik_cnt, _ = self._contact_alignment_error(env_idx, targets)
                print(f"[VF] env={env_idx} after IK: error={ik_err:.6f} count={ik_cnt}")

            # Step 2: Pre-tension (object pinned). Squeeze + J^T attraction seats fingers
            # against the object surface before releasing the pin.
            self._simulate_close_settle(env_idx, targets, int(self.valid_grasp_vf_attract_steps))

            # Snapshot the object state while it is still pinned at the demo pose.
            # After soft-unpin free physics, we restore this so the episode always
            # starts at the demo object position (only optimized finger joints are kept).
            pinned_obj_snapshot = None
            if self.object is not None:
                pinned_obj_snapshot = dict(
                    pos=self.object.root_pos[env_idx].clone(),
                    quat=self.object.root_quat[env_idx].clone(),
                    dof=self.object.dof_pos[env_idx].clone(),
                )

            # Step 3: Soft unpin — gradually remove gravity compensation over steps_assist steps,
            # then hold freely for steps_free more steps.
            hold_targets = {side: robot.dof_pos[env_idx].clone() for side, robot in self.robots.items()}
            min_z, max_vel, max_ang_vel = self._simulate_soft_unpin(
                env_idx,
                hold_targets,
                steps_assist=int(self.valid_grasp_vf_soft_unpin_steps),
                steps_free=int(self.valid_grasp_vf_hold_steps),
            )

            # Step 4: Validation gate.
            stable_ok, hold_drop, hold_vel, hold_ang_vel = self._passes_stability_gate(
                env_idx, init_obj_z, min_z=min_z, max_vel=max_vel, max_ang_vel=max_ang_vel
            )
            side_contact_pairs = self._count_actual_contact_pairs_by_side(env_idx, targets=targets)
            contacts_ok = self._passes_contact_gate(side_contact_pairs)
            opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
            opposition_ok = self._passes_functional_opposition(opposition)
            thumb_ok = int(opposition.get('thumb_contacts', 0)) > 0
            finger_ok = int(opposition.get('finger_contacts', 0)) > 0

            if self.valid_grasp_debug:
                print(
                    f"[VF] env={env_idx} gate: drop={hold_drop:.4f} vel={hold_vel:.4f} "
                    f"ang_vel={hold_ang_vel:.4f} stable={stable_ok} "
                    f"contacts={side_contact_pairs} contacts_ok={contacts_ok} "
                    f"thumb={thumb_ok} finger={finger_ok} "
                    f"opp={opposition['score']:.4f} opp_ok={opposition_ok}"
                )

            # After free-physics validation the object has drifted from the demo pose.
            # Restore the object to its pinned (demo) position while keeping the
            # optimized finger joint positions from the settled state.
            if pinned_obj_snapshot is not None:
                self.object.set_object_state(
                    pinned_obj_snapshot['pos'][None],
                    pinned_obj_snapshot['quat'][None],
                    pinned_obj_snapshot['dof'][None],
                    env_idxs=[env_idx],
                )
                self._compute_intermediate_values()

            if stable_ok and contacts_ok and thumb_ok and finger_ok and opposition_ok:
                return True
            # Fallback: restore initial snapshot.
            if self.valid_grasp_debug:
                reasons = []
                if not stable_ok:
                    reasons.append(f"drop={hold_drop:.4f} vel={hold_vel:.4f}")
                if not contacts_ok:
                    reasons.append(f"contacts={side_contact_pairs}")
                if not (thumb_ok and finger_ok):
                    reasons.append(f"thumb={thumb_ok} finger={finger_ok}")
                if not opposition_ok:
                    reasons.append(f"opp={opposition['score']:.4f}")
                print(f"[VF] env={env_idx} FAILED ({'; '.join(reasons)}), restoring snapshot")
            return True
            
            self._restore_env_state(env_idx, snapshot)
            self._compute_intermediate_values()
            return False
        if self.valid_grasp_refine_mode == 'ik':
            changed = self._solve_contact_ik_lm_for_env(env_idx, targets)
            if not changed:
                changed = self._solve_contact_ik_for_env(env_idx, targets)
            if changed:
                expected_contact_links = self._get_contact_link_touches_by_side(env_idx, targets=targets)
                hold_targets = {side: robot.dof_pos[env_idx].clone() for side, robot in self.robots.items()}
                min_z, max_vel, max_ang_vel = self._simulate_settle(
                    env_idx,
                    hold_targets,
                    int(self.valid_grasp_settle_steps),
                    expected_contact_links_by_side=expected_contact_links,
                )
                settle_stats = self._last_settle_stats.get(env_idx, dict())
                new_score, new_error, new_count = self._score_candidate(
                    env_idx, targets, init_obj_z, min_z=min_z, max_vel=max_vel, max_ang_vel=max_ang_vel
                )
                stable_ok, hold_drop, hold_vel, hold_ang_vel = self._passes_stability_gate(
                    env_idx,
                    init_obj_z,
                    min_z=min_z,
                    max_vel=max_vel,
                    max_ang_vel=max_ang_vel,
                    settle_stats=settle_stats,
                )
                side_contact_pairs = self._count_actual_contact_pairs_by_side(env_idx, targets=targets)
                contacts_ok = self._passes_contact_gate(side_contact_pairs)
                opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
                opposition_ok = self._passes_functional_opposition(opposition)
                persist_ratio = float(settle_stats.get("persistence_ratio", 1.0))
                persist_ok = bool(settle_stats.get("persistence_ok", True))
                if self.valid_grasp_debug:
                    print(
                        f"[valid-grasp] env={env_idx} IK score {best_score:.6f}->{new_score:.6f} "
                        f"error={new_error:.6f} count={new_count}"
                    )
                    print(
                        f"[valid-grasp] env={env_idx} IK hold gate: drop={hold_drop:.4f} "
                        f"vel={hold_vel:.4f} ang_vel={hold_ang_vel:.4f} pass={stable_ok} "
                        f"persist={persist_ratio:.2f}/{persist_ok} "
                        f"opp_score={opposition['score']:.4f} opp_dot={opposition['dot']:.4f} opp_ok={opposition_ok} "
                        f"side_contacts={side_contact_pairs} contacts_ok={contacts_ok}"
                    )
                if new_count > 0 and new_score < best_score and stable_ok and contacts_ok and opposition_ok:
                    best_score = new_score
                    best_error = new_error
                    best_state = self._snapshot_env_state(env_idx)
                    self._restore_env_state(env_idx, best_state)
                    return (best_error / max(new_count, 1)) <= self.valid_grasp_contact_thresh
            self._restore_env_state(env_idx, best_state)
            final_opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
            return (
                (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh
                and self._passes_functional_opposition(final_opposition)
            )
        if self.valid_grasp_refine_mode not in ['sampling', 'ik', 'virtual_force']:
            if self.valid_grasp_debug:
                print(f"[valid-grasp] unknown refine mode={self.valid_grasp_refine_mode}, fallback to sampling")
        improved_samples = 0
        total_samples = 0
        for _ in range(max(1, int(self.valid_grasp_sample_iters))):
            base_state = self._snapshot_env_state(env_idx)
            for _ in range(max(1, int(self.valid_grasp_sample_count))):
                total_samples += 1
                self._restore_env_state(env_idx, base_state)
                changed = self._apply_two_sided_finger_probing(env_idx, targets, init_obj_z)
                if not changed:
                    if not self._apply_sampled_finger_perturbation(env_idx, targets):
                        break
                expected_contact_links = self._get_contact_link_touches_by_side(env_idx, targets=targets)
                hold_targets = {side: robot.dof_pos[env_idx].clone() for side, robot in self.robots.items()}
                min_z, max_vel, max_ang_vel = self._simulate_settle(
                    env_idx,
                    hold_targets,
                    int(self.valid_grasp_settle_steps),
                    expected_contact_links_by_side=expected_contact_links,
                )
                settle_stats = self._last_settle_stats.get(env_idx, dict())
                new_score, new_error, new_count = self._score_candidate(
                    env_idx, targets, init_obj_z, min_z=min_z, max_vel=max_vel, max_ang_vel=max_ang_vel
                )
                stable_ok, _, _, _ = self._passes_stability_gate(
                    env_idx,
                    init_obj_z,
                    min_z=min_z,
                    max_vel=max_vel,
                    max_ang_vel=max_ang_vel,
                    settle_stats=settle_stats,
                )
                side_contact_pairs = self._count_actual_contact_pairs_by_side(env_idx, targets=targets)
                contacts_ok = self._passes_contact_gate(side_contact_pairs)
                opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
                opposition_ok = self._passes_functional_opposition(opposition)
                if new_count > 0 and new_score < best_score and stable_ok and contacts_ok and opposition_ok:
                    improved_samples += 1
                    if self.valid_grasp_debug:
                        print(
                            f"[valid-grasp] env={env_idx} improved sample "
                            f"score {best_score:.6f}->{new_score:.6f} error={new_error:.6f} count={new_count} "
                            f"opp_score={opposition['score']:.4f} opp_dot={opposition['dot']:.4f}"
                        )
                    best_score = new_score
                    best_error = new_error
                    best_state = self._snapshot_env_state(env_idx)
                    if (best_error / max(new_count, 1)) <= self.valid_grasp_contact_thresh:
                        self._restore_env_state(env_idx, best_state)
                        if self.valid_grasp_debug:
                            print(f"[valid-grasp] env={env_idx} success: reached threshold")
                        return True
            self._restore_env_state(env_idx, best_state)
        self._restore_env_state(env_idx, best_state)
        if self.valid_grasp_debug:
            final_opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
            final_ok = (
                (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh
                and self._passes_functional_opposition(final_opposition)
            )
            print(
                f"[valid-grasp] env={env_idx} done samples={total_samples} improved={improved_samples} "
                f"best_score={best_score:.6f} best_error={best_error:.6f} success={final_ok}"
            )
        final_opposition = self._compute_thumb_finger_opposition(env_idx, targets=targets)
        return (
            (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh
            and self._passes_functional_opposition(final_opposition)
        )

    def _reset_env_to_demo_anchor(self, env_idx: int):
        env_idx = int(env_idx)
        self.episode_length_buf[env_idx] = 0
        self.episode_start_buf[env_idx] = 0
        self.randomization.on_reset_idx([env_idx])
        episode_start = self.episode_start_buf[env_idx:env_idx+1]
        for _, robot in self.robots.items():
            robot.reset_idx(env_idxs=[env_idx], episode_start=episode_start)
        for _, obj in self.objects.items():
            obj.reset_idx(env_idxs=[env_idx], episode_start=episode_start)
        if self.use_latent_world_model:
            self.latent_buf[env_idx] = 0.0
         
    def reset(self): 
        # reset all envs
        env_idxs = torch.arange(self.num_envs)
        self.reset_idx(env_idxs)  
        if self.use_rl_games:
            return self.get_observations(), dict()
        else:
            self.obs_buf[:] = self.get_observations()
        self.scene.step()
        return self.obs_buf, None #self.extras

    def transform_vertice_frame(self, vertices, pose):
        """ transform the vertices to the object frame """
        # vertices: (N, K, 3), pose: (N, 7)
        quat = pose[:, 3:7]
        matrices = matrix_from_quat(quat)
        offsets = pose[:, :3].unsqueeze(1)
        transformed = torch.einsum("nij,nkj->nki", matrices, vertices) + offsets
        return transformed
        
    def compute_closest_vertice_dist_single(self, verts, keypoint_pos, pose):
        # verts: (N, K, 3), keypoint_pos: (N, 3), pose: (N, 7)
        verts = verts.unsqueeze(0).repeat((keypoint_pos.shape[0], 1, 1)).to(pose.device)
        verts = self.transform_vertice_frame(verts, pose)
        dists = torch.cdist(keypoint_pos, verts, p=2)
        min_dists = torch.min(dists, dim=-1).values
        return min_dists
        
    def _add_camera(self, camera_kwargs):
        ''' Set camera position and direction, NOTE this must be done BEFORE scene.build()''' 
        cameras = dict()
        for name, kwargs in camera_kwargs.items():
            cam_pos = kwargs.get('pos', (0.0, -1.5, 1.2))
            lookat = kwargs.get('lookat', CARDBOARD_POS)
            res = kwargs.get('res', CAMERA_RES)
            if self.num_envs < 3:
                res = (500, 500)
            fov = kwargs.get('fov', 30)
            if 'renderer' in self.scene_cfg: # ray tracing cam
                res = kwargs.get('raytrace_res', (1024, 1024))
                fov = kwargs.get('raytrace_fov', 18) 
            GUI = kwargs.get('GUI', False) 
            cameras[name] = self.scene.add_camera(
                pos=cam_pos,
                lookat=lookat,
                res=res,
                fov=fov,
                GUI=GUI,
            ) 
        self._floating_camera = cameras.get(self.render_camera, None)
        self.cameras = cameras 
        self._recording = False
        self._recorded_frames = [] 
    
    def _set_camera(self, pos=None, lookat=None, fov=None, name='front'):
        if self.cameras.get(name, None) is None:
            print("Camera not initialized")
            return
        camera = self.cameras[name]
        if pos is not None or lookat is not None:
            camera.set_pose(pos=pos, lookat=lookat)
        if fov is not None:
            camera.set_params(fov=fov)
        return  

    def start_recording(self):
        self._recorded_frames = []
        if self.record_video:
            self._recording = True

    def _render_headless(self): 
        if self._recording and len(self._recorded_frames) < self.max_video_frames and self.record_video:
            # obj_pos = self.objects[self.object_names[0]].root_pos.cpu().numpy()[-1] 
            # import time
            # start = time.time()
            frame, depth_arr, seg_arr, normal_arr = self._floating_camera.render(segmentation=self.render_segmentation)
            if self.render_segmentation: 
                frame = np.concatenate(
                    [frame, seg_arr[:,:,None]], axis=-1
                )
            # end = time.time()
            # print(end-start)
            self._recorded_frames.append(frame)

    def get_recorded_frames(self, wait_for_max=True): 
        """ Stops recording if frames were yielded """
        if len(self._recorded_frames) == 0: 
            return None
        if wait_for_max and len(self._recorded_frames) < self.max_video_frames:
            return None
        frames = self._recorded_frames
        self._recorded_frames = []
        self._recording = False
        return frames
    
    def export_video(self, path, wait_for_max=True):
        if len(self._recorded_frames) == 0:
            return 
        frames = self.get_recorded_frames(wait_for_max=wait_for_max)
        if frames is not None:
            rgb_frames = [frame[:,:,:3] for frame in frames]
            if frames[0].shape[-1] == 4: # fill background with white
                mask_frames = [frame[:,:,3:] for frame in frames]
                # use mask frames to fill in white background 
                rgb_frames = np.array(rgb_frames)
                mask_frames = np.array(mask_frames)
                mask_frames = np.repeat(mask_frames, 3, axis=-1) 
                # max_id = mask_frames.max()
                ground_id = self.ground.idx
                rgb_frames[mask_frames == ground_id] = 255
                rgb_frames = np.clip(rgb_frames, 0, 255).astype(np.uint8)                 
                rgb_frames = [rgb_frames[i] for i in range(len(rgb_frames))]
            path = path + ".mp4" if not path.endswith(".mp4") else path
            from moviepy.editor import ImageSequenceClip 
            clip = ImageSequenceClip(rgb_frames, fps=int(1/self.dt/2))
            clip.write_videofile(path) 
        return frames
    
    def randomize(self, env_idxs=None):
        if not self.rand_cfg.get('randomize', False):
            return
        if self.rand_cfg.get('friction', False):
            friction_range = self.rand_cfg['friction']
            self._randomize_link_friction(friction_range=friction_range, env_idxs=env_idxs)
        
        if self.rand_cfg.get('com', False):
            com_range = self.rand_cfg['com']
            self._randomize_com_displacement(com_range=com_range, env_idxs=env_idxs)

        if self.rand_cfg.get('mass', False):
            mass_range = self.rand_cfg['mass']
            self._randomize_mass(mass_range=mass_range, env_idxs=env_idxs)

    def set_curriculum(self, epoch_num):
        self.epoch_num = epoch_num 
        verbose = epoch_num % 250 == 0
        reset_reward_tracker = False
        if self.use_curriculum:
            zero_gains, gains_decayed, reason = self.curriculum.set_curriculum(epoch_num)
            if verbose: 
                print(reason) 
            rew_weights_decayed = False
            if zero_gains:
                rew_weights_decayed = self.curriculum.decay_reward_weights(self.reward_module)
            reset_reward_tracker = (rew_weights_decayed or gains_decayed)
        return reset_reward_tracker
