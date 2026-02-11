import os  
import torch
import numpy as np
import genesis as gs 
from dexmachina.envs.robot import BaseRobot
from dexmachina.envs.object import ArticulatedObject
from dexmachina.envs.rewards import RewardModule
from dexmachina.envs.math_utils import matrix_from_quat
from dexmachina.envs.contacts import get_filtered_contacts
from dexmachina.envs.randomizations import RandomizationModule
from dexmachina.envs.curriculum import Curriculum 
from dexmachina.envs.maniptrans_curr import ManipTransCurriculum 
from typing import Dict, List, Tuple, Union
from collections import deque
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

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
        "valid_grasp_resample_attempts": 8,
        "valid_grasp_require_both_hands": False,
        "valid_grasp_opt_samples": 12,
        "valid_grasp_step_scale": 0.4,
        "valid_grasp_object_move_scale": 0.5,
        "valid_grasp_fallback_to_zero": True,
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
        self.valid_grasp_resample_attempts = env_cfg.get('valid_grasp_resample_attempts', 8)
        self.valid_grasp_require_both_hands = env_cfg.get('valid_grasp_require_both_hands', False)
        self.valid_grasp_opt_samples = env_cfg.get('valid_grasp_opt_samples', 12)
        self.valid_grasp_step_scale = env_cfg.get('valid_grasp_step_scale', 0.4)
        self.valid_grasp_object_move_scale = env_cfg.get('valid_grasp_object_move_scale', 0.5)
        self.valid_grasp_fallback_to_zero = env_cfg.get('valid_grasp_fallback_to_zero', True)

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
        if len(self.robots) == 0:
            return
        for side, robot in self.robots.items():
            demo_side = self.demo_data.get(side, dict())
            link_names = demo_side.get('collision_link_names', [])
            if len(link_names) == 0:
                continue
            link_idxs = []
            missing = []
            for name in link_names:
                idx = robot.link_name_to_local_idx.get(name)
                if idx is None:
                    missing.append(name)
                else:
                    link_idxs.append(idx)
            if len(link_idxs) == 0 or len(missing) > 0:
                if len(missing) > 0:
                    print(f"[valid-grasp] Missing collision links for {side}: {missing}")
                continue
            self.contact_link_meta[side] = dict(
                link_names=link_names,
                link_local_idxs=torch.tensor(link_idxs, dtype=torch.long, device=self.device),
                num_links=len(link_idxs),
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
        self.contact_link_meta = dict()
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
        if len(invalid_envs) == 0:
            return
        for env_idx in invalid_envs:
            targets = targets_map.get(env_idx, dict())
            success = self._optimize_grasp_state(env_idx, targets)
            if not success and self.valid_grasp_fallback_to_zero:
                self._reset_env_to_demo_anchor(env_idx)

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
        error_sum, count, side_counts = self._contact_alignment_error(env_idx, targets)
        if count == 0:
            return self._keypoint_grasp_distance(env_idx)
        if self.valid_grasp_require_both_hands and len(side_counts) > 1:
            sides_with_contact = sum(1 for v in side_counts.values() if v > 0)
            if sides_with_contact < len(side_counts):
                return float('inf')
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

    def _compute_contact_error_vectors(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        vectors = dict()
        for side, target in targets.items():
            meta = self.contact_link_meta.get(side, None)
            robot = self.robots.get(side, None)
            if meta is None or robot is None:
                continue
            link_idxs = meta['link_local_idxs']
            link_pos = robot.entity.get_links_pos()
            env_link_pos = link_pos[env_idx, link_idxs]
            pos_expanded = env_link_pos.unsqueeze(0)
            diff = target['positions'] - pos_expanded
            mask = target['valid'].unsqueeze(-1)
            masked = torch.where(mask, diff, torch.zeros_like(diff))
            count = target['valid'].sum().item()
            if count == 0:
                vectors[side] = torch.zeros(3, device=self.device)
            else:
                vectors[side] = masked.sum(dim=(0,1)) / max(count, 1)
        return vectors

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

    def _apply_contact_correction_step(self, env_idx: int, vectors: Dict[str, torch.Tensor], step_scale: float):
        env_idx = int(env_idx)
        combined_vec = torch.zeros(3, device=self.device)
        num_vecs = 0
        for side, vec in vectors.items():
            robot = self.robots.get(side, None)
            if robot is None or vec is None:
                continue
            wrist_xyz = robot.get_wrist_xyz_joints()
            if len(wrist_xyz) != 3:
                continue
            norm = torch.norm(vec)
            if norm > 1e-6:
                direction = vec / norm
            else:
                direction = torch.zeros_like(vec)
            delta = direction * (step_scale * norm)
            joint_targets = robot.dof_pos[env_idx].clone()
            for axis_idx, joint_idx in enumerate(wrist_xyz):
                joint_targets[joint_idx] += delta[axis_idx]
            robot.set_joint_position(joint_targets[None], env_idxs=[env_idx])
            combined_vec += vec
            num_vecs += 1
        if self.object is not None and num_vecs > 0:
            avg_vec = combined_vec / max(num_vecs, 1)
            obj_delta = -avg_vec * (self.valid_grasp_object_move_scale * step_scale)
            new_pos = self.object.root_pos[env_idx].clone() + obj_delta
            self.object.set_object_state(
                new_pos[None],
                self.object.root_quat[env_idx][None],
                self.object.dof_pos[env_idx][None],
                env_idxs=[env_idx],
            )

    def _optimize_grasp_state(self, env_idx: int, targets: Dict[str, Dict[str, torch.Tensor]]):
        if len(targets) == 0:
            return False
        env_idx = int(env_idx)
        snapshot = self._snapshot_env_state(env_idx)
        best_state = snapshot
        best_error, best_count, _ = self._contact_alignment_error(env_idx, targets)
        if best_count == 0:
            self._restore_env_state(env_idx, snapshot)
            return False
        if (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh:
            return True
        for attempt in range(max(1, int(self.valid_grasp_opt_samples))):
            step_scale = torch.rand(1).item() * max(self.valid_grasp_step_scale, 1e-3)
            vectors = self._compute_contact_error_vectors(env_idx, targets)
            self._apply_contact_correction_step(env_idx, vectors, step_scale)
            new_error, new_count, _ = self._contact_alignment_error(env_idx, targets)
            if new_count > 0 and new_error < best_error:
                best_error = new_error
                best_state = self._snapshot_env_state(env_idx)
                if (best_error / max(new_count, 1)) <= self.valid_grasp_contact_thresh:
                    self._restore_env_state(env_idx, best_state)
                    return True
            else:
                self._restore_env_state(env_idx, best_state)
        self._restore_env_state(env_idx, best_state)
        return (best_error / max(best_count, 1)) <= self.valid_grasp_contact_thresh

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
