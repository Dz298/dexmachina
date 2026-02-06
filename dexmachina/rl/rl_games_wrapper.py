"""
Reference implementation: 
Isaac Lab RL-Games wrapper: https://github.com/isaac-sim/IsaacLab/blob/931679641ee84e1b4175c15b48f9286d44ba9b04/source/isaaclab_rl/isaaclab_rl/rl_games.py#L51

"""

# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations
import torch
import numpy as np

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import gymnasium

from rl_games.common import env_configurations
from rl_games.common.vecenv import IVecEnv

"""
Vectorized environment wrapper.
"""


class RlGamesVecEnvWrapper(IVecEnv):
    """Wraps around Isaac Lab environment for RL-Games.

    This class wraps around the Isaac Lab environment. Since RL-Games works directly on
    GPU buffers, the wrapper handles moving of buffers from the simulation environment
    to the same device as the learning agent. Additionally, it performs clipping of
    observations and actions.

    For algorithms like asymmetric actor-critic, RL-Games expects a dictionary for
    observations. This dictionary contains "obs" and "states" which typically correspond
    to the actor and critic observations respectively.

    To use asymmetric actor-critic, the environment observations from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`
    must have the key or group name "critic". The observation group is used to set the
    :attr:`num_states` (int) and :attr:`state_space` (:obj:`gym.spaces.Box`). These are
    used by the learning agent in RL-Games to allocate buffers in the trajectory memory.
    Since this is optional for some environments, the wrapper checks if these attributes exist.
    If they don't then the wrapper defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.


    Reference:
        https://github.com/Denys88/rl_games/blob/master/rl_games/common/ivecenv.py
        https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
    """

    def __init__(self, env, rl_device: str, clip_obs: float, clip_actions: float, use_sil: bool = False,
                 world_model_trainer=None):
        """Initializes the wrapper instance.

        Args:
            env: The environment to wrap around.
            rl_device: The device on which agent computations are performed.
            clip_obs: The clipping value for observations.
            clip_actions: The clipping value for actions.
            world_model_trainer: Optional WorldModelTrainer for latent world model.

        Raises:
            ValueError: The environment is not inherited from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
            ValueError: If specified, the privileged observations (critic) are not of type :obj:`gym.spaces.Box`.
        """ 
        # initialize the wrapper
        self.env = env
        # store provided arguments
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.device
        self.use_sil = use_sil 
        
        # World model trainer (optional)
        self.world_model_trainer = world_model_trainer
        self._wm_step_count = 0
        
        # information for privileged observations
        if self.state_space is None:
            self.rlg_num_states = 0
        else:
            self.rlg_num_states = self.state_space.shape[0]
        self.obs_dict = None

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return (
            f"<{type(self).__name__}{self.env}>"
            f"\n\tObservations clipping: {self._clip_obs}"
            f"\n\tActions clipping     : {self._clip_actions}"
            f"\n\tAgent device         : {self._rl_device}"
            f"\n\tAsymmetric-learning  : {self.rlg_num_states != 0}"
        )

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    def get_obs_dict(self):
        return self.obs_dict

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        # note: rl-games only wants single observation space
        # policy_obs_space = self.unwrapped.single_observation_space["policy"]
        policy_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,)
        )
        # if not isinstance(policy_obs_space, gymnasium.spaces.Box):
        #     raise NotImplementedError(
        #         f"The RL-Games wrapper does not currently support observation space: '{type(policy_obs_space)}'."
        #         f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
        #         " and if you are nice, please send a merge-request."
        #     )
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_obs_space.shape)

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        # note: rl-games only wants single action space
        # action_space = self.unwrapped.single_action_space
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.num_actions,))
            
        # if not isinstance(action_space, gymnasium.spaces.Box):
        #     raise NotImplementedError(
        #         f"The RL-Games wrapper does not currently support action space: '{type(action_space)}'."
        #         f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
        #         " and if you are nice, please send a merge-request."
        #     )
        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self): # -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env #.unwrapped

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        """Returns the base environment simulation device."""
        return self.unwrapped.device

    @property
    def state_space(self) -> gym.spaces.Box | None:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        # note: rl-games only wants single observation space
        # critic_obs_space = self.unwrapped.single_observation_space.get("critic")
        # check if we even have a critic obs
        if self.use_sil:
            return self.observation_space
        return None
        if critic_obs_space is None:
            return None
        elif not isinstance(critic_obs_space, gymnasium.spaces.Box):
            raise NotImplementedError(
                f"The RL-Games wrapper does not currently support state space: '{type(critic_obs_space)}'."
                f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
                " and if you are nice, please send a merge-request."
            )
        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, critic_obs_space.shape)

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        """Returns the Gym spaces for the environment."""
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return torch_utils.set_seed(seed)
        # return self.unwrapped.seed(seed)

    def reset(self):  # noqa: D102
        obs_dict, _ = self.env.reset()
        
        # Initialize latent buffer with world model if enabled
        if self.world_model_trainer is not None:
            obs_without_latent = self.env.get_obs_without_latent()
            initial_latent = self.world_model_trainer.get_latent(obs_without_latent)
            self.env.update_latent(initial_latent)
            # Re-get observations with updated latent
            obs_dict = self.env.get_observations()
            # Reset world model buffers for new episode
            self.world_model_trainer.reset_buffers()
        
        # process observations and states
        return self._process_obs(obs_dict)

    def step(self, actions):  # noqa: D102
        # move actions to sim-device
        actions = actions.detach().clone().to(device=self._sim_device)
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        
        # Get observation without latent BEFORE step (for world model training)
        obs_before_step = None
        if self.world_model_trainer is not None:
            obs_before_step = self.env.get_obs_without_latent().clone()
        
        # perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # World model: collect data, train, and update latent
        if self.world_model_trainer is not None:
            self._update_world_model(obs_before_step, actions, extras)

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        if not self.env.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        # process observations and states
        obs_and_states = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }
        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs_and_states, rew, dones, extras
    
    def _update_world_model(self, obs_before_step, actions, extras):
        """Update world model: collect data, train, and update latent buffer."""
        # Get object state (decoder target)
        object_state = self.env.get_object_state()
        
        # Add transition to buffer
        self.world_model_trainer.add_transition(obs_before_step, actions, object_state)
        self._wm_step_count += 1
        
        # Train world model when we have enough data (skip in eval_only mode)
        if not getattr(self.world_model_trainer, 'eval_only', False) and self.world_model_trainer.has_enough_data():
            with torch.enable_grad():
                loss_dict = self.world_model_trainer.train_step()
            if loss_dict is not None:
                # Add world model metrics to extras for logging
                if "log" not in extras:
                    extras["log"] = {}
                extras["log"].update(loss_dict)
        
        # Update latent buffer with new encoding (using current obs after step)
        obs_after_step = self.env.get_obs_without_latent()
        new_latent = self.world_model_trainer.get_latent(obs_after_step)
        self.env.update_latent(new_latent)

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _process_obs(self, obs_dict) -> torch.Tensor | dict[str, torch.Tensor]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms.

        Args:
            obs_dict: The current observations from environment.

        Returns:
            If environment provides states, then a dictionary containing the observations and states is returned.
            Otherwise just the observations tensor is returned.
        """
        self.obs_dict = obs_dict
        # process policy obs
        obs = obs_dict["policy"]
        # clip the observations
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        # move the buffer to rl-device
        obs = obs.to(device=self._rl_device).clone()

        # check if asymmetric actor-critic or not
        if self.rlg_num_states > 0:
            # acquire states from the environment if it exists
            try:
                states = obs_dict["critic"]
            except AttributeError:
                raise NotImplementedError("Environment does not define key 'critic' for privileged observations.")
            # clip the states
            states = torch.clamp(states, -self._clip_obs, self._clip_obs)
            # move buffers to rl-device
            states = states.to(self._rl_device).clone()
            # convert to dictionary
            return {"obs": obs, "states": states}
        else:
            return obs


"""
Environment Handler.
"""


class RlGamesGpuEnv(IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    # TODO: Adding this for now but do we really need this?

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
        """
        self.env: RlGamesVecEnvWrapper = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):  # noqa: D102
        return self.env.step(action)

    def reset(self):  # noqa: D102
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        """Get number of agents in the environment.

        Returns:
            The number of agents in the environment.
        """
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        """Get the Gym spaces for the environment.

        Returns:
            The Gym spaces for the environment.
        """
        return self.env.get_env_info()

