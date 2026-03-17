import unittest

import torch
from torch import nn

from dexmachina.rl.latent_world_model import WorldModelTrainer
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper


class RecordingWorldModelTrainer:
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.transitions = []
        self.eval_only = False

    def add_transition(self, obs, action, object_state, done):
        self.transitions.append(
            {
                "obs": obs.clone(),
                "action": action.clone(),
                "object_state": object_state.clone(),
                "done": done.clone(),
            }
        )

    def has_enough_data(self):
        return False

    def get_latent(self, obs):
        return torch.full((obs.shape[0], self.latent_dim), 7.0, dtype=obs.dtype, device=obs.device)


class DummyEnv:
    def __init__(self):
        self.device = "cpu"
        self.render_mode = None
        self.obs_dim = 5
        self.num_actions = 2
        self.is_finite_horizon = True
        self._obs_without_latent = torch.tensor([[1.0, 2.0, 3.0]])
        self._object_state = torch.tensor([[10.0, 11.0]])
        self.last_latent = None

    def reset(self):
        return self.get_observations(), {}

    def get_obs_without_latent(self):
        return self._obs_without_latent

    def get_object_state(self):
        return self._object_state

    def update_latent(self, latent):
        self.last_latent = latent.clone()

    def get_observations(self):
        policy = torch.cat(
            [self._obs_without_latent, torch.zeros((1, self.obs_dim - self._obs_without_latent.shape[1]))],
            dim=-1,
        )
        return {"policy": policy, "critic": policy}

    def step(self, actions):
        self._obs_without_latent = torch.tensor([[4.0, 5.0, 6.0]])
        self._object_state = torch.tensor([[20.0, 21.0]])
        return (
            self.get_observations(),
            torch.tensor([1.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            {},
        )


class DummyWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_obs_seq = None
        self.last_action_seq = None
        self.last_object_state_seq = None

    def compute_loss(self, obs_seq, action_seq, object_state_seq, recon_weight=1.0, dynamics_weight=1.0):
        self.last_obs_seq = obs_seq.detach().clone()
        self.last_action_seq = action_seq.detach().clone()
        self.last_object_state_seq = object_state_seq.detach().clone()
        loss = self.weight * 0.0 + obs_seq.sum() * 0.0
        return loss, {"wm/total_loss": 0.0}


class LatentWorldModelTests(unittest.TestCase):
    def test_wrapper_uses_pre_step_object_state_and_done_flags(self):
        env = DummyEnv()
        trainer = RecordingWorldModelTrainer()
        wrapper = RlGamesVecEnvWrapper(
            env,
            rl_device="cpu",
            clip_obs=100.0,
            clip_actions=100.0,
            world_model_trainer=trainer,
        )

        wrapper.step(torch.tensor([[0.25, -0.5]]))

        self.assertEqual(len(trainer.transitions), 1)
        transition = trainer.transitions[0]
        self.assertTrue(torch.equal(transition["obs"], torch.tensor([[1.0, 2.0, 3.0]])))
        self.assertTrue(torch.equal(transition["object_state"], torch.tensor([[10.0, 11.0]])))
        self.assertTrue(torch.equal(transition["done"], torch.tensor([True])))
        self.assertTrue(torch.equal(env.last_latent, torch.tensor([[7.0, 7.0]])))

    def test_world_model_trainer_skips_envs_that_cross_episode_boundaries(self):
        world_model = DummyWorldModel()
        trainer = WorldModelTrainer(
            world_model=world_model,
            rollout_length=3,
            num_envs=2,
            device=torch.device("cpu"),
        )

        trainer.add_transition(
            obs=torch.tensor([[1.0], [101.0]]),
            action=torch.tensor([[11.0], [111.0]]),
            object_state=torch.tensor([[21.0], [121.0]]),
            done=torch.tensor([False, True]),
        )
        trainer.add_transition(
            obs=torch.tensor([[2.0], [102.0]]),
            action=torch.tensor([[12.0], [112.0]]),
            object_state=torch.tensor([[22.0], [122.0]]),
            done=torch.tensor([False, False]),
        )
        trainer.add_transition(
            obs=torch.tensor([[3.0], [103.0]]),
            action=torch.tensor([[13.0], [113.0]]),
            object_state=torch.tensor([[23.0], [123.0]]),
            done=torch.tensor([False, False]),
        )

        loss_dict = trainer.train_step()

        self.assertIsNotNone(loss_dict)
        self.assertTrue(torch.equal(world_model.last_obs_seq, torch.tensor([[[1.0], [2.0], [3.0]]])))
        self.assertTrue(torch.equal(world_model.last_action_seq, torch.tensor([[[11.0], [12.0], [13.0]]])))
        self.assertTrue(torch.equal(world_model.last_object_state_seq, torch.tensor([[[21.0], [22.0], [23.0]]])))

    def test_world_model_trainer_returns_none_when_no_valid_sequence_exists(self):
        world_model = DummyWorldModel()
        trainer = WorldModelTrainer(
            world_model=world_model,
            rollout_length=2,
            num_envs=1,
            device=torch.device("cpu"),
        )

        trainer.add_transition(
            obs=torch.tensor([[1.0]]),
            action=torch.tensor([[11.0]]),
            object_state=torch.tensor([[21.0]]),
            done=torch.tensor([True]),
        )
        trainer.add_transition(
            obs=torch.tensor([[2.0]]),
            action=torch.tensor([[12.0]]),
            object_state=torch.tensor([[22.0]]),
            done=torch.tensor([False]),
        )

        self.assertIsNone(trainer.train_step())
        self.assertIsNone(world_model.last_obs_seq)


if __name__ == "__main__":
    unittest.main()
