"""
Latent World Model for online training alongside PPO.

Architecture:
- Encoder: policy observation -> latent
- Dynamics: (latent, action) -> next_latent  
- Decoder: latent -> object state (pose, velocities)

The encoder processes full policy observations, but the decoder only 
reconstructs object state to keep reconstruction loss inexpensive while
forcing the latent to carry object-centric information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Dict, Optional


def build_mlp(input_dim: int, output_dim: int, hidden_dims: list, activation: str = 'elu') -> nn.Sequential:
    """Build an MLP with the specified architecture."""
    if activation == 'elu':
        act_fn = nn.ELU
    elif activation == 'relu':
        act_fn = nn.ReLU
    elif activation == 'tanh':
        act_fn = nn.Tanh
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_fn())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class LatentWorldModel(nn.Module):
    """
    Latent world model with encoder, dynamics, and decoder.
    
    Args:
        obs_dim: Dimension of policy observation (without latent)
        action_dim: Dimension of action space
        latent_dim: Dimension of latent representation
        object_state_dim: Dimension of object state (decoder target)
        hidden_dims: Hidden layer dimensions for all MLPs
        activation: Activation function ('elu', 'relu', 'tanh')
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        object_state_dim: int = 14,  # pos(3) + quat(4) + dof(1) + lin_vel(3) + ang_vel(3)
        hidden_dims: list = [256, 256],
        activation: str = 'elu',
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.object_state_dim = object_state_dim
        
        # Encoder: obs -> latent
        self.encoder = build_mlp(obs_dim, latent_dim, hidden_dims, activation)
        
        # Dynamics: (latent, action) -> next_latent
        self.dynamics = build_mlp(latent_dim + action_dim, latent_dim, hidden_dims, activation)
        
        # Decoder: latent -> object_state
        self.decoder = build_mlp(latent_dim, object_state_dim, hidden_dims, activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to latent.
        
        Args:
            obs: (batch, obs_dim) or (batch, seq, obs_dim)
        Returns:
            latent: (batch, latent_dim) or (batch, seq, latent_dim)
        """
        return self.encoder(obs)
    
    def predict_next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent from current latent and action.
        
        Args:
            latent: (batch, latent_dim) or (batch, seq, latent_dim)
            action: (batch, action_dim) or (batch, seq, action_dim)
        Returns:
            next_latent: same shape as latent
        """
        x = torch.cat([latent, action], dim=-1)
        return self.dynamics(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to object state.
        
        Args:
            latent: (batch, latent_dim) or (batch, seq, latent_dim)
        Returns:
            object_state: (batch, object_state_dim) or (batch, seq, object_state_dim)
        """
        return self.decoder(latent)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode.
        
        Args:
            obs: (batch, obs_dim)
        Returns:
            latent: (batch, latent_dim)
            pred_object_state: (batch, object_state_dim)
        """
        latent = self.encode(obs)
        pred_object_state = self.decode(latent)
        return latent, pred_object_state
    
    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        object_state_seq: torch.Tensor,
        recon_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute world model loss over a sequence.
        
        Args:
            obs_seq: (batch, seq_len, obs_dim) - observation sequence
            action_seq: (batch, seq_len, action_dim) - action sequence
            object_state_seq: (batch, seq_len, object_state_dim) - ground truth object states
            recon_weight: weight for reconstruction loss
            dynamics_weight: weight for dynamics prediction loss
            
        Returns:
            total_loss: scalar loss
            loss_dict: dictionary with individual loss components
        """
        batch_size, seq_len, _ = obs_seq.shape
        
        # Encode all observations
        latents = self.encode(obs_seq)  # (batch, seq, latent_dim)
        
        # Reconstruction loss: decode latent -> predict object state
        pred_object_states = self.decode(latents)  # (batch, seq, object_state_dim)
        recon_loss = F.mse_loss(pred_object_states, object_state_seq)
        
        # Dynamics loss: predict next latent from current latent + action
        if seq_len > 1:
            # Use latent[t] and action[t] to predict latent[t+1]
            pred_next_latents = self.predict_next_latent(
                latents[:, :-1],  # (batch, seq-1, latent_dim)
                action_seq[:, :-1]  # (batch, seq-1, action_dim)
            )
            # Target is the encoded latent at t+1 (stop gradient to avoid collapse)
            target_next_latents = latents[:, 1:].detach()
            dynamics_loss = F.mse_loss(pred_next_latents, target_next_latents)
        else:
            # Use 0 * recon_loss so the tensor stays in the graph and has grad_fn
            dynamics_loss = recon_loss * 0.0
        
        # Total loss
        total_loss = recon_weight * recon_loss + dynamics_weight * dynamics_loss
        
        loss_dict = {
            'wm/total_loss': total_loss.item(),
            'wm/recon_loss': recon_loss.item(),
            'wm/dynamics_loss': dynamics_loss.item() if seq_len > 1 else 0.0,
        }
        
        return total_loss, loss_dict


class WorldModelTrainer:
    """
    Handles world model training alongside PPO.

    Dreamer-style: maintains a replay buffer of transitions (FIFO when full),
    trains on randomly sampled contiguous sequences from the buffer.
    Buffer is stored on CPU to save GPU memory; only the sampled batch is moved to device.
    """

    def __init__(
        self,
        world_model: LatentWorldModel,
        lr: float = 3e-4,
        recon_weight: float = 1.0,
        dynamics_weight: float = 1.0,
        rollout_length: int = 8,
        num_envs: int = 1,
        device: torch.device = torch.device('cuda'),
        eval_only: bool = False,
        replay_capacity: int = 5000,
    ):
        self.world_model = world_model
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.recon_weight = recon_weight
        self.dynamics_weight = dynamics_weight
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.device = device
        self.eval_only = eval_only  # if True, only run encoder for latent; no training
        self.replay_capacity = replay_capacity

        # Replay buffer (Dreamer-style): list of transitions, FIFO when full. Stored on CPU.
        self.obs_buffer = []
        self.action_buffer = []
        self.object_state_buffer = []
        self.done_buffer = []

        # Running statistics for logging
        self.total_updates = 0
        self.cumulative_loss = 0.0
        self.last_loss_dict = None

    def reset_buffers(self):
        """Clear replay buffers."""
        self.obs_buffer = []
        self.action_buffer = []
        self.object_state_buffer = []
        self.done_buffer = []

    def add_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        object_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Add a transition to the replay buffer. Stored on CPU to save GPU memory.
        When buffer exceeds replay_capacity, oldest transition is dropped (FIFO).

        Args:
            obs: (num_envs, obs_dim) - observation without latent
            action: (num_envs, action_dim)
            object_state: (num_envs, object_state_dim)
            done: (num_envs,) boolean tensor where True marks episode termination after this transition
        """
        self.obs_buffer.append(obs.detach().cpu())
        self.action_buffer.append(action.detach().cpu())
        self.object_state_buffer.append(object_state.detach().cpu())
        self.done_buffer.append(done.detach().cpu().to(dtype=torch.bool))
        if len(self.obs_buffer) > self.replay_capacity:
            self.obs_buffer.pop(0)
            self.action_buffer.pop(0)
            self.object_state_buffer.pop(0)
            self.done_buffer.pop(0)

    def has_enough_data(self) -> bool:
        """Check if we have at least one full sequence for training."""
        return len(self.obs_buffer) >= self.rollout_length

    def _sample_valid_sequence(self):
        """
        Sample a contiguous sequence that does not cross an episode boundary.

        Returns:
            Tuple of (obs_seq, action_seq, object_state_seq) on self.device, or None if no valid
            per-env sequence exists in the current replay buffer.
        """
        if not self.has_enough_data():
            return None

        n = len(self.obs_buffer)
        max_start = n - self.rollout_length
        candidate_starts = list(range(max_start + 1))
        random.shuffle(candidate_starts)

        for start in candidate_starts:
            end = start + self.rollout_length
            obs_seq = torch.stack(self.obs_buffer[start:end], dim=1)
            action_seq = torch.stack(self.action_buffer[start:end], dim=1)
            object_state_seq = torch.stack(self.object_state_buffer[start:end], dim=1)

            if self.rollout_length > 1:
                done_seq = torch.stack(self.done_buffer[start:end - 1], dim=1)
                valid_env_mask = ~done_seq.any(dim=1)
            else:
                valid_env_mask = torch.ones(obs_seq.shape[0], dtype=torch.bool)

            if valid_env_mask.any():
                return (
                    obs_seq[valid_env_mask].to(self.device),
                    action_seq[valid_env_mask].to(self.device),
                    object_state_seq[valid_env_mask].to(self.device),
                )

        return None

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Train world model on a randomly sampled contiguous sequence from the replay buffer
        (Dreamer-style: random sampling for diversity).

        Returns:
            loss_dict if training happened, None otherwise
        """
        if not self.has_enough_data():
            return None

        sampled = self._sample_valid_sequence()
        if sampled is None:
            return None

        obs_seq, action_seq, object_state_seq = sampled

        self.world_model.train()
        loss, loss_dict = self.world_model.compute_loss(
            obs_seq, action_seq, object_state_seq,
            recon_weight=self.recon_weight,
            dynamics_weight=self.dynamics_weight,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_updates += 1
        self.cumulative_loss += loss.item()
        self.last_loss_dict = loss_dict

        return loss_dict
    
    @torch.no_grad()
    def get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get latent encoding for observation (detached, no gradient).
        
        Args:
            obs: (num_envs, obs_dim) - observation without latent
        Returns:
            latent: (num_envs, latent_dim)
        """
        self.world_model.eval()
        return self.world_model.encode(obs)
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        avg_loss = self.cumulative_loss / max(self.total_updates, 1)
        return {
            'wm/avg_loss': avg_loss,
            'wm/total_updates': self.total_updates,
        }
    
    def save(self, path: str):
        """Save world model checkpoint."""
        torch.save({
            'model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path: str):
        """Load world model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint.get('total_updates', 0)


def load_world_model_for_eval(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    object_state_dim: int = 14,
    hidden_dims: list = [256, 256],
) -> Tuple[LatentWorldModel, int]:
    """
    Load world model from checkpoint for evaluation.
    Infers latent_dim from the saved encoder.
    Returns (world_model, latent_dim).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # Infer latent_dim from encoder output (last encoder layer weight shape is (latent_dim, in_features))
    encoder_weight_keys = [k for k in state_dict if k.startswith('encoder.') and k.endswith('.weight')]
    assert encoder_weight_keys, "No encoder weights in checkpoint"
    last_encoder_key = max(encoder_weight_keys, key=lambda x: int(x.split('.')[1]))
    latent_dim = int(state_dict[last_encoder_key].shape[0])
    world_model = LatentWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        object_state_dim=object_state_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    world_model.load_state_dict(state_dict)
    return world_model, latent_dim
