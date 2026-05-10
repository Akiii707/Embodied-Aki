"""
World Model: Recurrent State-Space Model (RSSM) for Embodied-Aki
Based on Dreamer v3 architecture with enhancements for robotics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Configuration for World Model"""
    
    # RSSM parameters
    rssm_deterministic_dim: int = 512
    rssm_stochastic_dim: int = 30
    rssm_num_categories: int = 32
    rssm_hidden_dim: int = 512
    rssm_num_layers: int = 4
    
    # Encoder/Decoder parameters
    image_encoder_type: str = "cnn"  # "cnn" or "vit"
    image_decoder_type: str = "cnn"  # "cnn" or "vit"
    
    # Training parameters
    kl_free: float = 1.0
    kl_balance: float = 0.8
    kl_weight: float = 0.5
    grad_clip: float = 100.0
    
    # Horizon
    imagination_horizon: int = 15


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM)
    
    Core components:
    - Deterministic state (h_t): Captures long-term memory
    - Stochastic state (z_t): Captures uncertainty and multi-modality
    - Transition model: p(z_t | z_{t-1}, a_{t-1})
    - Representation model: q(z_t | h_t, obs_t)
    - Dynamics model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    
    Architecture inspired by:
    - Dreamer v3 (Google DeepMind)
    - PlaNet (Google)
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        self.config = config
        
        # Deterministic state dimension
        self.deterministic_dim = config.rssm_deterministic_dim
        
        # Stochastic state dimensions
        self.stochastic_dim = config.rssm_stochastic_dim
        self.num_categories = config.rssm_num_categories
        self.categorical_dim = self.stochastic_dim * self.num_categories
        
        # GRU for deterministic state dynamics
        self.gru_cell = nn.GRUCell(
            input_size=self.categorical_dim + config.rssm_hidden_dim,  # z + action
            hidden_size=self.deterministic_dim,
        )
        
        # Prior network (predicts z_t from h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deterministic_dim, config.rssm_hidden_dim),
            nn.LayerNorm(config.rssm_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.rssm_hidden_dim, self.categorical_dim),
        )
        
        # Posterior network (predicts z_t from h_t and observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.deterministic_dim + config.rssm_hidden_dim, config.rssm_hidden_dim),
            nn.LayerNorm(config.rssm_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.rssm_hidden_dim, self.categorical_dim * 2),  # mean and logvar
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize RSSM state"""
        return {
            "deterministic": torch.zeros(batch_size, self.deterministic_dim, device=device),
            "stochastic_mean": torch.zeros(batch_size, self.stochastic_dim, self.num_categories, device=device),
            "stochastic_logvar": torch.zeros(batch_size, self.stochastic_dim, self.num_categories, device=device),
            "stochastic_sample": torch.zeros(batch_size, self.stochastic_dim, self.num_categories, device=device),
        }
    
    def get_categorical_features(self, posterior: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get flattened categorical features from posterior"""
        shape = posterior["stochastic_mean"].shape
        return posterior["stochastic_mean"].reshape(shape[0], -1)
    
    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass through RSSM
        
        Args:
            actions: Action sequence [B, T, action_dim]
            observations: Observation embeddings [B, T, obs_dim]
            initial_state: Initial RSSM state (optional)
        
        Returns:
            posterior: Final posterior state
            trajectory: List of states at each timestep
        """
        batch_size, horizon = actions.shape[:2]
        device = actions.device
        
        # Initialize state
        if initial_state is None:
            state = self.initial_state(batch_size, device)
        else:
            state = initial_state
        
        trajectory = []
        
        # Unroll through time
        for t in range(horizon):
            action = actions[:, t]
            obs_embed = observations[:, t]
            
            # Update deterministic state with GRU
            categorical_flat = self.get_categorical_features(state).detach()
            gru_input = torch.cat([categorical_flat, action], dim=-1)
            state["deterministic"] = self.gru_cell(gru_input, state["deterministic"])
            
            # Compute prior
            prior_logits = self.prior_net(state["deterministic"])
            prior_logits = prior_logits.view(batch_size, self.stochastic_dim, self.num_categories)
            prior = self._categorical_distribution(prior_logits)
            
            # Compute posterior (using observation)
            posterior_input = torch.cat([state["deterministic"], obs_embed], dim=-1)
            posterior_logits = self.posterior_net(posterior_input)
            posterior_logits = posterior_logits.view(batch_size, self.stochastic_dim, self.num_categories * 2)
            
            mean, logvar = torch.chunk(posterior_logits, 2, dim=-1)
            posterior = {"mean": mean, "logvar": logvar}
            
            # Sample stochastic state (reparameterization trick)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sample = mean + std * eps
            
            # Update state
            state["stochastic_mean"] = mean
            state["stochastic_logvar"] = logvar
            state["stochastic_sample"] = z_sample
            
            trajectory.append(state.copy())
        
        return state, trajectory
    
    def _categorical_distribution(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create categorical distribution from logits"""
        return {"logits": logits}
    
    def compute_kl_divergence(
        self,
        posterior: Dict[str, torch.Tensor],
        prior: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # Simplified KL for categorical distributions
        posterior_logits = posterior.get("logits", posterior["mean"])
        prior_logits = prior.get("logits", torch.zeros_like(posterior_logits))
        
        posterior_probs = F.softmax(posterior_logits, dim=-1)
        prior_probs = F.softmax(prior_logits, dim=-1)
        
        # KL divergence
        kl = posterior_probs * (torch.log(posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8))
        kl = kl.sum(dim=-1).mean()
        
        return kl


class WorldModel(nn.Module):
    """
    Complete World Model for Embodied-Aki
    
    Components:
    - Visual encoder: Images -> latent observations
    - RSSM: Latent dynamics model
    - Reward predictor: r_t = r(h_t, z_t)
    - Observation decoder: reconstruct observations from latent
    - Value function: V(h_t, z_t) for planning
    
    Supports:
    - Imagination-based planning
    - Model-predictive control (MPC)
    - Data augmentation via simulation
    """
    
    def __init__(
        self,
        config: WorldModelConfig,
        action_dim: int = 14,
        proprio_dim: int = 10,
        image_channels: int = 3,
        image_size: int = 64,
    ):
        super().__init__()
        
        self.config = config
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        
        # Visual encoder (CNN-based)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, config.rssm_hidden_dim),
            nn.LayerNorm(config.rssm_hidden_dim),
            nn.ReLU(),
        )
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, config.rssm_hidden_dim),
            nn.LayerNorm(config.rssm_hidden_dim),
            nn.ReLU(),
        )
        
        # RSSM core
        self.rssm = RSSM(config)
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.rssm_deterministic_dim + config.rssm_stochastic_dim * config.rssm_num_categories, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        
        # Observation decoder (for reconstruction)
        self.image_decoder = nn.Sequential(
            nn.Linear(config.rssm_deterministic_dim + config.rssm_stochastic_dim * config.rssm_num_categories, 512),
            nn.ReLU(),
            nn.Unflatten(-1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 128, 5, stride=2),
            nn.ReLU(),
 















































































































































































































































































