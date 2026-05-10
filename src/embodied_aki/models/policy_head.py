"""
Policy Head: Action Generation Module for Embodied-Aki
Implements Diffusion Policy and ACT (Action Chunking with Transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class DiffusionPolicyConfig:
    """Configuration for Diffusion Policy"""
    
    # Diffusion parameters
    num_diffusion_steps: int = 100
    noise_schedule: str = "cosine"  # "linear" or "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Network architecture
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1


@dataclass
class ACTConfig:
    """Configuration for ACT Policy"""
    
    # Transformer parameters
    num_queries: int = 8  # Number of actions to predict
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1


class DiffusionPolicyHead(nn.Module):
    """
    Diffusion Policy for continuous action generation
    
    Key features:
    - Iterative denoising for action prediction
    - Handles multi-modal action distributions
    - Robust to distribution shifts
    
    Architecture inspired by:
    - Diffusion Policy (Stanford)
    - DiT (Diffusion Transformer)
    """
    
    def __init__(
        self,
        config: DiffusionPolicyConfig,
        action_dim: int = 14,
        action_chunk_size: int = 8,
        condition_dim: int = 512,
    ):
        super().__init__()
        
        self.config = config
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.total_action_dim = action_dim * action_chunk_size
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Condition encoder (for VLA backbone features)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(self.total_action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        
        # Denoising network (MLP-based)
        layers = []
        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ])
        self.denoising_net = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, self.total_action_dim),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Pre-compute noise schedule
        self.register_buffer("betas", self._get_betas())
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_betas(self) -> torch.Tensor:
        """Get noise schedule betas"""
        if self.config.noise_schedule == "linear":
            return torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.num_diffusion_steps,
            )
        elif self.config.noise_schedule == "cosine":
            # Cosine schedule (better for diffusion)
            steps = self.config.num_diffusion_steps + 1
            x = torch.linspace(0, steps, steps)
            alphas_cumprod = torch.cos(((x / steps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown noise schedule: {self.config.noise_schedule}")
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Extract values at specific timesteps"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *([1] * (len(shape) - 1))).to(t.device)
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through diffusion policy (denoising step)
        
        Args:
            noisy_actions: Noisy action sequence [B, action_dim * chunk_size]
            timestep: Diffusion timestep [B]
            condition: Condition features from VLA backbone [B, condition_dim]
        
        Returns:
            Predicted noise [B, action_dim * chunk_size]
        """
        batch_size = noisy_actions.shape[0]
        
        # Time embedding
        time_embed = self.time_embed(timestep.float().unsqueeze(-1))
        
        # Encode condition
        cond_embed = self.condition_encoder(condition)
        
        # Encode noisy actions
        action_embed = self.action_encoder(noisy_actions)
        
        # Concatenate and process
        combined = torch.cat([time_embed, cond_embed, action_embed], dim=-1)
        output = self.denoising_net(combined)
        
        # Predict noise
        predicted_noise = self.output_layer(output)
        
        return predicted_noise
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        batch_size: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample actions using reverse diffusion
        
        Args:
            condition: Condition features [B, condition_dim]
            batch_size: Batch size (if different from condition)
            temperature: Sampling temperature
        
        Returns:
            Generated actions [B, action_chunk_size, action_dim]
        """
        if batch_size is None:
            batch_size = condition.shape[0]
        
        device = condition.device
        
        # Initialize with random noise
        actions = torch.randn(
            batch_size,
            self.total_action_dim,
            device=device,
        )
        
        # Reverse diffusion loop
        for t in reversed(range(self.config.num_diffusion_steps)):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(actions, timestep, condition)
            
            # Compute alphas
            alpha_t = self._extract(1 - self.betas, timestep, actions.shape)
            alpha_t_prod = self._extract(
                torch.cumprod(1 - self.betas, dim=0),
                timestep,
                actions.shape,
            )
            alpha_t_prod_prev = self._extract(
                torch.cat([torch.ones(1, device=device), torch.cumprod(1 - self.betas, dim=0)[:-1]]),
                timestep,
                actions.shape,
            )
            
            # Compute mean
            sigma_t = temperature * torch.sqrt(
                (1 - alpha_t_prod_prev) / (1 - alpha_t_prod) * (1 - alpha_t)
            )
            
            # Sample next action
            noise = torch.randn_like(actions) if t > 0 else 0
            actions = (
                (actions - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                + sigma_t * noise
            )
        
        # Reshape to [B, chunk_size, action_dim]
        actions = actions.view(batch_size, self.action_chunk_size, self.action_dim)
        
        return actions
    
    def compute_loss(
        self,
        clean_actions: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss
        
        Args:
            clean_actions: Ground truth actions [B, chunk_size, action_dim]
            condition: Condition features [B, condition_dim]
        
        Returns:
            Diffusion loss (scalar)
        """
        batch_size = clean_actions.shape[0]
        device = clean_actions.device
        
        # Flatten actions
        clean_actions_flat = clean_actions.view(batch_size, -1)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.config.num_diffusion_steps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        
        # Add noise
        alpha_t_prod = self._extract(
            torch.cumprod(1 - self.betas, dim=0),
            timesteps,
            clean_actions_flat.shape,
        )
        noise = torch.randn_like(clean_actions_flat)
        noisy_actions = torch.sqrt(alpha_t_prod) * clean_actions_flat + to























































































































































































































































































