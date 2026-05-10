"""
VLA Backbone: Vision-Language-Action Foundation Model
Based on Qwen2.5-VL / InternVL 2.5 architecture with action token integration
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, List, Optional, Tuple, Union


class VLABackbone(nn.Module):
    """
    Vision-Language-Action Backbone for Embodied-Aki
    
    This module integrates:
    - Visual encoder (ViT-based)
    - Language model (LLM backbone)
    - Action token embedding layer
    - Cross-modal attention mechanisms
    
    Architecture inspired by:
    - UnifoLM-VLA (Unitree)
    - HY-Embodied-0.5-X (Tencent)
    - OpenVLA
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        action_dim: int = 14,
        action_chunk_size: int = 8,
        freeze_vision_encoder: bool = False,
        freeze_llm: bool = False,
        use_lora: bool = True,
        lora_rank: int = 64,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.total_action_tokens = action_dim * action_chunk_size
        
        # Load pre-trained VLM
        print(f"Loading VLM backbone: {model_name}")
        self.vlm = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Freeze components if needed
        if freeze_vision_encoder:
            self._freeze_vision_encoder()
        if freeze_llm:
            self._freeze_llm()
        
        # Action token embedding layer
        # Maps continuous actions to discrete tokens for LLM integration
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.vlm.config.hidden_size),
        )
        
        # Action prediction head
        # Predicts action tokens from LLM hidden states
        self.action_head = nn.Sequential(
            nn.Linear(self.vlm.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim * action_chunk_size),
        )
        
        # LoRA adapter (optional, for efficient fine-tuning)
        if use_lora:
            self._setup_lora(lora_rank)
        
        print(f"VLA Backbone initialized with {self.get_num_parameters():,} parameters")
    
    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.vlm.visual.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")
    
    def _freeze_llm(self):
        """Freeze LLM parameters"""
        for name, param in self.vlm.language_model.named_parameters():
            if "norm" not in name:  # Keep normalization layers trainable
                param.requires_grad = False
        print("LLM backbone frozen")
    
    def _setup_lora(self, rank: int):
        """Setup LoRA adapters for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.vlm = get_peft_model(self.vlm, lora_config)
            print(f"LoRA adapters setup (rank={rank})")
        except ImportError:
            print("PEFT not installed, skipping LoRA setup")
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        images: torch.Tensor,
        instructions: List[str],
        actions: Optional[torch.Tensor] = None,
        proprioception: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of VLA Backbone
        
        Args:
            images: Batch of images [B, T, C, H, W] or [B, C, H, W]
            instructions: List of language instructions
            actions: Ground truth actions [B, chunk_size, action_dim] (for training)
            proprioception: Robot state [B, proprio_dim] (optional)
        
        Returns:
            Dictionary containing:
                - action_predictions: Predicted actions [B, chunk_size, action_dim]
                - loss: Training loss (if actions provided)
                - hidden_states: LLM hidden states for downstream modules
        """
        batch_size = images.shape[0]
        
        # Process visual input
        if images.dim() == 4:
            images = images.unsqueeze(1)  # Add temporal dimension
        
        B, T, C, H, W = images.shape
        
        # Prepare inputs for VLM processor
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Instruction: {instr}"},
                    ],
                }
            ]
            for instr in instructions
        ]
        
        # Process text and images
        text_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Encode through VLM
        vlm_outputs = self.vlm(
            pixel_values=images.view(B * T, C, H, W),
            input_texts=text_inputs,
            return_dict=True,
        )
        
        # Get LLM hidden states
        hidden_states = vlm_outputs.logits[:, -1, :]  # Last token representation
        
        # Integrate proprioception if provided
        if proprioception is not None:
            prop embed = self.action_embedder(proprioception)
            hidden_states = hidden_states + prop_embed
        
        # Predict actions
        action_predictions = self.action_head(hidden_states)
        action_predictions = action_predictions.view(
            batch_size, self.action_chunk_size, self.action_dim
        )
        
        # Compute loss if ground truth actions provided
        loss = None
        if actions is not None:
            loss = nn.functional.mse_loss(action_predictions, actions)
        
        return {
            "action_predictions": action_predictions,
            "loss": loss,
            "hidden_states": hidden_states,
            "vlm_outputs": vlm_outputs,
        }
    
    @torch.no_grad()
    def generate_actions(
        self,
        images: torch.Tensor,
        instructions: List[str],
        proprioception: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate actions autoregressively
        
        Args:
            images: Input images [B, T, C, H, W]
            instructions: Language instructions
            proprioception: Robot state (optional)
            temperature: Sampling temperature
            num_samples: Number of samples for diffusion
        
        Returns:
            Generated actions [B, chunk_size, action_dim]
        """
        self.eval()
        
        outputs = self.forward(
            images=images,
            instructions=instructions,
            proprioception=proprioception,
        )
        
        actions = outputs["action_predictions"]
        
        # Apply temperature scaling
        if temperature != 1.0:
            actions = actions / temperature
        
        return actions
    
    def save_pretrained(self, save_path: str):
        """Save model weights and configuration"""
        import json
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # Save config
        config = {
            "model_name": self.model_name,
            "action_dim": self.action_dim,
            "action_chunk_size": self.action_chunk_size,
            "total_action_tokens": self.total_action_tokens,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        s.path.join(load_path, "pytorch_model.bin"),
            map_location="cpu",
)
        model.load_state_dict(state_dict, strict=False)
        
        print(f"Model loaded from {load_path}")
        return model


# Convenience function for quick initialization
def create_vla_backbone(
    model_size: str = "7b",
    pretrained: bool = True,
    **kwargs,
) -> VLABackbone:
    """
    Create VLA backbone with preset configurations
    
    Args:
        model_size: Model size ("3b", "7b", "72b")
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for VLABackbone
    
    Returns:
        Initialized VLABackbone model
    """
    model_map = {
        "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    }
    
    model_name = model_map.get(model_size, model_map["7b"])
    
    return VLABackbone(
        model_name=model_name,
        **kwargs,
    )



































        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load model from checkpoint"""
        import json
        import os
        
        # Load config
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name=config["model_name"],
            action_dim=config["action_dim"],
            action_chunk_size=config["action_chunk_size"],
            **kwargs,
        )
        
        # Load weights
        state_dict = torch.load(
            o































































































































































































































































































