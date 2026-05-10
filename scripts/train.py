#!/usr/bin/env python3
"""
Embodied-Aki Training Script
Main entry point for training VLA models with hierarchical architecture
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embodied_aki.models import (
    VLABackbone,
    WorldModel,
    DiffusionPolicyHead,
    ACTPolicyHead,
    LLMPlanner,
    CrossModelReviewer,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Embodied-Aki model")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, default="7b", choices=["3b", "7b", "72b"])
    parser.add_argument("--policy-type", type=str, default="diffusion", choices=["diffusion", "act"])
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--action-chunk-size", type=int, default=8)
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Checkpoint configuration
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume-from", type=str, default=None)
    
    # Distributed training
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--use-mixed-precision", action="store_true")
    
    # Review and validation
    parser.add_argument("--enable-reviewer", action="store_true")
    parser.add_argument("--safety-threshold", type=float, default=0.8)
    
    return parser.parse_args()


class EmbodiedAkiTrainer:
    """Main trainer class for Embodied-Aki model"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Training on device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if args.use_mixed_precision else None
        
        # Initialize reviewer (optional)
        self.reviewer = None
        if args.enable_reviewer:
            self.reviewer = CrossModelReviewer(
                reviewer_model_name=f"Qwen/Qwen2.5-{args.model_size}-Instruct"
            ).to(self.device)
    
    def _init_models(self):
        """Initialize all model components"""
        print("Initializing models...")
        
        # VLA Backbone
        self.vla_backbone = VLABackbone(
            model_name=f"Qwen/Qwen2.5-VL-{self.args.model_size}-Instruct",
            action_dim=self.args.action_dim,
            action_chunk_size=self.args.action_chunk_size,
            use_lora=True,
        ).to(self.device)
        
        # World Model
        self.world_model = WorldModel(
            action_dim=self.args.action_dim,
            proprio_dim=10,
        ).to(self.device)
        
        # Policy Head
        if self.args.policy_type == "diffusion":
            self.policy_head = DiffusionPolicyHead(
                action_dim=self.args.action_dim,
                action_chunk_size=self.args.action_chunk_size,
                condition_dim=self.vla_backbone.vlm.config.hidden_size,
            ).to(self.device)
        else:
            self.policy_head = ACTPolicyHead(
                action_dim=self.args.action_dim,
                condition_dim=self.vla_backbone.vlm.config.hidden_size,
            ).to(self.device)
        
        # LLM Planner (frozen during policy training)
        self.planner = LLMPlanner(
            model_name=f"Qwen/Qwen2.5-{self.args.model_size}-Instruct"
        ).to(self.device)
        
        # Freeze planner during policy training
        for param in self.planner.parameters():
            param.requires_grad = False
        
        print("Models initialized successfully")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        # Collect trainable parameters
        trainable_params = []
        trainable_params.extend(self.vla_backbone.parameters())
        trainable_params.extend(self.world_model.parameters())
        trainable_params.extend(self.policy_head.parameters())
        
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        from transformers import get_linear_schedule_with_warmup
        
        total_steps = self.args.epochs * 1000  # Approximate
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps,
        )
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.vla_backbone.train()
        self.world_model.train()
        self.policy_head.train()
        
        total_loss = 0
        total_vla_loss = 0
        total_world_loss = 0
        total_policy_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            images = batch["images"].to(self.device)
            instructions = batch["instructions"]
            actions = batch["actions"].to(self.device)
            proprioception = batch.get("proprioception", None)
            if proprioception is not None:
                proprioception = proprioception.to(self.device)
            
            # Forward pass through VLA backbone
            with autocast(enabled=self.args.use_mixed_precision):
                vla_outputs = self.vla_backbone(
                    images=images,
                    instructions=instructions,
                    actions=actions,
                    proprioception=proprioception,
                )
                
                vla_loss = vla_outputs["loss"]
                
                # Forward pass through world model
                world_outputs = self.world_model(
                    images=images,
                    actions=actions,
                    proprioception=proprioception,
                )
                
                world_loss = world_outputs["total_loss"]
                
                # Forward pass through policy head
                condition = vla_outputs["hidden_states"]
                
                if self.args.policy_type == "diffusion":
                    policy_loss = self.policy_head.compute_loss(
                        clean_actions=actions,
                        condition=condition,
                    )
                else:
                    policy_loss = self.policy_head.compute_loss(
                        condition=condition,
                        target_actions=actions,
                    )
                
                # Total loss
                total_batch_loss = vla_loss + world_loss + policy_loss
                
                # Apply reviewer penalty if enabled
                if self.reviewer is not None:
                    with torch.no_grad():
                        review_report = self.reviewer.review(
                            actions=vla_outputs["action_predictions"].detach(),
                            plan_description=instructions[0],
                            instructions=instructions[0],
                        )
                        
                        if review_report.verdict.value in ["unsafe", "rejected"]:
                            # Add penalty for unsafe actions
                            safety_penalty = (1 - review_report.safety_score) * 0.1
                            total_batch_loss = total_batch_loss + safety_penalty
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimiz




































































































































































































































