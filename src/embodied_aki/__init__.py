"""
Embodied-Aki: A Hierarchical-End-to-End Hybrid VLA Framework for General-Purpose Embodied AI

This package implements a novel embodied AI architecture that combines:
- System 2 (Slow Thinking): LLM-based task planning and reasoning
- System 1 (Fast Thinking): Low-latency policy execution
- World Model: Environment prediction and imagination-based planning
- Cross-Model Reviewer: Adversarial safety and合理性 checking

References:
    - UnifoLM-VLA: https://github.com/unitreerobotics/unifolm-vla
    - HY-Embodied-0.5-X: https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X
    - Dreamer v3: https://github.com/danijar/dreamerv3
    - ARIS: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
"""

__version__ = "0.1.0"
__author__ = "Aki"
__description__ = "Hierarchical-End-to-End Hybrid VLA Framework for Embodied AI"

from .models.vla_backbone import VLABackbone
from .models.world_model import WorldModel
from .models.policy_head import DiffusionPolicyHead
from .models.planner import LLMPlanner
from .models.reviewer import CrossModelReviewer

__all__ = [
    "VLABackbone",
    "WorldModel", 
    "DiffusionPolicyHead",
    "LLMPlanner",
    "CrossModelReviewer",
]


























