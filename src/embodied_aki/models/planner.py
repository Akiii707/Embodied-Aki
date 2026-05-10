"""
LLM Planner: Task Planning and Decomposition Module
Based on HY-Embodied-0.5-X MoT architecture with System 2 reasoning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PlannerConfig:
    """Configuration for LLM Planner"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_context_length: int = 4096
    max_plan_length: int = 512
    
    # Planning parameters
    num_subgoals_max: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Reasoning mode
    enable_thinking: bool = True
    enable_self_correction: bool = True


@dataclass
class Subgoal:
    """Represents a subgoal in the plan"""
    id: int
    description: str
    preconditions: List[str]
    expected_outcome: str
    success_criteria: str


@dataclass
class Plan:
    """Complete task plan"""
    task_description: str
    subgoals: List[Subgoal]
    estimated_steps: int
    confidence_score: float


class LLMPlanner(nn.Module):
    """
    LLM-based Task Planner for Embodied-Aki
    
    Key features:
    - Hierarchical task decomposition
    - Long-horizon planning with constraint checking
    - Self-correction and refinement
    - Integration with world model for feasibility validation
    
    Architecture inspired by:
    - HY-Embodied-0.5-X (Tencent)
    - SayCan (Google)
    - Code as Policies
    """
    
    def __init__(self, config: PlannerConfig, device: Optional[str] = None):
        super().__init__()
        
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load LLM backbone
        print(f"Loading LLM planner: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Freeze LLM parameters (optional)
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Value function for feasibility estimation
        self.value_function = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        print(f"LLM Planner initialized on {self.device}")
    
    def _create_planning_prompt(
        self,
        task_description: str,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> str:
        """Create structured prompt for task planning"""
        
        system_prompt = """You are an expert robot task planner. Your job is to decompose complex manipulation tasks into clear, executable subgoals.

Guidelines:
1. Break down the task into 3-10 sequential subgoals
2. Each subgoal should be atomic and verifiable
3. Consider physical constraints and object affordances
4. Include preconditions and success criteria for each subgoal
5. Think step-by-step and validate feasibility

Format your response as JSON:
{
    "task": "original task description",
    "subgoals": [
        {
            "id": 1,
            "description": "clear action description",
            "preconditions": ["list of required conditions"],
            "expected_outcome": "what should be true after this step",
            "success_criteria": "how to verify completion"
        }
    ],
    "estimated_steps": number,
    "confidence": 0.0-1.0
}"""
        
        user_prompt = f"Task: {task_description}\n"
        
        if context:
            user_prompt += f"\nContext: {context}\n"
        
        if constraints:
            user_prompt += f"\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        user_prompt += "\n\nPlease create a detailed plan:"
        
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    @torch.no_grad()
    def plan(
        self,
        task_description: str,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        world_state: Optional[Dict] = None,
    ) -> Plan:
        """
        Generate a hierarchical task plan
        
        Args:
            task_description: High-level task description
            context: Additional context information
            constraints: List of constraints to respect
            world_state: Current world state (for feasibility check)
        
        Returns:
            Complete Plan object
        """
        # Create prompt
        prompt = self._create_planning_prompt(task_description, context, constraints)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_context_length,
            truncation=True,
        ).to(self.device)
        
        # Generate plan
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=self.config.max_plan_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse JSON response (simplified)
        plan = self._parse_plan_response(generated_text, task_description)
        
        # Validate plan feasibility
        if world_state is not None:
            plan = self._validate_plan_feasibility(plan, world_state)
        
        # Self-correction if enabled
        if self.config.enable_self_correction:
            plan = self._self_correct_plan(plan, task_description, context)
        
        return plan
    
    def _parse_plan_response(self, response: str, task_description: str) -> Plan:
        """Parse LLM response into Plan object"""
        import json
        import re
        
        # Try to extract JSON from response
        try:
            # Find JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan_dict = json.loads(json_match.group())
                
                subgoals = [
                    Subgoal(
                        id=s.get("id", i),
                        description=s.get("description", ""),
                        preconditions=s.get("preconditions", []),
                        expected_outcome=s.get("expected_outcome", ""),
                        success_criteria=s.get("success_criteria", ""),
                    )
                    for i, s in enumerate(plan_dict.get("subgoals", []))
                ]
                
                return Plan(
                    task_description=task_description,
                    subgoals=subgoals,
                    estimated_steps=plan_dict.get("estimated_steps", len(subgoals)),
                    confidence_score=plan_dict.get("confidence", 0.5),
                )
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
        
        # Fallback: create simple plan from text
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        subgoals = [
            Subgoal(
                id=i+1,
                description=line.replace(f"{i+1}.", "").replace("-", "").strip(),
                preconditions=[],
                expected_outcome="",
                success_criteria="",
            )
            for i, line in enumerate(lines[:self.config.num_subgoals_max])
        ]
        
        return Plan(
            task_description=task_description,
            subgoals=subgoals,
            estimated_steps=len(subgoals),
            confidence_score=0.5,
        )
    
    def _validate_plan_feasibility(self, plan: Plan, world_state: Dict) -> Plan:
        """Validate plan feasibility using value function"""
        
        # Encode plan into features
        plan_features = self._encode_plan(pl





























































































































































































































































