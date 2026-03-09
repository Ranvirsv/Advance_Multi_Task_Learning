import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.MoE import GatingNetwork, ExpertNetwork

class BasicMMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()
        
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertNetwork(input_dim, output_dim) for _ in range(self.num_experts)
            ]
        )

        self.engagement_gate = GatingNetwork(input_dim, self.num_experts)
        self.toxicity_gate = GatingNetwork(input_dim, self.num_experts)

        # 2. Task Head A: Engagement (Input 128 -> Output 1)
        self.engagement_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 2. Task Head B: Toxicity (Input 128 -> Output 1)
        self.toxicity_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        expert_weight_engagement = self.engagement_gate(x)
        expert_weight_toxicity = self.toxicity_gate(x)
        
        expert_out_list = [
            expert(x) for expert in self.experts
        ]
        expert_out = torch.stack(expert_out_list, dim=1)
        expert_engagement_out = torch.sum(expert_weight_engagement @ expert_out, dim=1)
        expert_toxicity_out = torch.sum(expert_weight_toxicity @ expert_out, dim=1)

        engagement_out = self.engagement_head(expert_engagement_out)
        toxicity_out = self.toxicity_head(expert_toxicity_out)
        return engagement_out, toxicity_out