import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.Softmax(self.gate(x))

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.linear(x))

class BasicMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, trained_experts=None):
        super().__init__()
        
        self.num_experts = num_experts
        
        if trained_experts:
            for expert in trained_experts:
                for param in expert.parameters():
                    param.requires_grad = False

        self.experts = nn.ModuleList(
            [
                expert_model.expert for expert_model in trained_experts
            ] if trained_experts else [
                ExpertNetwork(input_dim, output_dim) for _ in range(self.num_experts)
            ]
        )

        self.gating_network = GatingNetwork(input_dim, self.num_experts)

        # 2. Task Head A: Engagement (Input 128 -> Output 1)
        self.engagement_head = trained_experts[0].engagement_head if trained_experts else nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 2. Task Head B: Toxicity (Input 128 -> Output 1)
        self.toxicity_head = trained_experts[1].toxicity_head if trained_experts else nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        expert_weight = self.gating_network(x)
        expert_out_list = [
            expert(x) for expert in self.experts
        ]
        expert_out = torch.stack(expert_out_list, dim=1)
        model_out = torch.sum(expert_weight @ expert_out, dim=1)

        engagement_out = self.engagement_head(model_out)
        toxicity_out = self.toxicity_head(model_out)
        return engagement_out, toxicity_out