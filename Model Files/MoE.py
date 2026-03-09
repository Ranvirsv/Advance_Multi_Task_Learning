import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        pass

    def forward(self, x):
        pass

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        pass

    def forward(self, x):
        pass

class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = trained_experts

        for expert in trained_experts:
            for param in expert.parameters():
                param.requires_grad = False

        num_experts = len(trained_experts)
        input_dim = trained_experts[0].linear1.in_features
        self.gating_network = GatingNetwork(input_dim, num_experts)

        
    def forward(self, x):
        pass