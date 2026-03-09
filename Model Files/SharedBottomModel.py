import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBottomMLT(nn.Module):
    def __init__(self):
        super(SharedBottomMLT, self).__init__()
        # The Shared Bottom (Input 384 -> Output 128)
        self.shared = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU()
        )

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
        x1 = self.shared(x)

        engagement_x1 = self.engagement_head(x1)
        toxicity_x1 = self.toxicity_head(x1)

        return engagement_x1, toxicity_x1