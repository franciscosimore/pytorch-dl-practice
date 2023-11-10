from torch import nn
import torch

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.relu = nn.ReLU() # Non-linear activation function, max(0,x)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.layer_1(x))))