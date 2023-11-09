from torch import nn
import torch

# Linear Regression Model using PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize model parameters (randomly)
        self.weight = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float
        ))
        self.bias = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float
        ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defines the forward computation
        return self.weight * x + self.bias