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

# Linear Regression Model with just a Linear Layer
class LinearRegressionModelWithLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Use PyTorch pre-built Linear Layer / Linear Transform / Probing Layer / Fully Connected Layer / Dense Layer
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)