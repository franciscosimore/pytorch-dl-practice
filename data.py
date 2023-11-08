import torch

# Define parameters for the data
weight = 0.7
bias = 0.3

# Define paramaters for the data creation
start = 0
end = 1
step = 0.02

# Data creation
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias