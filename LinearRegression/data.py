import torch

# Define parameters for the data
WEIGHT = 0.7
BIAS = 0.3

# Define paramaters for the data creation
START = 0
END = 1
STEP = 0.02

# Data creation
X = torch.arange(START, END, STEP).unsqueeze(dim=1)
y = WEIGHT * X + BIAS

# Data splitting
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]