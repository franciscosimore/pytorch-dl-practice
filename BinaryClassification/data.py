from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

N_SAMPLES = 1000

# Create data (circles)
X, y = make_circles(N_SAMPLES, noise=0.03, random_state=42)

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

print(f"Values for one sample of X: {X[0]}") # [0.75424625 0.23148074]
print(f"Values for one sample of y: {y[0]}") # 1
print(f"Shapes for one sample of X: {X[0].shape}") # (2,)
print(f"Shapes for one sample of y: {y[0].shape}") # ()

# Visualize data as a dataframe
circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.head(10))

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

# Split data into training and test set randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)