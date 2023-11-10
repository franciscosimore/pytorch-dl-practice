from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

N_SAMPLES = 1000
N_CLASSES = 4
N_FEATURES = 2
RANDOM_SEED = 42

# Create multi-class data
X, y = make_blobs(n_samples=N_SAMPLES,
                  n_features=N_FEATURES,
                  centers=N_CLASSES,
                  cluster_std=1.5,
                  random_state=RANDOM_SEED)

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}") # [3 2 2 1 1]

print(f"Values for one sample of X: {X[0]}") # [-8.41339595  6.93516545]
print(f"Values for one sample of y: {y[0]}") # 3
print(f"Shapes for one sample of X: {X[0].shape}") # (2,)
print(f"Shapes for one sample of y: {y[0].shape}") # ()

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

# Split data into training and test set randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)