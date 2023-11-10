from model import LinearRegressionModel, LinearRegressionModelWithLayer
import torch
from data import X_train, X_test, y_train, y_test
from torch import nn
from pathlib import Path

MODEL_NAME = "basic.pth"
MODEL_PATH = Path("LinearRegression/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

LR = 0.01
EPOCHS = 100

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Put data on target device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Create a random seed, for reproducibility
torch.manual_seed(42)

# Create an instance of the model (subclass of nn.Module)
model = LinearRegressionModel()
model = model.to(device)

# Setup a MAE loss function
loss_fn = nn.L1Loss()
# Setup a SGD optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

# Track
epoch_count = []
loss_values = []
test_loss_values = []

# 0. Loops thorugh data (epochs)
for epoch in range(EPOCHS):
    # Set model to train mode: set gradient requirement to parameters with requires_grad=True
    model.train()
    y_pred = model(X_train) # 1. Forward pass
    loss = loss_fn(y_pred, y_train) # 2. Calculate loss
    optimizer.zero_grad() # 3. Zero the gradients of the optimizer
    loss.backward() # 4. Perform backpropagation on the loss
    optimizer.step() # 5. Progress/step the optimizer (gradient descent)

    # Set model to evaluation mode (turn off not needed settings)
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test) # Forward pass
        test_loss = loss_fn(test_pred, y_test) # Calculate the loss
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test loss: {test_loss}")
        print(model.state_dict())

# Save PyTorch model state dict
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}.")