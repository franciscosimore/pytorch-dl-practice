from LinearRegression import LinearRegressionModel
import torch
from data import X_train, X_test, y_train, y_test
from torch import nn

# Create a random seed, for reproducibility
torch.manual_seed(42)

# Create an instance of the model (subclass of nn.Module)
model_0 = LinearRegressionModel()

# Setup a MAE loss function
loss_fn = nn.L1Loss()
# Setup a SGD optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Loops thorugh data (epochs)
epochs = 200

# Track
epoch_count = []
loss_values = []
test_loss_values = []

# 0. Loop
for epoch in range(epochs):
    # Set model to train mode: set gradient requirement to parameters with requires_grad=True
    model_0.train()
    y_pred = model_0(X_train) # 1. Forward pass
    loss = loss_fn(y_pred, y_train) # 2. Calculate loss
    optimizer.zero_grad() # 3. Zero the gradients of the optimizer
    loss.backward() # 4. Perform backpropagation on the loss
    optimizer.step() # 5. Progress/step the optimizer (gradient descent)
    # Set model to evaluation mode (turn off not needed settings)
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test) # Forward pass
        test_loss = loss_fn(test_pred, y_test) # Calculate the loss
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())

# Make predictions with model, inference_mode makes code faster when just doing inference
with torch.inference_mode():
    y_preds = model_0(X_test)
    print(y_test)
    print(y_preds)