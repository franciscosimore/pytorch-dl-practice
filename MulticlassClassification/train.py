from model import BlobModel
import torch
from data import X_train, X_test, y_train, y_test
from torch import nn
from pathlib import Path

MODEL_NAME = "blobs_basic.pth"
MODEL_PATH = Path("MulticlassClassification/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

LR = 0.1
EPOCHS = 1000

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
model = BlobModel(input_features=2,output_features=4)
model = model.to(device)

# Setup a Cross Entropy Loss loss function
loss_fn = nn.CrossEntropyLoss()
# Setup a SGD optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

# Get predicction labels
def get_labels(y: torch.Tensor):
    """
    Get prediction labels from raw logits, for binary classification.
    1. Convert logits into prediction probabilities by using a sigmoid activation function.
    2. Convert prediction probabilities to prediction labels by taking the argmax().
    """
    y = torch.softmax(y, dim=1)
    y = y.argmax(dim=1)
    return y

# Accuracy (classes are balanced)
def get_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred)) * 100
    return accuracy

# Track
epoch_count = []
loss_values = []
test_loss_values = []
accuracy_values = []

# 0. Loops thorugh data (epochs)
for epoch in range(EPOCHS):
    # Set model to train mode: set gradient requirement to parameters with requires_grad=True
    model.train()
    y_logits = model(X_train) # 1. Forward pass
    loss = loss_fn(y_logits, y_train) # 2. Calculate loss
    y_pred = get_labels(y_logits)
    accuracy = get_accuracy(y_true=y_train, y_pred=y_pred)
    accuracy_values.append(accuracy)
    optimizer.zero_grad() # 3. Zero the gradients of the optimizer
    loss.backward() # 4. Perform backpropagation on the loss
    optimizer.step() # 5. Progress/step the optimizer (gradient descent)
    # Set model to evaluation mode (turn off not needed settings)
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test) # Forward pass
        test_pred = get_labels(test_logits)
        test_loss = loss_fn(test_logits, y_test) # Calculate the loss
        test_acc = get_accuracy(y_true=y_test, y_pred=test_pred)
    if epoch % 100 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Train Loss: {loss:.5f} | Train Accuracy: {accuracy:.2f} | Test loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}")

print(model.state_dict())

# Save PyTorch model state dict
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}.")