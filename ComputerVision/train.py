from model import FashionMNISTModel
import torch
from data import class_names, train_dataloader, test_dataloader
from torch import nn
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm

MODEL_NAME = "basic.pth"
MODEL_PATH = Path("ComputerVision/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

LR = 0.1
EPOCHS = 3

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a random seed, for reproducibility
torch.manual_seed(42)

# Create an instance of the model (subclass of nn.Module)
model = FashionMNISTModel(input_shape=28*28,
                          hidden_units=10,
                          output_shape=len(class_names))
model = model.to(device)

# Setup a Cross Entropy loss function
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

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """
    Prints difference between start and end time
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Track
epoch_count = []
loss_values = []
test_loss_values = []

# NOTE: Optimizer will update model's parameters once per batch (not just once per epoch)!
# 0. Loop thorugh epochs
start_time = timer()
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-----")
    train_loss = 0 # Accumulate loss per batch
    train_acc = 0 # Accumulate accuracy per batch
    # 0. Loop through training batches, perform training steps, calculate the train loss per batch
    for batch, (X_train, y_train) in enumerate(train_dataloader): # (image, label)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        # Set model to train mode: set gradient requirement to parameters with requires_grad=True
        model.train()
        y_logits = model(X_train) # 1. Forward pass
        loss = loss_fn(y_logits, y_train) # 2. Calculate loss (per batch)
        train_loss += loss
        y_pred = get_labels(y_logits)
        train_acc += get_accuracy(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad() # 3. Zero the gradients of the optimizer
        loss.backward() # 4. Perform backpropagation on the loss
        optimizer.step() # 5. Progress/step the optimizer (gradient descent)

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X_train)}/{len(train_dataloader.dataset)} samples")
    
    train_loss /= len(train_dataloader) # Average train loss per batch
    train_acc /= len(train_dataloader) # Average train accuracy per batch
    
    # TESTING
    test_loss, test_acc = 0, 0
    # Set model to evaluation mode (turn off not needed settings)
    model.eval()
    
    with torch.inference_mode():
        for X_test, y_test in test_dataloader: # Per batch...
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_logits = model(X_test) # Forward pass
            test_pred = get_labels(test_logits)
            test_loss += loss_fn(test_logits, y_test) # Calculate the loss
            test_acc += get_accuracy(y_true=y_test, y_pred=test_pred)
        
        test_loss /= len(test_dataloader) # Average test loss per batch
        test_acc /= len(test_dataloader) # Average test accuracy per batch

    if epoch % 1 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f} | Test loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}")

end_time = timer()
print_train_time(start=start_time,
                 end=end_time,
                 device=str(next(model.parameters()).device))

# Save PyTorch model state dict
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}.")