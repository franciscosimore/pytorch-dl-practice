from model import FoodVisionModelTinyVGG
import torch
from load_data import test_dataloader_custom, train_dataloader_custom_basic, train_dataloader_custom_flip, train_dataloader_custom_augmentation
from torch import nn
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary
from typing import Dict, List
import matplotlib.pyplot as plt

MODEL_NAME = "augmentation.pth"
MODEL_PATH = Path("FoodVision/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

train_dataloader = train_dataloader_custom_augmentation
test_dataloader = test_dataloader_custom

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a random seed, for reproducibility
torch.manual_seed(42)

# Create an instance of the model (subclass of nn.Module)
# TODO: do not hardcode output_shape, get it from something like 'len(class_names)'
model = FoodVisionModelTinyVGG(input_shape=3,
                               hidden_units=10,
                               output_shape=3)
model = model.to(device)

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

def train_step(model: nn.Module,
               data_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               get_accuracy,
               device: torch.device = device):
    """
    Perform a training step with model training on a DataLoader
    NOTE: Optimizer will update model's parameters once per batch (not just once per epoch)!
    """
    train_loss = 0 # Accumulate loss per batch
    train_acc = 0 # Accumulate accuracy per batch

    # Set model to train mode: set gradient requirement to parameters with requires_grad=True
    model.train()
    
    # 0. Loop through training batches, perform training steps, calculate the train loss per batch
    for batch, (X_train, y_train) in enumerate(data_loader): # (image, label)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        y_logits = model(X_train) # 1. Forward pass
        loss = loss_fn(y_logits, y_train) # 2. Calculate loss (per batch)
        train_loss += loss.item()
        y_pred = get_labels(y_logits)
        train_acc += get_accuracy(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad() # 3. Zero the gradients of the optimizer
        loss.backward() # 4. Perform backpropagation on the loss
        optimizer.step() # 5. Progress/step the optimizer (gradient descent)

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X_train)}/{len(train_dataloader.dataset)} samples")
    
    train_loss /= len(train_dataloader) # Average train loss per batch
    train_acc /= len(train_dataloader) # Average train accuracy per batch

    return train_loss, train_acc

def test_step(model: nn.Module,
              data_loader: DataLoader,
              loss_fn: nn.Module,
              get_accuracy,
              device: torch.device = device):
    """
    Perform a testing step with model testing on a DataLoader
    """
    test_loss, test_acc = 0, 0
    # Set model to evaluation mode (turn off not needed settings)
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader: # Per batch...
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_logits = model(X_test) # Forward pass
            test_pred = get_labels(test_logits)
            test_loss += loss_fn(test_logits, y_test).item() # Calculate the loss
            test_acc += get_accuracy(y_true=y_test, y_pred=test_pred)
        
        test_loss /= len(data_loader) # Average test loss per batch
        test_acc /= len(data_loader) # Average test accuracy per batch

        return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: torch.device = device):
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }
    start_time = timer()
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                data_loader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                get_accuracy=get_accuracy,
                                                device=device)
        test_loss, test_accuracy = test_step(model=model,
                                             data_loader=test_dataloader,
                                             loss_fn=loss_fn,
                                             get_accuracy=get_accuracy,
                                             device=device)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Train Accuracy: {train_accuracy:.2f} | Test loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}")
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)
    
    end_time = timer()
    print_train_time(start=start_time,
                     end=end_time,
                     device=str(next(model.parameters()).device))
    
    # Save PyTorch model state dict
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}.")

    # Run 'torchinfo' to get information about the shapes passing through the model
    summary(model, input_size=[1,3,64,64])

    return results

torch.manual_seed(42)
torch.cuda.manual_seed(42)
NUM_EPOCHS = 50
LR = 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LR)

model_results = train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS,
                      device=device)

# NOTE: Next should be part of evaluation...
def plot_loss_curves(results: Dict[str, List[float]]):
    """
    Plot training curves of a model's results dictionary
    """
    train_loss = results["train_loss"]
    train_accuracy = results["train_accuracy"]
    test_loss = results["test_loss"]
    test_accuracy = results["test_accuracy"]
    epochs = range(len(train_loss))
    plt.figure(figsize=(15,7))
    # Plot loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    # Plot accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('FoodVision/visualization/loss_curves_augmentation.png')

plot_loss_curves(model_results)