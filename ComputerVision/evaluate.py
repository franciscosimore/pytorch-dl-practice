import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model import FashionMNISTModel
from data import class_names, test_dataloader
from torch import nn
from tqdm.auto import tqdm

MODEL_NAME = "basic.pth"
MODEL_PATH = Path("ComputerVision/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
RANDOM_SEED = 42

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(RANDOM_SEED)

loaded_model = FashionMNISTModel(input_shape=28*28,hidden_units=10,output_shape=len(class_names)) # Instantiate a new instance of the model class
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) # Load PyTorch model state dict
loaded_model.to(device)
loss_fn = nn.CrossEntropyLoss()

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

def eval_model(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               get_accuracy):
    """
    Return a dictionary containing the results of model predicting on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X = X.to(device)
            y = y.to(device)
            test_logits = model(X) # Forward pass
            test_pred = get_labels(test_logits)
            loss += loss_fn(test_logits, y) # Calculate the loss
            acc += get_accuracy(y_true=y, y_pred=test_pred)
        # Scale loss and accuracy to find the average loss and accuracy (per batch)
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_accuracy": acc
    }

# Accuracy (classes are balanced)
def get_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred)) * 100
    return accuracy

# Evaluate
model_results = eval_model(model=loaded_model,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           get_accuracy=get_accuracy)
print(model_results)