import torch
from pathlib import Path
from model import BlobModel
from data import X_test, y_test
from torchmetrics import Accuracy

MODEL_NAME = "blobs_basic.pth"
MODEL_PATH = Path("MulticlassClassification/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model = BlobModel(input_features=2,output_features=4) # Instantiate a new instance of the model class
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) # Load PyTorch model state dict

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# Make predictions with model, inference_mode makes code faster when just doing inference
loaded_model.eval()
with torch.inference_mode():
    y_preds = loaded_model(X_test)
    y_preds = get_labels(y_preds)
    print(y_test)
    print(y_preds)
    
    # Instantiate Accuracy()
    torchmetric_accuracy = Accuracy(task="multiclass",num_classes=4).to(device)
    print(torchmetric_accuracy(y_preds, y_test))