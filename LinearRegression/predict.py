import torch
from pathlib import Path
from model import LinearRegressionModel, LinearRegressionModelWithLayer
from data import X_test, y_test

MODEL_NAME = "layered.pth"
MODEL_PATH = Path("LinearRegression/models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model = LinearRegressionModelWithLayer() # Instantiate a new instance of the model class
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) # Load PyTorch model state dict

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Make predictions with model, inference_mode makes code faster when just doing inference
loaded_model.eval()
with torch.inference_mode():
    y_preds = loaded_model(X_test)
    print(y_test)
    print(y_preds)