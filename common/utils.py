"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import os
import zipfile
import requests
import matplotlib.pyplot as plt
from typing import Dict, List

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.
    
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general PyTorch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA PyTorch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def get_data(data_path: str,
             image_path: str,
             url: str):
    zip_file = image_path + ".zip"
    data_path = Path(data_path)
    image_path = data_path / image_path

    if image_path.is_dir():
        print(f"'{image_path}' directory already exists... skipping creation.")
    else:
        print(f"{image_path} does not exist, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(data_path / zip_file):
        print(f"'{zip_file}' already exists... skipping download.")
    else:
        with open(data_path / zip_file, "wb") as f:
            print(f"'{zip_file}' does not exist, downloading...")
            request = requests.get(url)
            f.write(request.content)

    # TODO: do not unzip if already unzipped...
    with zipfile.ZipFile(data_path / zip_file, "r") as zip_ref:
        print(f"Unzipping '{zip_file}'...")
        zip_ref.extractall(image_path)

def plot_loss_curves(results: Dict[str, List[float]],
                     save_directory: str):
    """
    Plot training curves of a model's results dictionary
    """
    train_loss = results["train_loss"]
    train_accuracy = results["train_acc"]
    test_loss = results["test_loss"]
    test_accuracy = results["test_acc"]
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
    plt.savefig(save_directory)