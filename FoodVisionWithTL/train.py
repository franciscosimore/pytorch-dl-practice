from load_model import model
import torch
from torch import nn
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from load_data import test_dataloader, train_dataloader
from engine import train
from typing import Dict, List

NUM_EPOCHS = 20
LR = 0.001

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_time = timer()

model_results = train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS,
                      device=device)

end_time = timer()

print(f"[INFO] Total training time: {(end_time-start_time):.3f} seconds.")

# NOTE: Next should be part of evaluation...
def plot_loss_curves(results: Dict[str, List[float]]):
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
    plt.savefig('FoodVisionWithTL/visualization/loss_curves.png')

plot_loss_curves(model_results)