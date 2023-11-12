#%%
import random
from pathlib import Path
from predictions import pred_and_plot_image
import torch
from load_model import model
from torch import nn

MODEL_NAME = "model.pth"
MODEL_PATH = Path("models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
RANDOM_SEED = 42

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(RANDOM_SEED)

model.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) # Load PyTorch model state dict
model.to(device)
loss_fn = nn.CrossEntropyLoss()

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
test_dir = image_path / "test"

num_images = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_path_sample = random.sample(population=test_image_path_list,
                                       k=num_images)

for image_path in test_image_path_sample:
    pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=["pizza","steak","sushi"],
                        image_size=(224,224))
# %%
