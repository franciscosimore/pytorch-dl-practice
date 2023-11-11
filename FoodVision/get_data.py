import requests
import zipfile
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data_path = Path("FoodVision/data/")
image_path = data_path / "pizza_steak_sushi"
zip_file = "pizza_steak_sushi.zip"

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
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(request.content)

# TODO: do not unzip if already unzipped...
with zipfile.ZipFile(data_path / zip_file, "r") as zip_ref:
    print(f"Unzipping '{zip_file}'...")
    zip_ref.extractall(image_path)

def walk_through_dir(dir_path):
    """
    Walk through dir_path returning its contents.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_through_dir(image_path)

# Next, ImageFolder is used directly but a custom ImageFolder is created for this case by instantiating a Datset subclass
# Also, there's a plot_transformed_images function
# NOTE: Following code is IGNORED!
quit()

train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 1. Turn target data into tensors (numerical representation of images in this case)
# 2. Turn tensors into a `torch.utils.data.DataSet` and, subsequently, a `torch.utils.data.DataLoader`

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=None):
    """
    Select random images from a path of images and loads/transforms them,
    then plots the original vs the transformed version
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)
            # PyTorch default is [C, H, W] but Matplotlib is [H, W, C]... we have to permute!
            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

# plot_transformed_images(image_paths=image_path_list,
#                         transform=data_transform,
#                         n=3,
#                         seed=42)

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                  transform=data_transform,
                                  target_transform=None)

class_names = train_data.classes
class_dict = train_data.class_to_idx

# When using multiple workers (num_workers > 0), PyTorch uses multiprocessing,
# and there are certain restrictions on what can be done in the child processes.
# Ensure that the code creating and using the DataLoader is only executed in the main process, avoiding RuntimeError.
if __name__ == '__main__':
    # Turn loaded images into DataLoader, it helps turn DataSets into iterables to then customize batch sizes
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=32,
                                  num_workers=os.cpu_count(),
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=32,
                                 num_workers=os.cpu_count(),
                                 shuffle=False)
    
    img, label = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} == [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")