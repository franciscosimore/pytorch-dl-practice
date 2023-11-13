import requests
import zipfile
from pathlib import Path
import os

data_path = Path("ExperimentTracking/data/")
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