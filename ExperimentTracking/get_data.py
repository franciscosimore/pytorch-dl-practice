import requests
import zipfile
from pathlib import Path
import os

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

get_data(data_path="ExperimentTracking/data/",
         image_path="pizza_steak_sushi",
         url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")

get_data(data_path="ExperimentTracking/data/",
         image_path="pizza_steak_sushi_20_percent",
         url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")

# def walk_through_dir(dir_path):
#     """
#     Walk through dir_path returning its contents.
#     """
#     for dirpath, dirnames, filenames in os.walk(dir_path):
#         print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")