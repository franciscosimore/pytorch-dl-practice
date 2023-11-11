import torch
from PIL import Image
import pathlib
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, List
import random
import matplotlib.pyplot as plt

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Find class folder names in a target direcotry.
    """
    # Get class names (sorted) by scanning target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # Raise error if class names not found
    if not classes:
        raise FileNotFoundError(f"Could not find any classes in {directory}. Check file structure.")
    # Create a dictionary of index labels
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

class ImageFolderCustom(Dataset):
    """
    Subclass of 'torch.utils.data.Dataset', initialized with:
    - target directory: where to get the data from
    - transform: transformation to apply to the data
    With several attrbiutes:
    - path of images (target directory)
    - transform
    - classes: list of the target classes
    - class_to_idx: dict of target classes mapped to integer labels
    NOTE: __getitem__() MUST be overwritten, __len__() should also be overwritten...
    """
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    def load_image(self, index: int) -> Image.Image:
        """
        Open an image via a path and return it
        """
        image_path = self.paths[index]
        return Image.open(image_path)
    
    # Overwrite __len__() (advised/should)
    def __len__(self) -> int:
        """
        Return total number of samples
        """
        return len(self.paths)
    
    # Overwrite __getitem__() (MUST)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Return one sample of data, data and label (X, y)
        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # Expects path in format: data_folder/class_name/img.jpg
        class_idx = self.class_to_idx[class_name]

        if self.transform: # Optional
            return self.transform(img), class_idx # Return data, label (X, y)
        else:
            return img, class_idx # Return untrans formed data, label

train_transform_flip = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_transform_augmentation = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

train_transform_basic = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

data_path = Path("FoodVision/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

train_data_custom_basic = ImageFolderCustom(targ_dir=train_dir,
                                            transform=train_transform_basic)
train_data_custom_flip = ImageFolderCustom(targ_dir=train_dir,
                                           transform=train_transform_flip)
train_data_custom_augmentation = ImageFolderCustom(targ_dir=train_dir,
                                                   transform=train_transform_augmentation)

test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transform)

def display_random_images(dataset: Dataset,
                          classes: List[str],
                          n: int=10,
                          display_shape: bool=True,
                          seed: int=None):
    """
    Take a 'Dataset', class names, amount of images (capped to 10), random seed for reproducibility
    """
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n should not be larger than 10. n set to 10 by default.")
    
    if seed:
        random.seed(seed)
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16,8))

    for i, target_sample in enumerate(random_samples_idx):
        target_image, target_label = dataset[target_sample][0], dataset[target_sample][1]
        # Adjust Tensor dimensions for plotting
        target_image_adjust = target_image.permute(1,2,0) # [color_hannels,height,width] -> [height,width,color_hannels]
        
        plt.subplot(1, n, i+1)
        plt.imshow(target_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[target_label]}"
            if display_shape:
                title = title + f"\nShape: {target_image_adjust.shape}"
        plt.title(title)

# display_random_images(train_data_custom,classes=train_data_custom.classes,n=20)
# display_random_images(train_data,classes=class_names.classes,n=20)

BATCH_SIZE = 32
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0

# if __name__ == '__main__':
# Turn loaded images into DataLoader, it helps turn DataSets into iterables to then customize batch sizes
train_dataloader_custom_basic = DataLoader(dataset=train_data_custom_basic,
                                            batch_size=32,
                                            num_workers=NUM_WORKERS,
                                            shuffle=True)
train_dataloader_custom_flip = DataLoader(dataset=train_data_custom_flip,
                                            batch_size=32,
                                            num_workers=NUM_WORKERS,
                                            shuffle=True)
train_dataloader_custom_augmentation = DataLoader(dataset=train_data_custom_augmentation,
                                                    batch_size=32,
                                                    num_workers=NUM_WORKERS,
                                                    shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=32,
                                    num_workers=NUM_WORKERS,
                                    shuffle=False)

img, label = next(iter(train_dataloader_custom_basic))
print(f"Image shape: {img.shape} == [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")