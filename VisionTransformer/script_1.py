import sys, os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from common import data_setup, engine
from common.utils import set_seeds, save_model, get_data, plot_loss_curves
from PIL import Image
from patchembedding import PatchEmbedding

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

get_data(data_path="VisionTransformer/data/",
         image_path="pizza_steak_sushi",
         url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")

# Setup directories
train_dir = "VisionTransformer/data/pizza_steak_sushi/train"
test_dir = "VisionTransformer/data/pizza_steak_sushi/test"

IMG_SIZE = 224

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
    ])

BATCH_SIZE = 32

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=BATCH_SIZE,
                                                                               num_workers=0)

PATCH_SIZE = 16
assert IMG_SIZE & PATCH_SIZE == 0, "[ERROR] Image size must be dividible by the patch size!"
NUM_PATCHES = IMG_SIZE / PATCH_SIZE
print(f"Patch size: {PATCH_SIZE} pixels x {PATCH_SIZE} pixels\
      \nNumber of patches per row: {NUM_PATCHES}\
      \nNumber of patches per column: {NUM_PATCHES}\
      \nTotal patches: {NUM_PATCHES*NUM_PATCHES}")

# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# Change image shape to be compatible with matplotlib (color_channels, height, width) -> (height, width, color_channels)
image_permuted = image.permute(1, 2, 0)

# We can create the patches with a Conv2D...
# ... by setting kernel size and stride value to the patch size!

set_seeds()
patchify = PatchEmbedding(in_channels=3,
                          patch_size=PATCH_SIZE,
                          embedding_dim=768)

print(f"Input image size: {image.unsqueeze(0).shape}")

image = patchify(image.unsqueeze(0))

print(f"Output patch embedding sequence shape: {image.unsqueeze(0).shape}")

quit()
# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# Change image shape to be compatible with matplotlib (color_channels, height, width) -> (height, width, color_channels)
image_permuted = image.permute(1, 2, 0)

# Create a series of subplots
fig, axs = plt.subplots(nrows=IMG_SIZE // PATCH_SIZE, # need int not float
                        ncols=IMG_SIZE // PATCH_SIZE,
                        figsize=(NUM_PATCHES, NUM_PATCHES),
                        sharex=True,
                        sharey=True)

# Loop through height and width of image
for i, patch_height in enumerate(range(0, IMG_SIZE, PATCH_SIZE)): # iterate through height
    for j, patch_width in enumerate(range(0, IMG_SIZE, PATCH_SIZE)): # iterate through width

        # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
        axs[i, j].imshow(image_permuted[patch_height:patch_height+PATCH_SIZE, # iterate through height
                                        patch_width:patch_width+PATCH_SIZE, # iterate through width
                                        :]) # get all color channels

        # Set up label information, remove the ticks for clarity and set labels to outside
        axs[i, j].set_ylabel(i+1,
                             rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        axs[i, j].set_xlabel(j+1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].label_outer()

# Set a super title
fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
plt.show()