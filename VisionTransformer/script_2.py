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

train_dir = "VisionTransformer/data/pizza_steak_sushi/train"
test_dir = "VisionTransformer/data/pizza_steak_sushi/test"

BATCH_SIZE = 32
IMG_SIZE = 224

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
    ])

set_seeds()

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=BATCH_SIZE,
                                                                               num_workers=0)

patch_size = 16

image_batch, label_batch = next(iter(train_dataloader))

image, label = image_batch[0], label_batch[0]

image_permuted = image.permute(1, 2, 0)

print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

x = image.unsqueeze(0)
print(f"Input image shape: {x.shape}")

patch_embedding_layer = PatchEmbedding(in_channels=3,
                                       patch_size=patch_size,
                                       embedding_dim=768)

patch_embedding = patch_embedding_layer(x)
print(f"Patch embedding shape: {patch_embedding.shape}")

# Create the class token embedding
batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
# (only use .ones for demonstration purposes, it should be random)
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                           requires_grad=True) # LEARNABLE!
print(f"Class token embedding shape: {class_token.shape}")
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

# Create the position embedding
number_of_patches = int((height*width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                  requires_grad=True)
patch_and_position_embedding_class_token = patch_embedding_class_token + position_embedding
print(f"Patch and position embedding with class token shape: {patch_and_position_embedding_class_token.shape}")