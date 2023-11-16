"""
Adam Optimizer is used, parameters taken from the paper

Weight Decay = Regularization technique which adds a small penalty, usually the L2 norm of the weights,
(all the weights of the model), to the loss function
Regularization technique = Prevention of overfitting

Loss function is Cross Entropy Loss given that it's a multi-class classification problem
"""
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
from model import VisionTransformer
from common.engine import train

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_dir = "VisionTransformer/data/pizza_steak_sushi/train"
test_dir = "VisionTransformer/data/pizza_steak_sushi/test"

set_seeds()

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

BATCH_SIZE = 32
IMG_SIZE = 224

# When using a pretrained model, one has to make sure that data is formatted in the same way the model was trained on
vit_transforms = pretrained_vit_weights.transforms()
# vit_transforms

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=vit_transforms,
                                                                               batch_size=BATCH_SIZE,
                                                                               num_workers=0)

pretrained_vit.heads = nn.Linear(in_features=768,
                                 out_features=len(class_names))

optimizer = torch.optim.Adam(pretrained_vit.parameters(),
                             lr=1e-3,
                             betas=(0.9, 0.999),
                             weight_decay=0.1)

loss_fn = torch.nn.CrossEntropyLoss()

EPOCHS=10
train(model=pretrained_vit,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=EPOCHS,
      device=device)