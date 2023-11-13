from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from torchinfo import summary
from torch import nn
from torch import cuda
from utils import set_seeds

# Setup device-agnostic code
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device: {device}")

"""
https://github.com/pytorch/vision/issues/7744
"""

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

# efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model= efficientnet_b0(weights="DEFAULT")

# summary(model=model,
#         input_size=(1, 3, 224, 224), # [batch_size,color_channels,height,width]
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

set_seeds()

# Freeze the base model and change the output model to "suit FoodVision needs"
num_classes = 3
# Freeze all base layers
for param in model.features.parameters():
    param.requires_grad = False # Don't track gradients so don't change them when optimizing!
# Update classifier head of the model (to suit the 3 classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=num_classes)
)

# summary(model=model,
#         input_size=(32, 3, 224, 224), # [batch_size,color_channels,height,width]
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])