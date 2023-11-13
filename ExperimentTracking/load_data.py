from torchvision import transforms
from data_setup import create_dataloaders
from pathlib import Path
import torchvision

data_path = Path("ExperimentTracking/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

# When doing Transfer Learning, data passed through the model
# should be transformed in the same way that the data the model was trained on

normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]
)

manual_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize])

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transform=manual_transforms,
                                                                    batch_size=32)

print(train_dataloader, test_dataloader, class_names)

# NOTE: Automatic data transform creation based on pretrained model weigths used
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # "DEFAULT" = best available weights

auto_transforms = weights.transforms()

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transform=auto_transforms,
                                                                    batch_size=32)