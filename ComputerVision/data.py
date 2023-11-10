from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 32
RANDOM_SEED = 42

train_data = datasets.FashionMNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=ToTensor(), # How is data transformed?
                                   target_transform=None # How is labels/targets transformed?
)

test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(), # How is data transformed?
                                  target_transform=None # How is labels/targets transformed?
)

print(f"Train Data examples: {len(train_data)}")
print(f"Test Data examples: {len(test_data)}")

print("==================================================================")
print("Class names:")
class_names = train_data.classes
print(class_names)

print("Class indexes:")
class_idxs = train_data.class_to_idx
print(class_idxs)

print("==================================================================")
image, label = train_data[0]
# Only one color channel because the images are in b&w
print(f"Image shape: {image.shape} == [color_channels, height, width]")
print(f"Image label: {class_names[label]}")

# Currently, data is in the form of PyTorch DataSet
# Turn dataset into a Python iterable using DatLoader, specifically, turn data into bathces (or mini-batches)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

print("==================================================================")
print(f"Train Data Loader: {train_dataloader}")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Test Data Loader: {test_dataloader}")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

print("==================================================================")
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"Features batch shape: {train_features_batch.shape}") # [batch size, color channels, height, width]
print(f"Labels batch shape: {train_labels_batch.shape}") # [# labels] ( =  batch_size, # labels associated with the batch)

print("==================================================================")
# Show a sample
torch.manual_seed(RANDOM_SEED)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
image, label = train_features_batch[random_idx], train_labels_batch[random_idx]
print(f"Image with index : {random_idx}")
print(f"has label {label}")