"""
1. Create a Classed called PatchEmbedding that inherits from nn.Module
2. Initialize with appropiate hyperparameters, such as channels, embedding dimension, patch size.
3. Create a layer to turn an image into embedding patches (using nn.Conv2d)
4. Create a layer to flatten the feature maps of the output of the layer in (3)
5. Define a forward that defines the forward computation
6. Make sure that the output shape of the layer reflects the required output shape of the patch embedding
"""
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768):
        super().__init__()

        self.patch_size = patch_size

        # Layer to turn an image into embedded patches
        """
        Use a nn.Conv2d layer to turn image into patches of learnable feature maps (embeddings)
        [3,224,224] -> [1,768,14,14]
        [C,H,W] -> [1,(16*16)*3,14,14] (14 patches by 14 patches)
        [C,H,W] -> [batch_size,embedding_dim,feature_map_height,feature_map_width]
        """
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        # Layer to flatten feature map otuputs of Conv2d
        """
        Use a nn.Flatten layer to flatten the embeddings into a single sequence of patch embeddings (a sequence of vectors)
        Get a single feature map in Tensor form...
        (by flattening the [...,14,14])
        [batch_size,embedding_dim,feature_map_height,feature_map_width] -> [batch_size,number_of_patches,embedding_dim]
        [1,768,14,14] -> [1,768,196]
        """
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3) # [batch_size,embedding_dim,number_of_patches] => We need to permute it!
    
    def forward(self, x):
        # Check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"[ERROR] Input image size must be divisible by patch size.\
            \nImage size: {image_resolution}\
            \nPatch size: {self.patch_size}"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # Put dimensions in the right order [batch_size,number_of_patches,embedding_dim]
        return x_flattened.permute(0,2,1) # [1,768,196] -> [1,196,768]

"""
Create a class token and a position embedding
Class token: preprend class embedding (learnable) [1,196,768] -> [1,196+1,768]
Position embedding: a series of 1D learnable position embeddings and to add them to the sequence of patch embeddings [1,196+1,768] <- Shape stays the same!
"""
