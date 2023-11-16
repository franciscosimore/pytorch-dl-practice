from torch import nn
import torch
from patchembedding import PatchEmbedding
from transformerencoder import TransformerEncoderBlock
from torchinfo import summary

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 in_channels: int=3,
                 patch_size: int=16,
                 num_transformer_layers: int=12,
                 embedding_dim: int=768,
                 mlp_size: int=3072,
                 num_heads: int=12,
                 attn_dropout: float=0,
                 mlp_dropout: float=0.1,
                 embedding_dropout: float=0.1,
                 num_classes: int=1000):
        super().__init__()

        # Assertion of image size and patch size compatibility
        assert img_size & patch_size == 0, "[ERROR] Image size must be dividible by the patch size!"

        # Calculate number of patches
        self.num_patches = (img_size * img_size) // (patch_size**2)

        # Create learnable class embedding (needs to go at front of sequence of patches)
        self.class_embedding = nn.Parameter(data=torch.randn(1,1,embedding_dim),
                                            requires_grad=True)
        
        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim))

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # Create transformer encoder block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_droput=mlp_dropout,
                                                                           attn_droput=attn_dropout)
                                                                           for _ in range(num_transformer_layers)])
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0] # Get batch size
        # We want the class embedding to each image in the batch!
        class_token = self.class_embedding.expand(batch_size, -1, -1) # Create class token embedding and expand it to match the batch size (Eq. 1)
        # Create the patch embedding (Eq. 1)
        x = self.patch_embedding(x)
        # Concatenate the class token embedding and patch embedding (Eq. 1)
        x = torch.cat((class_token, x), dim=1)  # [batch size, number of patches, embedding dimensions]
        # Add position embedding to class token and patch embedding
        x = self.position_embedding + x
        # Apply dropout to patch embedding ("directly after adding position to patch embeddings")
        x = self.embedding_dropout(x)
        # Pass position and patch embedding to Transformer Encoder (Eq. 2 and Eq. 3)
        x = self.transformer_encoder(x)
        # Put "0th" index logit through classifier (Eq. 4)
        x = self.classifier(x[:,0])
        return x

summary(model=VisionTransformer(num_classes=3),
        input_size=(1,3,224,224), # [batch size, color channels, height, width]
        col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])