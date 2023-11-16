"""
DONE BY HAND

LayerNorm is a technique to normalize the distribution of intermediate layers.
Enables smoother gradients, faster training, and better generalization accuracy.
Normalization = Make everything have the same mean and standard deviation
Normalize values over D dimensions (in this case, D is the embedding dimension)
... Normalization is done along the embedding dimensions

Multihead self-attention (MSA) = Which part of a sequence should pay the most attention to itself?
In this case, having a series of embedded image patches, which patch significantly relates to another patch?
ViT should (hopes to) real this relationship/representation

Multilayer perceptron block (MLP) = Quite broad term for a block with a series of layer(s)
Layers can be multiple or even only one hidden layer
Layer can mean: fully-connected, dense, linear, feed-forward. All are often similar names for the same thing
... in PyTorch they are often called nn.Linear
... in TF they might be called layers.Dense()
In this case, contains two layers with a GELU non-linearity. GELU is like a softened ReLU, which can output neagtive values!
Dropout is used! After every dense layer
"""

from torch import nn
from torchinfo import summary

class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Creates a multi-head self-attention block (MSA block)
    Default values are taken from the paper
    """
    def __init__(self,
                 embedding_dim: int=768, # Hidden size D (embedding dimension)
                 num_heads: int=12, # Heads
                 attn_dropout: float=0):
        super().__init__()

        # Normalization Layer / LayerNorm (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multi-head self-attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # (batch,seq,features) = [batch,number_of_patches,embedding_dimension]
    
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    """
    Creates a multilayer perceptron block (MLP block)
    Default values are taken from the paper
    Linear -> Non-Linear -> Droput -> Linear -> Dropout
    """
    def __init__(self,
                 embedding_dim: int=768, # Hidden size D (embedding dimension)
                 mlp_size: int=3072, # MLP number of hidden units = MLP size
                 dropout: float=0.1):
        super().__init__()

        # Normalization Layer / LayerNorm (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multilayer perceptron block (MLP block)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

"""
The transformer encoder is a combination of alternating blocks of MSA and MLP
And there are residual connections between each block

Encoder = Turn a sequence into learnable representation
Decoder = Go from a learnable representation back to some sort of sequence
Residual connection = Add a layer(s) input to its subsequent output, enabling
the creation of deeper networks (prevents weights form getting too small)

Input -> MSA Block -> [MSA Block output + Input] -> MLP Block -> [MLP Block output + MSA Block output + Input] -> ...
"""

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 num_heads: int=12,
                 mlp_size: int=3072,
                 mlp_droput: float=0.1,
                 attn_droput: int=0):
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_droput)
        
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_droput)
    
    def forward(self, x):
        x = self.msa_block(x) + x # Residual/Skip Connection for Eq. 2
        x = self.mlp_block(x) + x # Residual/Skip Connection for Eq. 3
        return x