"""
Using nn.TransformerEncoderLayer
"""
from torch import nn
from torchinfo import summary

transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                       nhead=12,
                                                       dim_feedforward=3072,
                                                       dropout=0.1,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True)

summary(transformer_encoder_layer)