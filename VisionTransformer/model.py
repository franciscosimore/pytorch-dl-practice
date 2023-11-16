from torch import nn

class VisionTransformerModel(nn.Module):
    """
    Layers: number of transformer encoder layers (or blocks) (Lx)
    Hidden size D: embedding size throughout the architecture
    MLP size: number of hidden units/neurons in the MLP
    Head: number of multi-head self-attention heads
    """
    # Split data into patches and "create" the class, position and patch embedding