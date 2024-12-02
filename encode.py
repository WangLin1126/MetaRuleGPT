from utils import *
from forward import *

# 编码器层
class EncoderLayer(nn.Module):
    """
    A single layer of the encoder with self-attention and feed-forward sublayers.

    Args:
        size (int): The dimensionality of input and output vectors.
        self_attn (nn.Module): An instance of the multi-head self-attention mechanism.
        feed_forward (nn.Module): An instance of the feed-forward neural network layer.
        dropout (float): Dropout rate for regularization. Default is 0.1.
    """
    def __init__(self,size,self_attn,feed_forward,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        # Create sublayer connections (residual connections with layer normalization)
        self.sublayer = clone(SublayerConncetion(size,dropout),2)
        
    def forward(self,x,mask = None):
        """
        Forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor to the encoder layer.
            mask (torch.Tensor, optional): Optional mask tensor to prevent attention to certain positions.

        Returns:
            torch.Tensor: The output tensor after processing through the encoder layer.
        """
        if mask == None:
            x = self.sublayer[0](x,lambda x : self.self_attn(x,x,x))
        else:
            x = self.sublayer[0](x,lambda x : self.self_attn(x,x,x,mask))

        return self.sublayer[1](x,self.feed_forward)

class Encoder(nn.Module):
    """
    The encoder module consisting of multiple stacked encoder layers.

    Args:
        layer (EncoderLayer): An instance of the encoder layer.
        N (int): The number of encoder layers to stack.
    """
    def __init__(self,layer,N):

        super(Encoder,self).__init__()
        # Clone N instances of the encoder layer
        self.layers = clone(layer,N)

        # Add a layer normalization layer at the end of the encoder
        self.norm = LayerNorm(layer.size)
    
    def forward(self,x,mask):
        """
        Forward pass through the encoder module.

        Args:
            x (torch.Tensor): Input tensor to the encoder module.
            mask (torch.Tensor): Mask tensor to prevent attention to certain positions.

        Returns:
            torch.Tensor: The output tensor after processing through all encoder layers and normalization.
        """
        # Process through each encoder layer and apply normalization

        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
