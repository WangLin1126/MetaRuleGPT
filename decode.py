from utils import *
from forward import *

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout=0.1):
        """
        Initializes a single layer of the decoder.
        
        Args:
            size (int): Dimensionality of the input/output features.
            self_attn (nn.Module): Multi-head self-attention mechanism.
            src_attn (nn.Module): Multi-head source-attention mechanism.
            feed_forward (nn.Module): Feedforward neural network layer.
            dropout (float): Dropout rate for regularization.
        """
        super(DecoderLayer,self).__init__()
        
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clone(SublayerConncetion(size,dropout),3)
    def forward(self,x,memory,source_mask,target_mask):
        """
        Processes the input through the decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor from the previous layer.
            memory (torch.Tensor): Output from the encoder.
            source_mask (torch.Tensor): Mask for the source data.
            target_mask (torch.Tensor): Mask for the target data.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        m = memory
            
        x = self.sublayer[0](x,lambda x : self.self_attn(x,x,x,target_mask))
        x = self.sublayer[1](x,lambda x : self.src_attn(x,m,m,source_mask))

        return self.sublayer[2](x,self.feed_forward)


class Decoder(nn.Module):
    def __init__(self,layer,N):
        """
        Initializes the decoder with multiple stacked layers.
        
        Args:
            layer (DecoderLayer): A single decoder layer.
            N (int): Number of layers to stack in the decoder.
        """
        super(Decoder,self).__init__()
        
        self.layers = clone(layer,N)

        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,source_mask,target_mask):
        """
        Processes the input through the stacked decoder layers and applies normalization.
        
        Args:
            x (torch.Tensor): Input tensor from the target data.
            memory (torch.Tensor): Encoder output serving as the context.
            source_mask (torch.Tensor): Mask for the source data.
            target_mask (torch.Tensor): Mask for the target data.

        Returns:
            torch.Tensor: Normalized output tensor after processing through all decoder layers.
        """
        for layer in self.layers:
            x = layer(x,memory,source_mask,target_mask)
        
        return self.norm(x)

