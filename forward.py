from utils import *

class  PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network with two linear layers and ReLU activation.

    Args:
        d_model (int): Dimensionality of the input and output embeddings.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float): Dropout rate for regularization. Default is 0.1.
    """

    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()

       # Define two linear layers
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        # Dropout for regularization
        self.dropout = nn.Dropout(p = dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

'''
d_model = 8
d_ff = 24
dropout = 0.2
x = torch.rand(size = (1,8))
print(x)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
ff_result = ff(x)
print(ff_result.shape)
print(ff_result)
'''

class LayerNorm(nn.Module):
    """
    Layer normalization to stabilize training and improve performance.

    Args:
        feature (int): Number of features in the input tensor (typically `d_model`).
        eps (float): Small constant for numerical stability. Default is 1e-6.
    """
    def __init__(self,feature,eps = 1e-6):
        super(LayerNorm,self).__init__()
        
        self.a2 = nn.Parameter(torch.ones(feature))
        self.b2 = nn.Parameter(torch.zeros(feature))

        self.eps = eps

    def forward(self,x):
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor from the previous layer.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

'''
feature = 4
eps = 1e-6
test = LayerNorm(feature,eps)
x = torch.Tensor([[1,3,3,1]])
ln_result = test(x)
print(ln_result.shape)
print(ln_result)

x = torch.FloatTensor([[0,2,2,0]])
print(x.std(dim = -1))

'''

class SublayerConncetion(nn.Module):
    """
    Implements sublayer connection with residual connection and layer normalization.

    Args:
        size (int): Dimensionality of the input tensor (typically `d_model`).
        dropout (float): Dropout rate for regularization. Default is 0.1.
    """
    def __init__(self,size,dropout=0.1):
        super(SublayerConncetion,self).__init__()
        self.norm = LayerNorm(size)
        
        self.dropout = nn.Dropout(p = dropout)
        self.size = size

    def forward(self,x,sublayer):
        """
        Forward pass for the sublayer connection.

        Args:
            x (torch.Tensor): Input tensor from the previous layer.
            sublayer (Callable): A function representing the sublayer operation.

        Returns:
            torch.Tensor: Output tensor after applying sublayer operation with residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
'''
size = d_model = 512
head = 8
dropout = 0.2
x = pe_result
mask = Variable(torch.zeros(1,9,9))
self_attn = MultiHeadedAttention(head,d_model)
sublayer = lambda x : self_attn(x,x,x,mask)
sc = SublayerConncetion(size,dropout)
sc_result = sc(x,sublayer)

print(sc_result)
print(sc_result.shape)

'''        
