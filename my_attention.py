from utils import *

# query:
def attention(query,key,value,mask=None,dropout=None):
    """
    Computes the scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embedding_dim).
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embedding_dim).
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embedding_dim).
        mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions. Default is None.
        dropout (nn.Module, optional): Dropout instance for regularization. Default is None.

    Returns:
        tuple: Output tensor after attention computation and the attention weights.
    """
    d_k = query.size(-1)
    
    # Compute attention scores using the scaled dot-product formula
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    # Apply the mask if provided, filling masked positions with a very negative value
    if mask is not None:
        scores = scores.masked_fill(mask == 0,-1e9)

    # Apply softmax to normalize scores along the last dimension
    p_attn = F.softmax(scores, dim = -1)
    
    # Apply dropout if a dropout instance is provided
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn,value),p_attn

'''
q = k  = v = torch.rand(size=(1,4,4))
x,y = attention(q,k,v)
print('softmax(qk^t/ d_k)',y)
print('res:',x)
'''

class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention mechanism for handling multiple attention heads in parallel.

    Args:
        head (int): Number of attention heads.
        embedding_dim (int): Dimensionality of the input embeddings.
        dropout (float): Dropout rate for regularization. Default is 0.1.
    """

    def __init__(self,head,embedding_dim,dropout = 0.1):
        super(MultiHeadedAttention,self).__init__()
        # Ensure the embedding dimension is divisible by the number of heads
        assert  embedding_dim % head == 0

        # Dimensionality of each head
        self.d_k = embedding_dim // head

        self.head = head
        self.embedding_dim = embedding_dim
        
        # Four linear layers for query, key, value, and the final output transformation
        self.linears = clone(nn.Linear(embedding_dim,embedding_dim),4)
        
        # Placeholder for attention weights
        self.attn = None

       # Dropout layer for regularization
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self,query,key,value,mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embedding_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embedding_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embedding_dim).
            mask (torch.Tensor, optional): Mask tensor for attention. Default is None.

        Returns:
            torch.Tensor: Output tensor after multi-head attention.
        """
        
        if mask is not None:
            # Extend mask for multiple heads
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
            
        # Project query, key, and value for each head and reshape for parallel processing
        query,key,value = \
            [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2) for model,x in zip(self.linears,(query,key,value))]

        # Apply attention to each head
        x,self.attn = attention(query,key,value,mask = mask,dropout = self.dropout)
        # Concatenate the outputs from all heads
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head * self.d_k)
        return self.linears[-1](x)

'''
mult = MultiHeadedAttention(2,4)
q = k = v = torch.rand(size=(1,4,4))
print('q',q)
x = mult(q,k,v)
print(x)
'''
