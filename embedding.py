from utils import * 

class Embeddings(nn.Module):
    """
    Embedding layer for mapping input tokens to dense vectors.

    Args:
        d_model (int): The dimensionality of the embedding vectors.
        vocab (int): The size of the vocabulary (total unique tokens).
    """
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)

        self.d_model = d_model

    def forward(self,x,judge = False):
        """
        Forward pass to process input tokens.

        Args:
            x (torch.Tensor or str): Input tensor or string of characters.
            judge (bool): If True, converts input string to tensor based on ASCII values.

        Returns:
            torch.Tensor: Scaled embedding vectors for the input tokens.
        """
        if judge:
            x = torch.tensor([ord(char)  for char in x ], dtype=torch.long).unsqueeze(0)
        return self.lut(x) * math.sqrt(self.d_model)




class PositionalEncoding(nn.Module):
    """
    Adds positional information to the token embeddings using sinusoidal encodings.

    Args:
        d_model (int): The dimensionality of the embedding vectors.
        dropout (float): Dropout rate for regularization.
        max_len (int, optional): Maximum length of input sequences. Defaults to 5000.
    """
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        
        self.dropout = nn.Dropout(p = dropout)
        
        # Initialize the positional encoding matrix
        pe = torch.zeros(max_len,d_model)
        
        # Generate position indices
        position = torch.arange(0,max_len).unsqueeze(1)
        
        # Compute scaling factors for sinusoidal encodings
        div_term = torch.exp(torch.arange(0,d_model,2) * - (math.log(10000) / d_model))
        # Apply sine to even indices and cosine to odd indices
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)
        # Register `pe` as a buffer to ensure it is not treated as a model parameter
        self.register_buffer('pe',pe)

    def forward(self,x):
        """
        Forward pass to add positional encodings to the embeddings.

        Args:
            x (torch.Tensor): Input embeddings with shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Embeddings with positional encodings added.
        """
        # Add positional encodings (non-trainable) to the input embeddings
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad = False)
        return self.dropout(x)



