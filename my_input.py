from utils import * 

class Generator(nn.Module):
    """
    Generator layer that projects the output of the decoder to the vocabulary size.

    Args:
        d_model (int): The embedding dimension of the model.
        vocab_size (int): The size of the vocabulary.
    """
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()

        self.project = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        """
        Forward pass through the generator layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The log softmax of the projected output of shape (batch_size, seq_len, vocab_size).
        """
        return F.log_softmax(self.project(x),dim=-1)


class EncoderDecoder(nn.Module):
    """
    The encoder-decoder architecture combining an encoder, decoder, embeddings, and a generator.

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        source_embed (nn.Module): The embedding layer for the source sequence.
        target_embed (nn.Module): The embedding layer for the target sequence.
        generator (nn.Module): The generator layer for the output layer.
    """
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
        
    def forward(self,source,target,source_mask,target_mask):
        """
        Forward pass through the encoder-decoder.

        Args:
            source (torch.Tensor): The source input tensor of shape (batch_size, source_seq_len).
            target (torch.Tensor): The target input tensor of shape (batch_size, target_seq_len).
            source_mask (torch.Tensor): The source sequence mask tensor.
            target_mask (torch.Tensor): The target sequence mask tensor.

        Returns:
            torch.Tensor: The output tensor from the decoder after passing through the generator.
        """
        # Perform encoding and decoding, and then apply the generator layer
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)

        """
        Encode the source sequence.

        Args:
            source (torch.Tensor): The source input tensor.
            source_mask (torch.Tensor): The source sequence mask tensor.

        Returns:
            torch.Tensor: The encoded memory of the source sequence.
        """
        return self.encoder(self.src_embed(source),source_mask)

    def decode(self,memory,source_mask,target,target_mask):
        """
        Decode the target sequence using the encoded memory.

        Args:
            memory (torch.Tensor): The memory produced by the encoder.
            source_mask (torch.Tensor): The mask for the source sequence.
            target (torch.Tensor): The target input tensor.
            target_mask (torch.Tensor): The mask for the target sequence.

        Returns:
            torch.Tensor: The output of the decoder.
        """
        return self.decoder(self.tgt_embed(target),memory,source_mask,target_mask)
