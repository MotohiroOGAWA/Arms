import torch
import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttentionBlock
from .FeedForward import FeedForwardBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttentionBlock(embed_dim, h=heads, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(embed_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, enc_output, tgt_mask, memory_mask):
        # Shape of x [batch_size, seq_len, embed_dim]
        # Shape of h [batch_size, seq_len, embed_dim]
        # Shape of enc_output [batch_size, encoder_seq_len, embed_dim]
        output = self.self_attention(h, x, x, key_mask=tgt_mask)
        h = self.norm1(h + self.dropout(output))  # Apply self-attention block

        output = self.cross_attention(h, enc_output, enc_output, key_mask=memory_mask)
        h = self.norm2(h + self.dropout(output)) # Apply encoder-decoder attention block

        output = self.feed_forward(h)
        h = self.norm3(self.dropout(output))

        return h  # Shape: [batch_size, seq_len, embed_dim]