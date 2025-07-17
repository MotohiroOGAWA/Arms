import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_lib import LSTMGraph
from torch_geometric.data import Batch, Data
import math


class GatDecoderBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, ff_dim, dropout=0.1, T=1.0):
        super(GatDecoderBlock, self).__init__()
        self.gat = GATLayer(node_dim, node_dim, edge_dim, directed=True, softmax_per="dst", T=T)
        self.self_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(node_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        self.norm4 = nn.LayerNorm(node_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask):
        # Shape of x [batch_size, seq_len, node_dim]
        # Shape of h [batch_size, seq_len, node_dim]
        # Shape of enc_output [batch_size, encoder_seq_len, dim]
        output = self.gat(h, edge_index, edge_attr, tgt_mask, edge_mask)
        h = self.norm1(self.dropout(output))  # Apply GAT block

        output = self.self_attention(h, x, x, key_mask=tgt_mask)
        h = self.norm2(h + self.dropout(output))  # Apply self-attention block

        output = self.cross_attention(h, enc_output, enc_output, key_mask=memory_mask)
        h = self.norm3(h + self.dropout(output)) # Apply encoder-decoder attention block

        output = self.feed_forward(h)
        h = self.norm4(self.dropout(output))

        return h  # Shape: [batch_size, seq_len, node_dim]