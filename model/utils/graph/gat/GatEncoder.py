import torch
import torch.nn as nn
from ..gat.GatLayer import GATLayer
from torch_geometric.data import Batch


class GatEncoderBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, ff_dim, dropout=0.1, T=1.0):
        super(GatEncoderBlock, self).__init__()
        self.gat = GATLayer(node_dim, node_dim, edge_dim, directed=True, softmax_per="dst", T=T)
        self.self_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
        # self.self_attention2 = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(node_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: Batch):
        output = self.gat(batch)  # Apply GAT layer
        h = self.norm1(self.dropout(output))  # Apply GAT block

        output = self.self_attention(h, x, x, key_mask=node_mask)
        h = self.norm2(h + self.dropout(output))  # Apply self-attention block

        output = self.feed_forward(h)
        h = self.norm3(self.dropout(output))

        return h  # Shape: [batch_size, seq_len, node_dim]