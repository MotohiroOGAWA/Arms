import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math

from .transformer import GATLayer, GatDecoderBlock

    


class HierGATDecoder(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, num_heads, ff_dim, dropout=0.1, T=1.0, use_shared_block=False):
        """
        Args:
            num_layers (int): Number of decoder layers.
            node_dim (int): Dimension of node features.
            edge_dim (int): Dimension of edge features.
            hidden_dim (int): Dimension of hidden features.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feedforward dimension.
            dropout (float): Dropout rate.
            T (float): Temperature parameter for GAT.
            use_shared_block (bool): If True, reuse the same `GatDecoderBlock` for all layers. If False, use separate blocks.
        """
        super(HierGATDecoder, self).__init__()
        self.use_shared_block = use_shared_block
        self.num_layers = num_layers

        self.node_proj_linear = nn.Linear(node_dim, hidden_dim)

        if use_shared_block:
            self.shared_block = GatDecoderBlock(hidden_dim, edge_dim, num_heads, ff_dim, dropout, T)
        else:
            self.layers = nn.ModuleList([GatDecoderBlock(hidden_dim, edge_dim, num_heads, ff_dim, dropout, T) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, x, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        x = self.node_proj_linear(x)
        h = x
        if self.use_shared_block:
            # Reuse the same GatDecoderBlock for all layers
            for _ in range(self.num_layers): 
                h = self.shared_block(x, h, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask)
        else:
            # Use a different GatDecoderBlock for each layer
            for layer in self.layers:
                h = layer(x, h, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask)
            
        return self.norm(h)  # Shape: [batch_size, seq_len, hidden_dim]