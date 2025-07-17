import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math

from .transformer import GATLayer, GatEncoderBlock



class HierGatChemEncoder(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, num_heads, ff_dim, dropout=0.1, T=1.0, use_shared_block=False):
        """
        Hierarchical GAT-based Encoder.

        Args:
            num_layers (int): Number of encoder layers.
            node_dim (int): Dimension of node features.
            edge_dim (int): Dimension of edge features.
            hidden_dim (int): Hidden dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feedforward dimension.
            dropout (float): Dropout rate.
            T (float): Temperature parameter for GAT.
            use_shared_block (bool): If True, reuse the same `GatEncoderBlock` for all layers; otherwise, use separate blocks.
        """
        super(HierGatChemEncoder, self).__init__()
        self.use_shared_block = use_shared_block
        self.num_layers = num_layers

        self.node_proj_linear = nn.Linear(node_dim, hidden_dim)

        if use_shared_block:
            # Use the same `GatEncoderBlock` for all layers
            self.shared_block = GatEncoderBlock(hidden_dim, edge_dim, num_heads, ff_dim, dropout, T=T)
        else:
            # Use a different `GatEncoderBlock` for each layer
            self.layers = nn.ModuleList([GatEncoderBlock(hidden_dim, edge_dim, num_heads, ff_dim, dropout, T=T) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, x, edge_index, edge_attr, node_mask, edge_mask):
        """
        Forward pass of the Hierarchical GAT Encoder.

        Args:
            x (torch.Tensor): Input node features (batch_size, seq_len, node_dim).
            edge_index (torch.Tensor): Edge indices for the graph.
            edge_attr (torch.Tensor): Edge attributes.
            node_mask (torch.Tensor): Node mask for self-attention.
            edge_mask (torch.Tensor): Edge mask for filtering valid edges.

        Returns:
            torch.Tensor: Encoded node features.
        """
        x = self.node_proj_linear(x)  # Shape: [batch_size, seq_len, hidden_dim]
        h = x
        if self.use_shared_block:
            # Reuse the same `GatEncoderBlock` for all layers
            for _ in range(self.num_layers):  # Loop for the specified number of layers
                h = self.shared_block(x, h, edge_index, edge_attr, node_mask, edge_mask)
        else:
            # Use a different `GatEncoderBlock` for each layer
            for layer in self.layers:
                h = layer(x, h, edge_index, edge_attr, node_mask, edge_mask)
        
        return self.norm(h)  # Shape: [batch_size, seq_len, hidden_dim]

        

        



