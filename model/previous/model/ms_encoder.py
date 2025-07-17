import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math

from .transformer import DecoderBlock



class MsCrossEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, hidden_dim, num_heads, ff_dim, precursor_seq_len, dropout=0.1, use_shared_block=False):
        super(MsCrossEncoder, self).__init__()
        self.use_shared_block = use_shared_block
        self.num_layers = num_layers
        self.precursor_seq_len = precursor_seq_len

        self.fragment_proj_linear = nn.Linear(embed_dim, hidden_dim)
        self.precursor_proj_linear = nn.Linear(embed_dim, hidden_dim*precursor_seq_len)

        if use_shared_block:
            self.shared_block = DecoderBlock(hidden_dim, num_heads, ff_dim, dropout)
        else:
            # Use a different `GatEncoderBlock` for each layer
            self.layers = nn.ModuleList([DecoderBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        # GAT Layer for final aggregation to root
        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, fragment_x, precursor_x, fragment_mask):
        """
        Forward pass of the Hierarchical GAT Encoder.

        Args:
            fragment_x (torch.Tensor): Input node features for fragment graph (batch_size, seq_len, embed_dim).
            precursor_x (torch.Tensor): Input node features for precursor graph (batch_size, seq_len, embed_dim).
            fragment_mask (torch.Tensor): Node mask for fragment graph.
            precursor_mask (torch.Tensor): Node mask for precursor graph.

        Returns:
            torch.Tensor: Encoded node features.
        """
        x = self.fragment_proj_linear(fragment_x)  # Shape: [batch_size, seq_len, hidden_dim]
        h = x
        precursor_x = self.precursor_proj_linear(precursor_x)
        precursor_x = precursor_x.unsqueeze(1).reshape(*precursor_x.shape[:-1], self.precursor_seq_len, -1)
        precursor_mask = torch.zeros(*precursor_x.shape[:-1], dtype=torch.bool, device=precursor_x.device)
        if self.use_shared_block:
            # Reuse the same `GatEncoderBlock` for all layers
            for _ in range(self.num_layers):  # Loop for the specified number of layers
                h = self.shared_block(x, h, precursor_x, fragment_mask, precursor_mask)
        else:
            # Use a different `GatEncoderBlock` for each layer
            for layer in self.layers:
                h = layer(x, h, precursor_x, fragment_mask, precursor_mask)
        
        return self.norm(h)

        
