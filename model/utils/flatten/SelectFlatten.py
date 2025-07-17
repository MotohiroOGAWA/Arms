import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_lib import LSTMGraph
from torch_geometric.data import Batch, Data
import math


class SelectFlatten(nn.Module):
    def __init__(self, embed_dim):
        super(SelectFlatten, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, select_indices):
        # shape of x [batch_size, seq_len, embed_dim]
        x = x[..., select_indices, :]
        return self.norm(x)  # Shape: [batch_size, embed_dim]