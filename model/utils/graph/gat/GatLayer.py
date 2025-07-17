import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Literal


class GATLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: int,
        directed: bool = False,
        softmax_per: Literal['graph', 'dst'] = "dst",
        T: float = 1.0
    ):
        """
        Graph Attention Layer (GAT) supporting batched graphs and flexible softmax behavior.

        Args:
            in_features (int): Dimensionality of input node features.
            out_features (int): Dimensionality of output node features.
            edge_dim (int): Dimensionality of edge attributes.
            directed (bool): If False, undirected edges are assumed (adds reverse edges).
            softmax_per (str): Softmax scope: "graph" for per-graph, "dst" for per-destination-node.
            T (float): Temperature for scaling attention logits (softmax). Lower T = sharper attention.
        """
        super().__init__()

        self.directed = directed
        assert softmax_per in ["graph", "dst"], "softmax_per must be either 'graph' or 'dst'"
        self.softmax_per = softmax_per
        self.T = T

        # Learnable parameters for node transformation and attention scoring
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features + edge_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    @property
    def in_features(self):
        return self.W.shape[0]

    @property
    def out_features(self):
        return self.W.shape[1]

    @property
    def edge_dim(self):
        return self.a.shape[0] - 2 * self.out_features

    def reset_parameters(self):
        # Xavier initialization for stability
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, batch: Batch):
        """
        Forward pass for GAT layer.

        Args:
            batch (Batch): PyG Batch object containing x, edge_index, edge_attr, batch.

        Returns:
            torch.Tensor: Output node embeddings of shape [num_nodes, out_features].
        """
        # Unpack batch data
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_id = batch.batch

        num_nodes, node_dim = x.shape
        num_edges, edge_feat_dim = edge_attr.shape
        num_graphs = batch_id.max() + 1

        if not torch.jit.is_tracing():
            assert node_dim == self.in_features
            assert edge_feat_dim == self.edge_dim
            assert edge_index.shape[1] == num_edges, "Edge index must match number of edges in edge_attr"

        src, dst = edge_index[0], edge_index[1]

        if not self.directed:
            # Add reverse edges for undirected graph
            src, dst = torch.cat([src, dst]), torch.cat([dst, src])
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # Add self-loops to each node
        loop_idx = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, loop_idx])
        dst = torch.cat([dst, loop_idx])
        loop_edge_attr = torch.zeros(num_nodes, edge_feat_dim, dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_edge_attr], dim=0)
        edge_batch_id = batch_id[src]  # Assign each edge to a graph based on source node

        # Linear transform on node features
        x_transformed = x @ self.W
        h_src = x_transformed[src]
        h_dst = x_transformed[dst]

        # Concatenate source, destination, and edge features
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)

        # Compute unnormalized attention scores
        e_logits = self.leaky_relu((edge_input @ self.a).squeeze(-1))
        attention_weights = torch.zeros_like(e_logits)

        if self.softmax_per == "graph":
            # Softmax normalization per graph
            edge_batch_id_sorted, perm = torch.sort(edge_batch_id)
            e_logits = e_logits[perm]
            h_src = h_src[perm]
            dst = dst[perm]

            edge_counts = torch.bincount(edge_batch_id_sorted, minlength=num_graphs)
            e_split = torch.split(e_logits, edge_counts.tolist())
            softmax_split = [F.softmax(ei / self.T, dim=0) for ei in e_split]
            attention_weights = torch.cat(softmax_split)
            attention_weights = attention_weights[torch.argsort(perm)]  # Undo sort

        elif self.softmax_per == "dst":
            # Softmax normalization per destination node
            dst_sorted, perm = torch.sort(dst)
            e_logits = e_logits[perm]
            h_src = h_src[perm]
            dst = dst[perm]

            dst_counts = torch.bincount(dst, minlength=num_nodes)
            e_split = torch.split(e_logits, dst_counts.tolist())
            softmax_split = [F.softmax(ei / self.T, dim=0) for ei in e_split]
            attention_weights = torch.cat(softmax_split)
            attention_weights = attention_weights[torch.argsort(perm)]  # Undo sort

        # Weighted sum of source node features (attention-based aggregation)
        out = torch.zeros_like(x_transformed)
        out.index_add_(0, dst, attention_weights.unsqueeze(-1) * h_src)

        return out

    @staticmethod
    def add_root_and_edges(x: torch.Tensor, node_mask: torch.Tensor):
        """
        Adds a virtual root node at index 0, connects valid nodes to it, and generates masks.

        Args:
            x (torch.Tensor): [B, N, D] node features.
            node_mask (torch.Tensor): [B, N] where True indicates masked (invalid) node.

        Returns:
            x_ext (torch.Tensor): [B, N+1, D] with prepended root node.
            edge_index (torch.Tensor): [B, N, 2] edge indices connecting nodes to root.
            node_mask_ext (torch.Tensor): [B, N+1] extended node mask.
            edge_mask (torch.Tensor): [B, N] boolean mask for edges (True = masked).
        """
        batch_size, num_nodes = node_mask.shape
        device = x.device

        # Identify valid node indices
        valid_node_idx = torch.nonzero(~node_mask, as_tuple=True)

        # Create edge index tensor initialized with -1
        edge_index = torch.full((batch_size, num_nodes, 2), -1, dtype=torch.long, device=device)
        edge_index[valid_node_idx[0], valid_node_idx[1], 0] = valid_node_idx[1] + 1  # source = node + 1
        edge_index[valid_node_idx[0], valid_node_idx[1], 1] = 0  # destination = root (index 0)

        # Add virtual root node to feature and mask tensors
        x_ext = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)  # prepend root node
        node_mask_ext = torch.cat([torch.zeros_like(node_mask[:, :1]), node_mask], dim=1)

        # Initialize all edges as masked, then unmask valid edges
        edge_mask = torch.ones_like(node_mask, dtype=torch.bool)
        edge_mask[valid_node_idx[0], valid_node_idx[1]] = False

        return x_ext, edge_index, node_mask_ext, edge_mask
