import torch
import torch.nn as nn


def get_flatten_layer(flatten_info, in_features, out_features):
    if flatten_info['mode'] == 'gat_aggregate':
        mode = 'gat_aggregate'
        flatten_layer = GatAggregateFlatten(
            in_features=in_features,
            out_features=out_features,
            T=flatten_info['gat_T'],
        )
        return mode, flatten_layer
    else:
        raise NotImplementedError("Flatten layer mode not implemented.")

class GatAggregateFlatten(nn.Module):
    def __init__(self, in_features, out_features, T=1.0):
        super(GatAggregateFlatten, self).__init__()
        directed = True
        softmax_per = "dst"
        self.aggregate_layer = GATLayer(in_features, out_features, edge_dim=0, directed=directed, softmax_per=softmax_per, T=T)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, node_mask):
        # shape of x [batch_size, seq_len, in_features]
        x, edge_index, node_mask, edge_mask = GATLayer.create_aggregate_to_new_root_edge(x, node_mask)
        edge_attr = torch.empty((*edge_index.shape[:-1], 0), dtype=torch.float32, device=x.device)
        x = self.aggregate_layer(x, edge_index, edge_attr, node_mask, edge_mask)
        return self.norm(x[..., 0, :])  # Shape: [batch_size, out_features]