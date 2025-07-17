import unittest
import torch
from model.chem_encoder.HierGatChemEncoder import GatChemEncoder
from tests.common.get_graph_data import get_enzyme_graph_data

class TestGatChemEncoder(unittest.TestCase):
    def setUp(self):
        # Load sample graph data
        self.data = get_enzyme_graph_data()

        # Model parameters
        self.num_layers = 2
        self.node_dim = self.data.x.size(-1)
        self.edge_dim = self.data.edge_attr.size(-1) if self.data.edge_attr is not None else 1
        self.hidden_dim = 32
        self.num_heads = 4
        self.ff_dim = 64
        self.dropout = 0.1

        # Instantiate model
        self.model = GatChemEncoder(
            num_layers=self.num_layers,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            use_shared_block=False
        )

    def test_forward_pass(self):
        data = self.data
        edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 1
        edge_attr = data.edge_attr if data.edge_attr is not None else torch.zeros(data.edge_index.shape[1], edge_dim)

        x = data.x.unsqueeze(0)  # [1, num_nodes, node_dim]
        edge_index = data.edge_index.T.unsqueeze(0)  # [1, num_edges, 2]
        edge_attr = edge_attr.unsqueeze(0)  # [1, num_edges, edge_dim]

        node_mask = torch.zeros(1, x.shape[1], dtype=torch.bool)
        edge_mask = torch.zeros(1, edge_index.shape[1], dtype=torch.bool)

        output = self.model(x, edge_index, edge_attr, node_mask, edge_mask)
        self.assertEqual(output.shape, x.shape)

if __name__ == "__main__":
    unittest.main()
