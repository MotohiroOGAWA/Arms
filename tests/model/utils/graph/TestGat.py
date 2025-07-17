import unittest
import torch
from torch_geometric.data import Data, Batch
import os

from model.utils.graph.gat.GatLayer import GATLayer  # Update the import path as needed
from tests.common.get_graph_data import get_enzyme_graph_data  # Update the import path as needed

class TestGATLayer(unittest.TestCase):
    def setUp(self):
        
        self.batch = get_enzyme_graph_data([0, 1, 2])

        edge_dim = 2
        self.batch.edge_attr = torch.ones(self.batch.num_edges, edge_dim)  # Add edge attributes if needed


        num_nodes = 7
        edge_index = torch.tensor([
            [0, 1, 2, 2, 3, 4, 5, 6, 1, 2, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 0, 1, 0, 1, 5, 2, 3, 4, 3]
        ], dtype=torch.long)  # 9 edges, directed

        x = torch.randn((num_nodes, 3)) 
        edge_attr = torch.randn((edge_index.shape[1], 2))  
        batch = torch.zeros(num_nodes, dtype=torch.long) 
        self.simple_batch = Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )

        # Initialize the GAT layer
        self.model = GATLayer(
            in_features=3,     # Node feature dimension
            out_features=4,    # Desired output dimension
            edge_dim=edge_dim,        # Edge attribute dimension
            directed=False,    # Treat graphs as undirected
            softmax_per='dst', # Normalize attention per destination node
            T=1.0              # Temperature for softmax
        )

    def test_output_shape(self):
        # Run the forward pass
        output = self.model(self.batch)

        # Expecting 4 output nodes (2 nodes per graph × 2 graphs), each with 4 features
        self.assertEqual(output.shape, (self.batch.num_nodes, self.model.out_features))

    def test_export_to_onnx(self):
        """
        This test exports the model to ONNX format for visualization/debugging.
        It does not assert anything and is skipped unless explicitly run.
        """
        import torch.onnx

        self.model.eval()

        # Extract required tensors from the batch
        x = self.simple_batch.x
        edge_index = self.simple_batch.edge_index
        edge_attr = self.simple_batch.edge_attr
        batch = self.simple_batch.batch  # Needed if GATLayer uses it internally

        # Export path
        output_dir = "tests/data/model/utils/graph"
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, "gat_layer.onnx")

        # Define a wrapper to use tuple input (needed for ONNX tracing)
        class GATWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, edge_index, edge_attr, batch):
                data = Batch(batch=batch, x=x, edge_index=edge_index, edge_attr=edge_attr)
                return self.model(data)

        wrapper = GATWrapper(self.model)

        # Export to ONNX
        torch.onnx.export(
            wrapper,
            (x, edge_index, edge_attr, batch),
            export_path,
            input_names=["x", "edge_index", "edge_attr", "batch"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_attr": {0: "num_edges"},
                "batch": {0: "num_nodes"},
                "output": {0: "num_nodes"}
            },
            opset_version=11,
            verbose=False
        )
        print(f"Exported ONNX model to {export_path}")


if __name__ == "__main__":
    unittest.main()
