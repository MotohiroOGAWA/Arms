import os
import torch
from torch_geometric.data import Data, Batch

def get_enzyme_graph_data(indices: list[int] = [0]) -> Batch:
    """
    Load one or more graphs from the ENZYMES dataset by index.
    If not saved yet, download and save the graph data and an image of its structure.

    Args:
        indices (list[int]): List of indices to retrieve.

    Returns:
        Batch: A PyTorch Geometric Batch object containing multiple graphs.
    """
    dataset = None
    graphs = []

    for index in indices:
        data_path = f'tests/data/graph/enzymes/graph{index:03}/data.pt'
        save_dir = os.path.dirname(data_path)

        if not os.path.exists(data_path):
            if dataset is None:
                # Load dataset once
                from torch_geometric.datasets import TUDataset
                from torch_geometric.utils import to_networkx
                import matplotlib.pyplot as plt
                import networkx as nx

                dataset = TUDataset(root='tests/data/graph', name='ENZYMES')

            os.makedirs(save_dir, exist_ok=True)
            data = dataset[index]
            torch.save(data, data_path)

            # Save graph image
            G = to_networkx(data, to_undirected=True)
            pos = nx.spring_layout(G, seed=42)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.gca()
            nx.draw(G, pos, ax=ax, node_size=150, node_color='lightblue', font_size=6)
            ax.set_axis_off()
            plt.margins(0.0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(os.path.join(save_dir, f"image.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Save info
            info_path = os.path.join(save_dir, f"graph_info.txt")
            with open(info_path, "w") as f:
                f.write(f"Graph Index: {index}\n")
                f.write(f"Number of Nodes: {data.num_nodes}\n")
                f.write(f"Number of Edges: {data.num_edges}\n")
                f.write(f"Graph Label: {data.y.item() if data.y.numel() == 1 else data.y.tolist()}\n")
                f.write(f"Node Feature Dimension: {data.x.shape[1] if data.x is not None else 'None'}\n")
                f.write(f"Edge Feature Dimension: {data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 'None'}\n")

            print(f"Saved graph {index} to {data_path}")

        # Load from file
        data = torch.load(data_path)
        graphs.append(data)

    return Batch.from_data_list(graphs)
