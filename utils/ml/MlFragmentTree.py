from typing import List, Dict, Any, Union
import torch
from torch_geometric.data import Data

from cores.MassMolKit.Fragment.FragmentTree import *
from cores.MassEntity.MassEntityCore.MSDataset import *

class MlFragmentTree(FragmentTree):
    def __init__(self, compound: Compound, nodes: List[FragmentNode], edges: List[FragmentEdge]):
        super(MlFragmentTree, self).__init__(compound, nodes, edges)

    @classmethod
    def from_fragment_tree(cls, tree: FragmentTree) -> 'MlFragmentTree':
        return cls(tree.compound, tree.nodes, tree.edges)

    def to_pyg(self, fp_dim: int = 128) -> Data:
        pass
    