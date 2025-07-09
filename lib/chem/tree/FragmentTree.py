import os
from typing import Union, List, Dict, Tuple, Literal
from collections import defaultdict
import dill
import copy
from enum import Enum
import networkx as nx

from lib import Molecule
from ...ms.Adduct import Adduct
from ...ms.AdductedIon import AdductedIon
from ..mol.Formula import Formula
from .BondPosition import BondPosition


class FragmentTree:
    """
    FragmentTree class to represent a tree of fragments.
    """
    class StructureType(Enum):
        """
        Enum to represent the structure type of the fragment tree.
        """
        ORIGINAL = "original"
        TOPOLOGICAL = "topological"

    def __init__(self, molecule: Molecule, nodes: List['FragmentNode'], edges: List['FragmentEdge']):
        self.molecule = molecule
        self.nodes: List[FragmentNode] = nodes
        self.edges: List[FragmentEdge] = edges

    def __repr__(self):
        return f"FragmentTree(molecule={self.molecule.smiles}, nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    def get_all_formulas(self, sources: bool = False) -> Union[Tuple[Formula, ...], Dict[Formula, List[str]]]:
        """
        Get all formulas from the fragment tree.
        """
        if sources:
            formulas = defaultdict(list)
        else:
            formulas = set()

        for node in self.nodes:
            for adduct in node.adducts:
                adducted_ion = AdductedIon(
                    fragment_molecule=Molecule(node.smiles),
                    adduct=adduct,
                )
                if sources:
                    formulas[adducted_ion.formula].append(f'{node.smiles}|{adduct}')
                else:
                    formulas.add(adducted_ion.formula)
        if sources:
            # Convert defaultdict to a regular dict for consistent output
            formulas = [(k, v) for k, v in formulas.items()]
            formulas.sort(key=lambda x: x[0].exact_mass)
            formulas = {k: v for k, v in formulas}
            return formulas
        else:
            formulas = list(formulas)
            formulas.sort(key=lambda x: x.exact_mass)
            return tuple(formulas)
    
    def save(self, path: str):
        """
        Save the fragment tree to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dill.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path: str) -> 'FragmentTree':
        """
        Load a fragment tree from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FragmentTree file not found: {path}")
        return dill.load(open(path, 'rb'))
    
    def get_node_count(self) -> int:
        """
        Get the number of nodes in the fragment tree.
        """
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """
        Get the number of edges in the fragment tree.
        """
        return len(self.edges)

    def rebuild_tree_topologically(self) -> 'FragmentTree':
        new_edges = []
        node_depths = {}
        node_depths[0] = 0
        for edge in self.edges:
            if edge.source_id not in node_depths:
                raise ValueError(f"Source ID {edge.source_id} not found")
            
            depth = node_depths[edge.source_id] + 1
            if edge.target_id in node_depths:
                if depth < node_depths[edge.target_id]:
                    raise ValueError("Ordering error: target depth is greater than source depth")
                elif depth == node_depths[edge.target_id]:
                    pass
                else:
                    continue  # Skip this edge if the target depth is less than source depth
            else:
                node_depths[edge.target_id] = depth
            
            new_edges.append(edge.copy())
        
        tree = FragmentTree(
            molecule=copy.deepcopy(self.molecule),
            nodes=copy.deepcopy(self.nodes),
            edges=new_edges
        )
        return tree, {'depth': node_depths}
        
    
    def save_tsv(self, node_path: str, edge_path: str, structure_type: StructureType = StructureType.ORIGINAL):
        """
        Save FragmentTree to two TSV files (Cytoscape-compatible):
        - nodes.tsv: id, smiles
        - edges.tsv: source, target, fragment_index, bond_positions, attribute

        Parameters:
            structure_type (StructureType): which structure to export (original or topological)
        """
        if structure_type == FragmentTree.StructureType.ORIGINAL:
            # Use the original tree
            tree = self
            info = None
        elif structure_type == FragmentTree.StructureType.TOPOLOGICAL:
            tree, info = self.rebuild_tree_topologically()
        else:
            raise NotImplementedError(f"Structure type {structure_type} is not implemented")
        
        # Save nodes
        os.makedirs(os.path.dirname(node_path), exist_ok=True)
        with open(node_path, mode='w', newline='', encoding='utf-8') as f:
            f.write(FragmentNode.header() + "\n")
            for i, node in enumerate(tree.nodes):
                f.write(f"{i}\t{node.to_tsv()}\n")

        # Save edges
        os.makedirs(os.path.dirname(edge_path), exist_ok=True)
        with open(edge_path, mode='w', newline='', encoding='utf-8') as f:
            f.write(FragmentEdge.header() + "\n")
            for edge in tree.edges:
                f.write(edge.to_tsv() + "\n")

        return tree, info
    
    def save_topological_tsv(self, node_path: str, edge_path: str, layer_scale: float = 10.0, iterations: int = 1000):
        """
        Save FragmentTree to two TSV files (Cytoscape-compatible) in topological order.
        """
        tree, info = self.save_tsv(node_path, edge_path, structure_type=FragmentTree.StructureType.TOPOLOGICAL)
        G = nx.DiGraph()
        G.add_edges_from((edge.source_id, edge.target_id) for edge in tree.edges)
        depth = info['depth']
        max_depth = max(depth.values())

        # Add a virtual node to pull everything downward
        # Connect all real nodes to the virtual node
        # Use larger weights for deeper nodes to pull them harder
        virtual_node = -1
        G.add_node(virtual_node)

        for node, d in depth.items():
            G.add_edge(node, virtual_node, weight=(d + 1)/(max_depth + 1))

        # Set fixed positions for root and virtual node
        fixed = [0, virtual_node]
        pos = {
            0: (0, 0),               # Root node fixed at top
            virtual_node: (0, (max_depth+1) * layer_scale)  # Virtual node fixed far below
        }


        # Get 1D spring layout
        pos_1d = nx.spring_layout(G, dim=2, pos=pos, fixed=fixed, weight='weight', iterations=iterations)
        pos_2d = {
            node: (float(pos_1d[node][0]), depth[node] * layer_scale)
            for node in G.nodes if node != virtual_node
        }

        x_by_depth = defaultdict(list)
        for node_id, (x, _) in pos_2d.items():
            depth = info['depth'][node_id]
            x_by_depth[depth].append(x)

        max_width = max(
            (max(xs) - min(xs)) for xs in x_by_depth.values() if len(xs) > 1
        )
        max_x_cnt = max(
            len(xs) for xs in x_by_depth.values() if len(xs) > 1
        )

        x_scale = max_x_cnt * layer_scale / max_width
        for node_id, (x, y) in pos_2d.items():
            pos_2d[node_id] = (x * x_scale, y)

        node_filename, _ = os.path.splitext(node_path)
        pos_path = node_filename + "_positions.tsv"

        with open(pos_path, mode='w', newline='', encoding='utf-8') as f:
            f.write("id\tx\ty\n")
            for node_id, (x, y) in pos_2d.items():
                f.write(f"{node_id}\t{x}\t{y}\n")


class FragmentNode:
    """
    FragmentNode class to represent a node in the fragment tree.
    """
    def __init__(self, smiles: str):
        self.smiles: str = smiles
        self.adducts: Tuple[Adduct] = ()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"FragmentNode(smiles={self.smiles}, adducts={'|'.join(str(adduct) for adduct in self.adducts)})"    
    
    def copy(self) -> 'FragmentNode':
        """
        Create a copy of the FragmentNode.
        """
        return FragmentNode(
            smiles=self.smiles,
            adducts=tuple(adduct.copy() for adduct in self.adducts)
        )

    @staticmethod
    def header() -> str:
        """
        Return the header for the TSV representation of FragmentNode.
        """
        return "id\tsmiles\tadducts"

    def to_tsv(self):
        """
        Convert the FragmentNode to a TSV string.
        """
        return f"{self.smiles}\t{'|'.join(str(adduct) for adduct in self.adducts)}"

class FragmentEdge:
    def __init__(
        self,
        source_id: int,
        target_id: int,
        bond_positions: Tuple['BondPosition'],
        fragment_index: int,
        attribute: Dict = {},
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.bond_positions = bond_positions
        self.fragment_index = fragment_index
        self.attribute = attribute

    def __eq__(self, other):
        if not isinstance(other, FragmentEdge):
            raise TypeError(f"Cannot compare FragmentEdge with {type(other).__name__}")
        return (
            self.source_id == other.source_id and
            self.target_id == other.target_id and
            self.bond_positions == other.bond_positions and
            self.fragment_index == other.fragment_index and
            self.attribute == other.attribute
        )

    def __hash__(self):
        return hash((
            self.source_id,
            self.target_id,
            self.bond_positions,
            self.fragment_index, 
            frozenset(self.attribute.items())
        ))

    def __repr__(self):
        return (
            f"FragmentEdge({self.source_id} -> {self.target_id}, "
            f"bonds={self.bond_positions}, fragment_index={self.fragment_index}, "
            f"attr={self.attribute})"
        )
    
    def copy(self) -> 'FragmentEdge':
        """
        Create a copy of the FragmentEdge.
        """
        return FragmentEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            bond_positions=self.bond_positions,
            fragment_index=self.fragment_index,
            attribute=self.attribute.copy()
        )
    
    @staticmethod
    def header(attributes: Tuple[str] = ('FragmentType', 'Adduct')) -> str:
        """
        Return the header for the TSV representation of FragmentEdge.
        """
        attr_header = "\t".join(attributes)
        return f"Source\tTarget\tFragmentIdx\tBond\t{attr_header}"

    def to_tsv(self, attributes: Tuple[str] = ('fragment_type', 'adduct')) -> str:
        """
        Convert the FragmentEdge to a TSV string.
        """
        bond_str = '(' + ','.join([f'({bond_position[0]},{bond_position[1]})' for bond_position in self.bond_positions]) + ')'
        attr_str = '\t'.join(
            str(self.attribute.get(attr, '')) for attr in attributes
        )
        return f"{self.source_id}\t{self.target_id}\t{self.fragment_index}\t{bond_str}\t{attr_str}"

