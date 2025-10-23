from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data
from rdkit import Chem

from ....cores.MassMolKit.mmkit.chem import Compound
from .AtomEncoder import AtomEncoder
from .BondEncoder import BondEncoder


class MolEncoder(nn.Module):
    """
    MolEncoder converts an RDKit Mol into numerical feature tensors
    using AtomEncoder and BondEncoder.

    Returns:
        - node_features: torch.Tensor of shape [num_atoms, atom_feature_dim]
        - edge_features: torch.Tensor of shape [num_bonds, bond_feature_dim]
        - edge_index: torch.LongTensor of shape [2, num_bonds * 2]
          (bi-directional edge index)
    """

    def __init__(
            self,
            symbols,
            ):
        super(MolEncoder, self).__init__()

        self.atom_encoder = AtomEncoder(symbols=symbols)
        self.bond_encoder = BondEncoder()

    def encode(self, compound: Compound) -> Data:
        """
        Encode a Compound into a PyTorch Geometric Data object.

        Args:
            compound (Compound): Molecule wrapper containing RDKit Mol.

        Returns:
            torch_geometric.data.Data: Graph data with x, edge_index, edge_attr.
        """
        node_features, edge_features, edge_index = self.encode_components(compound)

        data = Data(
            x=node_features,        # [N, atom_dim]
            edge_index=edge_index,  # [2, E]
            edge_attr=edge_features # [E, bond_dim]
        )

        # Attach metadata
        data.num_nodes = node_features.size(0)
        data.compound = compound
        data.smiles = compound.smiles
        return data

    def encode_components(self, compound: Compound) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return individual molecular graph components as tensors.

        Args:
            compound (Compound): Input molecule as a Compound object.

        Returns:
            (node_features, edge_features, edge_index)
        """
        mol = compound.mol
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # === Node (Atom) features ===
        atom_features = []
        for atom in mol.GetAtoms():
            assert atom.GetIdx() == len(atom_features), "Atom indices are not sequential."
            atom_features.append(self.atom_encoder.encode(atom))
        node_features = torch.stack(atom_features, dim=0)  # [N, atom_feature_dim]

        # === Edge (Bond) features ===
        bond_features = []
        edge_indices = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_vec = self.bond_encoder.encode(bond)
            bond_features.append(bond_vec)
            # Undirected → register both (i→j) and (j→i)
            edge_indices.append((i, j))
            edge_indices.append((j, i))
            bond_features.append(bond_vec.clone())  # same bond feature for reverse direction

        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T  # [2, E]
        edge_features = torch.stack(bond_features, dim=0)  # [E, bond_feature_dim]

        return node_features, edge_features, edge_index

    # ------------------------------------------------------------------
    @property
    def atom_dim(self) -> int:
        """Return atom feature dimension."""
        return self.atom_encoder.feature_dim

    @property
    def bond_dim(self) -> int:
        """Return bond feature dimension."""
        return self.bond_encoder.feature_dim
