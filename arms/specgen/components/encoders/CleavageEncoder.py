from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....cores.MassMolKit.mmkit.fragment.CleavagePattern import CleavagePattern
from ....cores.TorchUtils.nn_utils import build_fc_layers

class CleavageEncoder(nn.Module):

    def __init__(self, 
                 cleavage_pattern: CleavagePattern,
                 cleavage_dim: int,
                 atom_dim: int,
                 fc_dims:Tuple[int],
                 mol_dim: int,
                 mol_fc_dims:Tuple[int],
                 dropout: float,
                 ):
        super(CleavageEncoder, self).__init__()

        self.cleavage_dim = cleavage_dim
        self.atom_dim = atom_dim
        self.mol_dim = mol_dim
        
        self.num_react_atoms = cleavage_pattern.num_reactant_atoms
        self.num_prod_atoms = cleavage_pattern.num_product_atoms

        self.encoder = build_fc_layers(
            input_dim = atom_dim * (self.num_react_atoms + self.num_prod_atoms),
            fc_dims=fc_dims,
            output_dim=cleavage_dim,
            dropout=dropout,
        )

        self.fc_mol = build_fc_layers(
            input_dim = cleavage_dim,
            fc_dims = mol_fc_dims,
            output_dim = mol_dim,
            dropout = dropout,
        )

    def forward(self, reactant_atom_feats: torch.Tensor) -> torch.Tensor:
        return self.encode_cleavage(reactant_atom_feats)

    def encode_cleavage(self, reactant_atom_feats: torch.Tensor) -> torch.Tensor:
        """
        Encode cleavage features from reactant atom features.

        Args:
            reactant_atom_feats (torch.Tensor): Tensor of shape
                [num_reactant_atoms, atom_dim].

        Returns:
            torch.Tensor: Encoded cleavage features of shape [cleavage_dim].
        """
        assert reactant_atom_feats.shape == (self.num_react_atoms, self.atom_dim), \
            f"Expected reactant_atom_feats shape {(self.num_react_atoms, self.atom_dim)}, got {reactant_atom_feats.shape}"
        flat_reactant_feats = reactant_atom_feats.view(-1)  # Flatten to 1D tensor
        cleavage_feat = self.encoder(flat_reactant_feats)  # [cleavage_dim]
        return cleavage_feat


    def predict_mol_features(
        self,
        cleavage_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict molecular-level feature representation from a cleavage embedding.
        """
        assert cleavage_feat.shape == (self.cleavage_dim,), \
            f"Expected cleavage_feat shape {(self.cleavage_dim,)}, got {cleavage_feat.shape}"
        mol_feat_pred = self.fc_mol(cleavage_feat)
        return mol_feat_pred


    def compute_alignment_loss(
        self,
        cleavage_feat: torch.Tensor,
        target_mol_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity loss between predicted and target molecule features.
        """
        mol_feat_pred = self.predict_mol_features(cleavage_feat)
        return F.mse_loss(mol_feat_pred, target_mol_feat)
