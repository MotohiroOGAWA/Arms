from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data

from ....cores.MassMolKit.mmkit.fragment.CleavagePatternLibrary import CleavagePatternLibrary, CleavagePattern
from ....cores.MassMolKit.mmkit.chem import Compound
from ....cores.TorchUtils.ModelBase import ModelBase

from .CleavageEncoder import CleavageEncoder

class CleavageScorer(ModelBase):
    def __init__(self,
                 cleavage_pattern_lib_dict: dict,
                 cleavage_dim: int,
                 atom_dim: int,
                 cleavage_fc_dims:Tuple[int],
                 dropout: float,
                 ):
        super(CleavageScorer, self).__init__(
            ignore_config_keys=['cleavage_pattern_lib'],
            **{k: v for k, v in locals().items() if k != 'self'}
        )
        self.cleavage_pattern_lib = CleavagePatternLibrary.from_dict(cleavage_pattern_lib_dict)
        
        self.cleavage_key_to_label = {
            cleavage_pattern.key(): f'Cleave_{i}'
            for i, cleavage_pattern in enumerate(self.cleavage_pattern_lib.patterns)
        }
        self.cleavage_patterns = {
            self.cleavage_key_to_label[cleavage_pattern.key()]: cleavage_pattern
            for cleavage_pattern in self.cleavage_pattern_lib.patterns
        }
        self.cleavage_encoders = nn.ModuleDict({
            self.cleavage_key_to_label[cleavage_pattern.key()]: CleavageEncoder(
                cleavage_pattern=cleavage_pattern,
                cleavage_dim=cleavage_dim,
                atom_dim=atom_dim,
                fc_dims=cleavage_fc_dims,
                dropout=dropout,
            )
            for cleavage_pattern in self.cleavage_pattern_lib.patterns
        })

    def calc_cleavage_score(
            self,
            mol_graph: Data,
        ) -> torch.Tensor:
        """
        Calculate cleavage likelihood scores for all registered cleavage patterns.

        Args:
            mol_graph (torch_geometric.data.Data):
                Graph representation of the molecule.
                Must contain:
                    - x: Node (atom) features [N, atom_dim]
                    - edge_index: Connectivity [2, E]
                    - edge_attr: Edge (bond) features [E, bond_dim]
                Additional fields (e.g., smiles) may be attached.

        Returns:
            torch.Tensor: Tensor of shape [num_patterns],
                        each entry represents the predicted cleavage score.
        """
        if not hasattr(mol_graph, 'compound'):
            if not hasattr(mol_graph, 'smiles'):
                raise ValueError("mol_graph must have either 'compound' or 'smiles' attribute.")
            else:
                compound = Compound.from_smiles(mol_graph.smiles)
        else:
            compound = mol_graph.compound

        atom_feats = mol_graph.x  # [N_atoms, atom_dim]

        for label, cleavage_pattern in self.cleavage_patterns.items():
            cleavage_encoder = self.cleavage_encoders[label]

            if not cleavage_pattern.exists(compound):
                continue  # Skip patterns that do not exist in the compound

            matches = cleavage_pattern.matches(compound)
            for m in matches:
                reactant_atom_indices = torch.tensor(m, device=atom_feats.device)
                reactant_atom_feats = atom_feats[reactant_atom_indices]  # [num_reactant_atoms, atom_dim]
                
                cleavage_feat = cleavage_encoder(reactant_atom_feats)  # [cleavage_dim]
                pass
            

