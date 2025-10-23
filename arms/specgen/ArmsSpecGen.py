import torch
import torch.nn as nn
from typing import Tuple, Dict

from ..cores.MassMolKit.mmkit.chem.Compound import Compound

from ..cores.TorchUtils.ModelBase import ModelBase
from .components.encoders.MolEncoder import MolEncoder
from .components.encoders.CleavageScorer import CleavageScorer


class ArmsSpecGen(ModelBase):
    def __init__(self,
                 symbols:Tuple[str], # ('C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I')
                 cleavage_scorer_params:Dict,
                 dropout: float,
                 ):
        super(ArmsSpecGen, self).__init__(
            ignore_config_keys=[],
            **{k: v for k, v in locals().items() if k != 'self'}
        )

        self.mol_encoder = MolEncoder(symbols=symbols)

        cleavage_scorer_params['atom_dim'] = self.mol_encoder.atom_dim
        cleavage_scorer_params['dropout'] = dropout
        self.cleavage_scorer = CleavageScorer.from_params(cleavage_scorer_params)
        pass

    def forward(self, compound: Compound):
        mol_graph = self.mol_encoder.encode(compound)
        cleavage_scores = self.cleavage_scorer.calc_cleavage_score(mol_graph)
        return mol_graph, cleavage_scores