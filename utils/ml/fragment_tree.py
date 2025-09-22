import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from cores.MassMolKit.Mol.Compound import Compound
from cores.MassMolKit.Fragment.FragmentTree import FragmentTree
from cores.MassMolKit.Fragment.Fragmenter import Fragmenter, AdductType
from cores.MassEntity.MassEntityCore.MSDataset import *
from .MlFragmentTree import MlFragmentTree

def fragmenttree_to_pyg(record: SpectrumRecord, smiles_column:str='SMILES', fragmenter: Fragmenter=None, tree: FragmentTree=None) -> Data:
    """
    Convert FragmentTree into a PyTorch Geometric Data object.
    
    Args:
        tree (FragmentTree): Fragment tree object
        fp_dim (int): Fingerprint dimension for node features
    
    Returns:
        torch_geometric.data.Data
    """
    if tree is None:
        compound = Compound.from_smiles(record[smiles_column])
        _tree = fragmenter.create_fragment_tree(compound)
        tree, _ = _tree.rebuild_tree_topologically()
    ml_tree = MlFragmentTree.from_fragment_tree(tree)
    pass


if __name__ == "__main__":
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    file_path = 'data/raw/MoNA/positive/final_assign_mona_positive.hdf5'
    dataset = MSDataset.from_hdf5(file_path)
    record = dataset[dataset['Canonical'] == 'C#CC(C)N(C)C(=O)Nc1ccc(Cl)cc1'][0]
    fragmenter = Fragmenter(adduct_type=(AdductType.M_PLUS_H_POS,), max_depth=3)

    fragmenttree_to_pyg(record, smiles_column='Canonical', fragmenter=fragmenter)