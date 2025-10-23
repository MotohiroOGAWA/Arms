import torch
from rdkit import Chem

from ..cores.MassEntity.msentity.core import SpecCondition, MSDataset

class AllowedAtomsCondition(SpecCondition):
    """
    Select spectra whose SMILES contain only allowed atoms.

    Args:
        allowed_atoms (List[str]): List of allowed atom symbols (e.g. ["C","H","O","N"]).
        smiles_column (str): Column name containing SMILES strings in dataset metadata.
    """
    def __init__(self, allowed_atoms: list[str] = None, smiles_column: str = "SMILES"):

        self.smiles_column = smiles_column
        self.allowed_atoms = set(allowed_atoms) if allowed_atoms is not None else set()

    def evaluate(self, ds: MSDataset) -> torch.BoolTensor:
        # get SMILES column from dataset
        smiles_list = ds.meta[self.smiles_column].tolist()
        mask = []

        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    mask.append(False)
                    continue

                atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
                mask.append(atoms.issubset(self.allowed_atoms))
            except Exception:
                mask.append(False)

        return torch.tensor(mask, dtype=torch.bool, device=ds.device)