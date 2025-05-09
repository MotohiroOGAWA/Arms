from typing import Union
from rdkit import Chem
from ..utilities.Formula import Formula

class Molecule:
    """
    Molecule class to represent a chemical structure.
    """

    def __init__(self, mol: Union[str, Chem.Mol]):
        if isinstance(mol, str):
            # if the input is a SMILES string
            self.smiles = Chem.CanonSmiles(mol)
            self.mol = Chem.MolFromSmiles(self.smiles)
        elif isinstance(mol, Chem.Mol):
            # if the input is an RDKit Mol object
            self.smiles = Chem.MolToSmiles(mol, canonical=True)
            self.mol = Chem.MolFromSmiles(self.smiles) 
        else:
            raise TypeError(f"Unsupported type for Molecule: {type(mol)}")
        
        self.formula = self._formula()
        
    def __repr__(self):
        return f"Molecule(smiles={self.smiles})"
    
    def __str__(self):
        return self.smiles
    
    @property
    def charge(self) -> int:
        """
        Get the charge of the molecule.
        """
        return sum(atom.GetFormalCharge() for atom in self.mol.GetAtoms())
    
    def _formula(self) -> Formula:
        """
        Get the molecular formula of the molecule.
        """
        formula_str = Chem.rdMolDescriptors.CalcMolFormula(self.mol)
        formula = Formula(formula_str)
        return formula
    
    @property
    def exact_mass(self) -> float:
        """
        Get the exact mass of the molecule.
        """
        return self.formula.exact_mass
    
