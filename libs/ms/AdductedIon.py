from rdkit import Chem
from typing import Union
from .Adduct import Adduct
from ..chem.mol.Formula import Formula
from ..chem.mol.Molecule import Molecule


# Suppress warnings and informational messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

class AdductedIon:
    def __init__(self, fragment_molecule: Molecule, adduct: Adduct, overwrite_atom_map: bool = True):
        atom_map_dict = {}
        fragment_molecule.with_atom_map(inplace=True, overwrite=overwrite_atom_map, atom_map_dict=atom_map_dict)
        self.molecule = fragment_molecule
        self.adduct = adduct
        self.atom_map_dict = atom_map_dict

        pass

    def __repr__(self):
        return f"AdductedIon({self.molecule.smiles}, {self.adduct})"
    
    @property
    def formula(self) -> Formula:
        """
        Get the molecular formula of the fragment with the adduct applied.
        """
        base_formula = self.molecule.formula.copy()
        for elem, delta in self.adduct.element_diff.items():
            base_formula.elements[elem] = base_formula.elements.get(elem, 0) + delta
            if base_formula.elements[elem] == 0:
                del base_formula.elements[elem]  # remove zero-count elements

        base_formula.charge = self.adduct.charge
        base_formula.raw_formula = base_formula.to_string(no_charge=True)
        return base_formula
    
    @property
    def charge(self) -> int:
        """
        Get the charge of the fragment with the adduct applied.
        """
        return self.adduct.charge
        
    
    @property
    def mz(self) -> float:
        """
        Get the mass-to-charge ratio of the fragment.
        """
        total_charge = self.charge
        total_mass = self.formula.exact_mass

        # assert total_charge != 0, "Charge must be non-zero to calculate m/z"
        if total_charge == 0:
            raise ValueError("Charge must be non-zero to calculate m/z")

        return total_mass / abs(total_charge)
    
    