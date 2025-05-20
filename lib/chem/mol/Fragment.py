from __future__ import annotations
from typing import Union, List
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ResonanceMolSupplier, ResonanceFlags
from collections import Counter, OrderedDict

from .Molecule import Molecule
from .Atom import Atom
from .Formula import Formula
from ...ms.Adduct import Adduct
from ...ms.constants import AdductType

# Suppress warnings and informational messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

class Fragment:
    def __init__(self, fragment_molecule: Molecule, adduct_type: AdductType):
        
        self._raw_mol: Molecule = fragment_molecule # original molecule with dummy atoms
        self._stable_mol: Molecule = None # stable molecule without dummy atoms

        self._adducts_in: List[Chem.Mol] = []  # + adducts
        self._adducts_out: List[Chem.Mol] = []  # - adducts
        self._adduct_type: AdductType = adduct_type

        self._reconstruct_fragment()

        self.adduct = self._adduct()

        pass

    def __repr__(self):
        return f"Fragment({Chem.MolToSmiles(self.raw_mol.mol, canonical=True)}, {self.adduct})"


    def _reconstruct_fragment(self):
        """
        Reconstruct a fragmented molecule to satisfy valency and charge stability.
        """
        rw_mol = Chem.RWMol(self._raw_mol.mol)

        dummy_idxs = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_idxs) == 0:
            self._reconstruct_no_bond_fragment(rw_mol)
        elif len(dummy_idxs) == 1:
            self._reconstruct_single_bond_fragment(rw_mol, dummy_idxs[0])
        else:
            raise NotImplementedError(f"Fragment reconstruction for multiple dummy atoms ({len(dummy_idxs)}) is not yet implemented.")

    def _reconstruct_no_bond_fragment(self, rw_mol: Chem.RWMol):
        self.stable_mol = Molecule(rw_mol.GetMol())
        
        if self._adduct_type == AdductType.NONE:
            self._reset_adducts()
        elif self._adduct_type == AdductType.M_PLUS_H_POS:
            self._add_proton_adduct_in(1)
        elif self._adduct_type == AdductType.M_MINUS_H_NEG:
            self._add_proton_adduct_out(1)
        else:
            raise NotImplementedError(f"reconstruct_no_bond_fragment: Unsupported adduct type '{self._adduct_type}'.")

    def _reconstruct_single_bond_fragment(self, rw_mol: Chem.RWMol, dummy_idx: int):
        """
        Reconstruct a fragment with a single broken bond (i.e., one dummy atom).
        """
        dummy_atom = rw_mol.GetAtomWithIdx(dummy_idx)
        neighbors = dummy_atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError("Expected exactly one neighbor for the dummy atom")

        connected_atom = neighbors[0]

        symbol = connected_atom.GetSymbol()
        charge = connected_atom.GetFormalCharge()
        total_valence = connected_atom.GetTotalValence()
        fragmented_valence = total_valence - 1  # Assume one bond is missing

        is_stable = Atom.is_stable(symbol, charge, fragmented_valence)
        if not is_stable:
            if symbol == "C":
                connected_atom.SetFormalCharge(+1)
                rw_mol.RemoveAtom(dummy_idx)
                m = rw_mol.GetMol()
                suppl = ResonanceMolSupplier(m, ResonanceFlags.KEKULE_ALL)
                for s in suppl:
                    is_stable = all(Atom.is_stable(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence()) for atom in s.GetAtoms())
                    total_charge = sum(atom.GetFormalCharge() for atom in s.GetAtoms())
                    if is_stable and abs(total_charge) <= 1:
                        m = s
                        break
                        
                self.stable_mol = Molecule(m)

            elif symbol in ["O", "N", "F", "Cl", "Br", "I"]:
                rw_mol.RemoveAtom(dummy_idx)
                idx = connected_atom.GetIdx()
                h_idx = rw_mol.AddAtom(Chem.Atom(1))  # H atom
                rw_mol.AddBond(idx, h_idx, Chem.BondType.SINGLE)
                m = Chem.RemoveHs(rw_mol)
                suppl = ResonanceMolSupplier(m, ResonanceFlags.KEKULE_ALL)
                if len(suppl) > 0:
                    m = suppl[0]
                self.stable_mol = Molecule(m)
                if self.stable_mol.charge == 0:
                    self._add_proton_adduct_in(1)

            else:
                raise NotImplementedError(f"Reconstruction not implemented for atom type: {symbol}")

    def _reset_adducts(self):
        """
        Reset the adducts of the fragment.
        """
        self._adducts_in = []
        self._adducts_out = []

    def _add_adduct_in(self, mol: Chem.Mol):
        self._adducts_in.append(mol)

    def _add_adduct_out(self, mol: Chem.Mol):
        self._adducts_out.append(mol)

    def _add_proton_adduct_in(self, count: int = 1):
        """
        Add proton(s) [H+] to the fragment as a positive adduct.

        Args:
            count (int): number of protons to add. Default is 1.
        """
        hplus = Chem.MolFromSmiles("[H+]")
        for _ in range(count):
            self._adducts_in.append(hplus)

    def _add_proton_adduct_out(self, count: int = 1):
        """
        Remove proton(s) [H+] from the fragment as a negative adduct.

        Args:
            count (int): number of protons to remove. Default is 1.
        """
        hminus = Chem.MolFromSmiles("[H+]")
        for _ in range(count):
            self._adducts_out.append(hminus)

    @property
    def stable_mol(self) -> Molecule:
        """
        Get the stable molecule without dummy atoms.
        """
        if self._stable_mol is None:
            raise ValueError("Stable molecule has not been reconstructed yet.")
        return self._stable_mol
    
    @stable_mol.setter
    def stable_mol(self, molecule: Molecule):
        """
        Set the stable molecule without dummy atoms.
        """
        assert isinstance(molecule, Molecule), "Stable molecule must be of type Molecule"
        self._stable_mol = molecule
        
    @property
    def raw_mol(self) -> Molecule:
        """
        Get the raw molecule with dummy atoms.
        """
        if self._raw_mol is None:
            raise ValueError("Raw molecule has not been set yet.")
        return self._raw_mol
    
    @raw_mol.setter
    def raw_mol(self, molecule: Molecule):
        """
        Set the raw molecule with dummy atoms.
        """
        assert isinstance(molecule, Molecule), "Raw molecule must be of type Molecule"
        self._raw_mol = molecule

    def _adduct(self) -> Adduct:
        """Return the adduct of the fragment."""
        # Count formulas
        pos_counts = Counter(Formula.from_mol(m) for m in self._adducts_in)
        neg_counts = Counter(Formula.from_mol(m) for m in self._adducts_out)

        element_diff, charge_diff = Formula.from_mol(self._stable_mol.mol).diff(Formula.from_mol(self._raw_mol.mol))

        for f, count in pos_counts.items():
            for e, c in f.elements.items():
                element_diff[e] = element_diff.get(e, 0) + c * count
            charge_diff += f.charge * count
        
        for f, count in neg_counts.items():
            for e, c in f.elements.items():
                element_diff[e] = element_diff.get(e, 0) - c * count
            charge_diff -= f.charge * count

        if self.stable_mol.smiles == self.raw_mol.smiles:
            adduct = Adduct('M', element_diff, charge_diff)
        else:
            adduct = Adduct('F', element_diff, charge_diff)

        return adduct
    
    @property
    def formula(self) -> Formula:
        """
        Get the molecular formula of the fragment.
        """
        formula = self.raw_mol.formula
        for elem, count in self.adduct._element_diff.items():
            f = Formula(elem)
            if count > 0:
                for _ in range(count):
                    formula += f
            elif count < 0:
                for _ in range(-count):
                    formula -= f
        formula.charge = self.raw_mol.charge + self.adduct.charge
        return formula
    
    @property
    def mz(self) -> float:
        """
        Get the mass-to-charge ratio of the fragment.
        """
        total_charge = self.raw_mol.charge + self.adduct.charge
        # total_mass = self.raw_mol.exact_mass + self.adduct.exact_mass
        total_mass = self.formula.exact_mass

        # assert total_charge != 0, "Charge must be non-zero to calculate m/z"
        if total_charge == 0:
            return None

        return total_mass / abs(total_charge)
    
    