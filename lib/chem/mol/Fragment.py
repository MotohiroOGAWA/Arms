from __future__ import annotations
from typing import Union, List
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ResonanceMolSupplier, ResonanceFlags
from collections import Counter, deque

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

        # is_stable = Atom.is_stable(symbol, charge, fragmented_valence)
        total_charge = sum(atom.GetFormalCharge() for atom in rw_mol.GetAtoms())

        if (total_charge == 0) and (symbol == "C"):
            rw_mol.RemoveAtom(dummy_idx)
            connected_atom.SetFormalCharge(+1)
            routes = self._find_all_charge_shift_path(rw_mol, connected_atom.GetIdx())
            rw_mol = self._shift_charge_along_path(rw_mol, routes[0]["path"])
            m = rw_mol.GetMol()
                    
            self.stable_mol = Molecule(m)

        elif symbol in ["C", "O", "N", "F", "Cl", "Br", "I"]:
            rw_mol.RemoveAtom(dummy_idx)
            idx = connected_atom.GetIdx()
            h_idx = rw_mol.AddAtom(Chem.Atom(1))  # H atom
            rw_mol.AddBond(idx, h_idx, Chem.BondType.SINGLE)
            m = Chem.RemoveHs(rw_mol)
            self.stable_mol = Molecule(m)

        else:
            raise NotImplementedError(f"Reconstruction not implemented for atom type: {symbol}")

        if self.stable_mol.charge == 0:
            if self._adduct_type == AdductType.M_PLUS_H_POS:
                self._add_proton_adduct_in(1)
            else:
                raise NotImplementedError(
                    f"reconstruct_single_bond_fragment: Unsupported adduct type '{self._adduct_type}' for stable molecule with no charge."
                )


    def _find_all_charge_shift_path(self, rw_mol: Chem.RWMol, start_idx: int) -> List[int]:
        start_atom = rw_mol.GetAtomWithIdx(start_idx)
        if start_atom.GetFormalCharge() == 0:
            return 
        
        if start_atom.GetSymbol() == "O":
            return
        if start_atom.GetSymbol() == "N":
            return
        
        queue = deque([(start_idx, [start_idx])])  # (current index, path)
        completed_paths = [{
            "path": [start_idx],
            "symbol": start_atom.GetSymbol(),
            "carbon_degree": sum(1 for n in start_atom.GetNeighbors() if n.GetSymbol() == "C")
        }]
        while queue:
            current_idx, path = queue.popleft()

            current_atom = rw_mol.GetAtomWithIdx(current_idx)
            
            for neighbor in current_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in path:
                    continue
                
                bond = rw_mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
                if neighbor.GetSymbol() in ["O", "N"] \
                    and bond.GetBondType() in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC]:
                    carbon_degree = sum(1 for n in neighbor.GetNeighbors() if n.GetSymbol() == "C")
                    completed_paths.append({
                        "path": path + [neighbor_idx],
                        "symbol": neighbor.GetSymbol(),
                        "carbon_degree": carbon_degree
                    })
                    continue
                
                elif (bond.GetBondType() == Chem.BondType.SINGLE or bond.GetBondType() == Chem.BondType.AROMATIC)\
                    and neighbor.GetSymbol() == "C"\
                          and neighbor.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                    # If the neighbor is a carbon with SP2 hybridization, continue the search
                    for next_neighbor in neighbor.GetNeighbors():
                        next_idx = next_neighbor.GetIdx()
                        if next_idx == current_idx:
                            continue

                        next_bond = rw_mol.GetBondBetweenAtoms(neighbor_idx, next_idx)
                        if (next_bond.GetBondType() == Chem.BondType.DOUBLE or next_bond.GetBondType() == Chem.BondType.AROMATIC)\
                              and next_neighbor.GetSymbol() == "C":
                            queue.append((next_idx, path + [neighbor_idx, next_idx]))
                            completed_paths.append({
                                "path": path + [neighbor_idx, next_idx],
                                "symbol": next_neighbor.GetSymbol(),
                                "carbon_degree": sum(1 for n in next_neighbor.GetNeighbors() if n.GetSymbol() == "C")
                            })

        # Sort the completed paths based on priority:
        # 1. Prioritize symbols "O" and "N" over "C"
        # 2. Prefer higher carbon degree
        # 3. Prefer shorter paths
        def priority(entry):
            # Assign lower priority value to "O" and "N", higher to "C", and highest to others
            symbol_priority = 0 if entry["symbol"] in ["O", "N"] else (1 if entry["symbol"] == "C" else 2)
            return (
                symbol_priority,               # Lower is better (O/N preferred)
                -entry["carbon_degree"],       # Higher carbon degree is better
                len(entry["path"])             # Shorter path is better
            )

        # Apply sorting based on the defined priority
        completed_paths = sorted(completed_paths, key=priority)
        return completed_paths

    def _shift_charge_along_path(self, rw_mol: Chem.RWMol, path: List[int]):
        if len(path) < 2:
            return rw_mol # Cannot shift with a single atom

        current_i = 0
        while current_i < len(path) - 1:
            src_idx = path[current_i]
            dst_idx = path[current_i + 1]

            bond = rw_mol.GetBondBetweenAtoms(src_idx, dst_idx)
            src_atom = rw_mol.GetAtomWithIdx(src_idx)
            dst_atom = rw_mol.GetAtomWithIdx(dst_idx)

            if src_atom.GetSymbol() == 'C' and src_atom.GetFormalCharge() == 1:
                if dst_atom.GetSymbol() == 'C' \
                    and dst_atom.GetFormalCharge() == 0 \
                        and bond.GetBondType() in [Chem.BondType.SINGLE, Chem.BondType.AROMATIC]:
                    
                    nbr_idx = path[current_i + 2]
                    nbr_atom = rw_mol.GetAtomWithIdx(nbr_idx)
                    nbr_bond = rw_mol.GetBondBetweenAtoms(dst_idx, nbr_idx)
                    if nbr_atom.GetSymbol() == 'C' \
                        and nbr_atom.GetFormalCharge() == 0 \
                            and nbr_bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.AROMATIC]:
                        # C+–C=C → C=C–C+
                        rw_mol.RemoveBond(src_idx, dst_idx)
                        rw_mol.RemoveBond(dst_idx, nbr_idx)
                        src_atom.SetFormalCharge(0)
                        nbr_atom.SetFormalCharge(1)
                        rw_mol.AddBond(dst_idx, nbr_idx, Chem.BondType.SINGLE)
                        rw_mol.AddBond(src_idx, dst_idx, Chem.BondType.DOUBLE)
                        
                        current_i += 2
                    else:
                        raise NotImplementedError(
                            f"Charge shift not implemented for bond type {bond.GetBondType()} between {src_atom.GetSymbol()} and {dst_atom.GetSymbol()}"
                        )
                elif dst_atom.GetSymbol() in ['O', 'N'] \
                    and dst_atom.GetFormalCharge() == 0:
                    # C+–O → C=O+
                    if bond.GetBondType() in [Chem.BondType.SINGLE, Chem.BondType.AROMATIC]:
                        _bond_type = Chem.BondType.DOUBLE
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        _bond_type = Chem.BondType.TRIPLE
                    else:
                        raise NotImplementedError(
                            f"Charge shift not implemented for bond type {bond.GetBondType()} between {src_atom.GetSymbol()} and {dst_atom.GetSymbol()}"
                        )
                
                    rw_mol.RemoveBond(src_idx, dst_idx)
                    src_atom.SetFormalCharge(0)
                    dst_atom.SetFormalCharge(1)
                    rw_mol.AddBond(src_idx, dst_idx, _bond_type)
                    current_i += 1

                else:
                    raise NotImplementedError(
                        f"Charge shift not implemented for bond type {bond.GetBondType()} between {src_atom.GetSymbol()} and {dst_atom.GetSymbol()}"
                    )
            else:
                raise NotImplementedError(
                    f"Charge shift not implemented for atom {src_atom.GetSymbol()} with charge {src_atom.GetFormalCharge()}"
                )
        Chem.SanitizeMol(rw_mol)
        return rw_mol


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
    
    