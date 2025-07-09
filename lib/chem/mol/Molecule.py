from typing import Union, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import ResonanceMolSupplier, ResonanceFlags
from .Formula import Formula
from ..tree.BondPosition import BondPosition
from typing import Optional
from collections import defaultdict
from bidict import bidict

class Molecule:
    """
    Molecule class to represent a chemical structure.
    """

    def __init__(self, mol: Union[str, Chem.Mol], overwrite_atom_map: bool = False):
        if isinstance(mol, str):
            # if the input is a SMILES string
            smiles = Chem.CanonSmiles(mol)
            self._mol = Chem.MolFromSmiles(smiles)
        elif isinstance(mol, Chem.Mol):
            # if the input is an RDKit Mol object
            smiles = Chem.MolToSmiles(mol, canonical=True)
            self._mol = Chem.MolFromSmiles(smiles) 
        else:
            raise TypeError(f"Unsupported type for Molecule: {type(mol)}")
        
        self.with_atom_map(inplace=True, overwrite=overwrite_atom_map)
        self.formula = self._formula()
        
    def __repr__(self):
        return f"Molecule(smiles={self.smiles})"
    
    def __str__(self):
        return self.smiles
    
    @property
    def mol(self) -> Chem.Mol:
        """
        Get the RDKit Mol object of the molecule.
        """
        _mol = Chem.Mol(self._mol)  # Create a copy to avoid modifying the original
        for atom in _mol.GetAtoms():
            atom.SetAtomMapNum(0)  # Reset atom map numbers to 0
        return _mol
    
    @property
    def smiles(self) -> str:
        """
        Get the SMILES representation of the molecule.
        """
        return Chem.MolToSmiles(self.mol, canonical=True)
    
    @property
    def _smiles(self) -> str:
        """
        Get the SMILES with atom map of the molecule.
        """
        return Chem.MolToSmiles(self._mol, canonical=True)
    
    @property
    def atom_map_to_idx(self) -> bidict[int, int]:
        """
        Get a mapping from atom map numbers to canonical atom indices.
        This maps the atom map numbers assigned before canonicalization
        to the new atom indices after canonical SMILES generation.
        """
        _mol = Chem.Mol(self._mol)  # Create a copy to avoid modifying the original

        # Preserve original atom map numbers
        old_atom_map_num = {atom.GetIdx(): atom.GetAtomMapNum() for atom in _mol.GetAtoms() if atom.GetAtomMapNum() > 0}

        # Remove atom map numbers for canonicalization
        for atom in _mol.GetAtoms():
            atom.SetAtomMapNum(0)

        # Generate canonical SMILES (this triggers atom ordering)
        Chem.MolToSmiles(_mol, canonical=True)

        # Retrieve canonical atom order
        atom_order = list(map(int, _mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))

        # Build mapping: atom_map_num → new atom index
        atom_map_to_idx = {}
        for new_idx, old_idx in enumerate(atom_order):
            atom_map_num = old_atom_map_num.get(old_idx, None)
            if atom_map_num is not None:
                atom_map_to_idx[atom_map_num] = new_idx

        return bidict(atom_map_to_idx)
    
    @property
    def charge(self) -> int:
        """
        Get the charge of the molecule.
        """
        return sum(atom.GetFormalCharge() for atom in self._mol.GetAtoms())
    
    def _formula(self) -> Formula:
        """
        Get the molecular formula of the molecule.
        """
        formula_str = Chem.rdMolDescriptors.CalcMolFormula(self._mol)
        formula = Formula(formula_str)
        return formula
    
    @property
    def exact_mass(self) -> float:
        """
        Get the exact mass of the molecule.
        """
        return self.formula.exact_mass

    def with_atom_map(self, inplace: bool = False, overwrite: bool = False, atom_map_dict: Optional[Dict[int, int]] = None) -> Optional['Molecule']:
        """
        Add atom map numbers to atoms and optionally track old->new mapping.

        Args:
            inplace (bool): Modify molecule in place if True.
            overwrite (bool): If True, overwrite all existing map numbers.
            atom_map_dict (dict, optional): Dict to store old->new atom map number mappings.

        Returns:
            Molecule or None: Modified molecule if not inplace, otherwise None.
        """
        assert (atom_map_dict is None) or (isinstance(atom_map_dict, dict) and len(atom_map_dict) == 0), "atom_map_dict must be a dict or empty if provided"

        # Use a copy or the original mol
        mol = self._mol if inplace else Chem.Mol(self._mol)
        n_atoms = mol.GetNumAtoms()
        
        used_map_nums = set()
        if not overwrite:
            used_map_nums = {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            max_map_num = max(used_map_nums, default=0)
            diff = set(range(1, max_map_num)).difference(used_map_nums)
            next_map_nums = list(diff) + list(range(max_map_num + 1, max_map_num + n_atoms + 1))
        else:
            next_map_nums = list(range(1, n_atoms + 1))
        
        for atom in mol.GetAtoms():
            if overwrite or atom.GetAtomMapNum() == 0:
                old_map_num = atom.GetAtomMapNum()
                new_map_num = next_map_nums.pop(0)
                atom.SetAtomMapNum(new_map_num)
                if atom_map_dict is not None and old_map_num > 0:
                    atom_map_dict[old_map_num] = new_map_num

        if inplace:
            self._mol = mol
        else:
            return Molecule(mol)

    def copy(self) -> 'Molecule':
        """
        Create a copy of the molecule.
        """
        return Molecule(self.smiles)
    
    def get_atom_index_from_map(self, map_num: int) -> Optional[int]:
        """
        Get the atom index from a given atom map number.
        """
        for atom in self._mol.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                return atom.GetIdx()
        return None  # Not found
    
    def _split_simple_bond(self, bond_position: BondPosition) -> tuple[Chem.Mol, ...]:
        rw_mol = Chem.RWMol(self._mol)
        
        if isinstance(bond_position, BondPosition):
            atom_idx1 = self.get_atom_index_from_map(bond_position[0])
            atom_idx2 = self.get_atom_index_from_map(bond_position[1])
            if atom_idx1 is None or atom_idx2 is None:
                raise ValueError("Invalid bond position: atom map numbers not found in molecule")
            bond = self._mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
            
            if bond is None:
                raise ValueError("No bond found between the specified atom indices")
            
            fragmented = Chem.FragmentOnBonds(rw_mol, [bond.GetIdx()], addDummies=True)
            frags: tuple[Chem.Mol, ...] = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)

            return frags
            
        else:
            raise TypeError("bond_position must be an instance of BondPosition")
        
    def fragment_non_ring_single_bonds(self, ignore_bond_positions: Tuple[BondPosition] = ()) -> dict[BondPosition, list['Molecule']]:
        split_bonds = []
        for bond in self._mol.GetBonds():
            # Skip bonds that are in the ignore list
            if BondPosition.from_bond(bond) in ignore_bond_positions:
                continue

            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                if atom1.GetAtomMapNum() == 0 or atom2.GetAtomMapNum() == 0:
                    raise ValueError("Both atoms must have atom map numbers set to split the bond")
                bond_position = BondPosition(atom1.GetAtomMapNum(), atom2.GetAtomMapNum())
                split_bonds.append(bond_position)
        
        fragment_list = defaultdict(list)
        for bond_position in split_bonds:
            fragments = self._split_simple_bond(bond_position)
            for frag in fragments:
                frag_mol = Molecule(frag)
                fragment_list[(bond_position,)].append(frag_mol)

        return dict(fragment_list)
    
    def fragment_ring_single_bonds(self, ignore_bond_positions: Tuple[BondPosition] = ()) -> dict[BondPosition, list['Molecule']]:
        split_bonds = []
        for bond in self._mol.GetBonds():
            # Skip bonds that are in the ignore list
            if BondPosition.from_bond(bond) in ignore_bond_positions:
                continue

            if (
                bond.GetBondType() == Chem.BondType.SINGLE 
                and bond.IsInRing()
                ):
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                if atom1.GetAtomMapNum() == 0 or atom2.GetAtomMapNum() == 0:
                    raise ValueError("Both atoms must have atom map numbers set to split the bond")
                bond_position = BondPosition(atom1.GetAtomMapNum(), atom2.GetAtomMapNum())
                split_bonds.append(bond_position)
        
        fragment_list = defaultdict(list)
        for bond_position in split_bonds:
            fragments = self._split_simple_bond(bond_position)
            for frag in fragments:
                frag_mol = Molecule(frag)
                fragment_list[(bond_position,)].append(frag_mol)

        return dict(fragment_list)
    
    def get_aromatic_ring_bonds(self) -> tuple[BondPosition]:
        """
        Get aromatic ring bonds from a molecule.
        """
        ring_info = self._mol.GetRingInfo()
        mol = self._mol
        aromatic_bonds = set()

        for ring_atoms in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms):
                atoms = [mol.GetAtomWithIdx(idx) for idx in ring_atoms]
                for atom in atoms:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() in ring_atoms:
                            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                            aromatic_bonds.add(BondPosition.from_bond(bond))
        return tuple(aromatic_bonds)
    
    @staticmethod
    def get_resonance_bonds(molecule: 'Molecule') -> tuple[BondPosition]:
        """
        Get resonance bonds from a molecule.
        """
        mol = molecule._mol
        res_supplier = ResonanceMolSupplier(mol)
        resonance_bonds = set()
        resonance_candidates = {}
        for res_mol in res_supplier:
            for bond in res_mol.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    resonance_bonds.add(BondPosition.from_bond(bond))
                else:
                    bond_pos = BondPosition.from_bond(bond)
                    if bond_pos not in resonance_candidates:
                        resonance_candidates[bond_pos] = bond.GetBondType()
                    else:
                        if resonance_candidates[bond_pos] != bond.GetBondType():
                            resonance_bonds.add(bond_pos)

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        resonance_ring_bonds = set()
        for ring in atom_rings:
            ring_bonds = set()
            for i in range(len(ring)):
                begin_atom_idx = ring[i]
                begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
                for neighbor in begin_atom.GetNeighbors():
                    if neighbor.GetIdx() in ring:
                        ring_bonds.add(BondPosition(begin_atom.GetAtomMapNum(), neighbor.GetAtomMapNum()))
            
            if ring_bonds & resonance_bonds:
                resonance_ring_bonds.update(ring_bonds)
            
        return tuple(resonance_ring_bonds)
        


    
