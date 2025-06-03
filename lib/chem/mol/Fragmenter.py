from __future__ import annotations

from typing import Union, List, Dict, Tuple, Literal
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ...ms.constants import AdductType
from .Molecule import Molecule
from .Fragment import Fragment
from .Formula import Formula

class Fragmenter:
    """
    Fragmenter class to fragment a molecule into smaller parts.
    """
    def __init__(self, adduct_types: Tuple[AdductType], max_depth: int):
        self.adduct_types = adduct_types
        self.max_depth = max_depth

    def create_fragment_tree(self, molecule: Molecule) -> FragmentTree:
        """
        Create a fragment tree from the molecule.
        """
        fragment_tree = FragmentTree(molecule)

        for adduct_type in self.adduct_types:
            fragment = Fragment(molecule, adduct_type=adduct_type)
            node = self._create_fragment_node(fragment, depth=1)
            fragment_tree.root.add_child(bond_pos=BondPosition(), adduct_type=AdductType.NONE, child=node)

        return fragment_tree

    def _create_fragment_node(self, fragment: Fragment, depth: int) -> FragmentNode:
        """
        Create a fragment node for the tree.
        """
        node = FragmentNode(fragment, depth=depth)
        if depth < self.max_depth:
            fragmented_list = self._fragment_all(fragment.stable_mol)

            for (bond_pos, adduct_type), fragments in fragmented_list.items():
                for f in fragments:
                    child = self._create_fragment_node(f, depth=depth + 1)
                    node.add_child(bond_pos=bond_pos, adduct_type=adduct_type, child=child)

        return node

    def _fragment_all(self, molecule: Molecule) -> Dict[(BondPosition, AdductType), List[Fragment]]:
        
        fragments_by_bond_pos_and_adduct = defaultdict(list)
        for (bond_pos, adduct_type), fragments in self._fragment_non_ring_single_bonds(molecule).items():
            fragments_by_bond_pos_and_adduct[(bond_pos, adduct_type)].extend(fragments)

        return dict(fragments_by_bond_pos_and_adduct)

    def _fragment_non_ring_single_bonds(self, molecule: Molecule) -> Dict[(BondPosition, AdductType), List[Fragment]]:
        """
        Fragment the molecule at all splitable bonds.
        """
        split_bonds = self._get_non_ring_single_bonds(molecule)

        fragment_list = defaultdict(list)
        for bond in split_bonds:
            # Split the bond and create a new fragment
            fragmented_by_adducts = self._split_simple_bond(molecule, bond)
            for adduct_type, fragmented in fragmented_by_adducts.items():
                fragment_list[(BondPosition(bond.GetIdx()), adduct_type)].extend(fragmented)

        return dict(fragment_list)
    
    def _get_non_ring_single_bonds(self, molecule: Molecule) -> Tuple[Chem.Bond]:
        """
        Get all non-ring single (σ) bonds that are candidates for cleavage.

        Returns:
            List of tuples containing a single Chem.Bond object.
        """
        split_bonds = []
        for bond in molecule.mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                split_bonds.append(bond)
        return tuple(split_bonds)

        
    def _split_simple_bond(self, molecule: Molecule, bond: Union[Chem.Bond, int]) -> Dict[List[Fragment]]:
        rw_mol = Chem.RWMol(molecule.mol)

        if isinstance(bond, int):
            bond = rw_mol.GetBondWithIdx(bond)
        elif not isinstance(bond, Chem.Bond):
            raise TypeError(f"Expected bond to be of type Chem.Bond or int, got {type(bond)}")
        
        bond_indices = [bond.GetIdx()]

        fragmented = Chem.FragmentOnBonds(rw_mol, bond_indices, addDummies=True)
        frags: tuple[Chem.Mol] = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)

        fragment_list_by_adduct = {}
        for adduct_type in self.adduct_types:
            fragment_list_by_adduct[adduct_type] = []
            fragment_list = []
            for f in frags:
                # Create a new Fragment object for each fragment
                _f = Fragment(Molecule(f), adduct_type=adduct_type)
                fragment_list.append(_f)
            fragment_list_by_adduct[adduct_type].extend(fragment_list)

        return fragment_list_by_adduct

    def copy(self) -> Fragmenter:
        """
        Create a copy of the current Fragmenter instance.

        Returns:
            Fragmenter: A new instance with the same adduct_types and max_depth.
        """
        return Fragmenter(adduct_types=self.adduct_types, max_depth=self.max_depth)
    
class FragmentTree:
    """
    FragmentTree class to represent a tree of fragments.
    """
    def __init__(self, molecule: Molecule):
        self.molecule = molecule
        
        self.fragment = Fragment(molecule, adduct_type=AdductType.NONE)
        self.root = FragmentNode(self.fragment, depth=0)
    
    def get_all_formulas(self) -> List[Formula]:
        """
        Get all formulas from the fragment tree.
        """
        def get_formulas(node: FragmentNode) -> List[Formula]:
            formulas = []
            for child in node.children.values():
                for fragment_node in child:
                    formulas.append(fragment_node.fragment.formula)
                    formulas.extend(get_formulas(fragment_node))
            return formulas
        
        formulas = get_formulas(self.root)
        formulas = list(set(formulas))
        formulas.sort(key=lambda x: x.exact_mass)

        return formulas



class FragmentNode:
    """
    FragmentNode class to represent a node in the fragment tree.
    """
    def __init__(self, fragment: Fragment, depth: int):
        self.fragment = fragment
        self.depth = depth
        self.children: Dict[BondPosition, List[FragmentNode]] = {}

    def add_child(self, bond_pos: BondPosition, adduct_type: AdductType, child: FragmentNode):
        if (bond_pos, adduct_type) not in self.children:
            self.children[(bond_pos, adduct_type)] = []
        self.children[(bond_pos, adduct_type)].append(child)


    def tree_str(self, level: int = 0, display_fields: tuple[Literal['SMILES', 'mz', 'formula'], ...] = ('SMILES', 'mz', 'formula')) -> str:
        """
        Get the string representation of the fragment tree.
        """
        return self._tree_str(self, level=level, display_fields=display_fields)

    @staticmethod
    def _tree_str(node:FragmentNode, level: int = 0, display_fields: tuple[Literal['SMILES', 'mz', 'formula'], ...] = ('SMILES', 'mz', 'formula')) -> str:
        indent = ' ' * (level * 3)
        entry_prefix = f"{indent}|++"
        output = ""

        for field in display_fields:
            if field == 'SMILES':
                output += f"{entry_prefix}SMILES  : {node.fragment.raw_mol.smiles} / {node.fragment.adduct}\n"
                output += f"{entry_prefix}SMILES  : {node.fragment.stable_mol.smiles}\n"
            elif field == 'mz':
                mz = node.fragment.mz
                if mz is None:
                    output += f"{entry_prefix}EM      : {node.fragment.stable_mol.exact_mass:.4f}\n"
                else:
                    output += f"{entry_prefix}m/z     : {node.fragment.mz:.4f}\n"
            elif field == 'formula':
                output += f"{entry_prefix}Formula : {node.fragment.formula}\n"
            else:
                raise ValueError(f"Unknown display field: '{field}'")

        for (bond_pos, adduct_type), children in node.children.items():
            for child in children:
                output += '\n'
                output += f"{indent}|- Bond: {bond_pos}, Adduct: {adduct_type.value}\n"
                output += FragmentNode._tree_str(child, level=level + 1, display_fields=display_fields)

        return output
        


class BondPosition(tuple):
    """
    BondPosition is a tuple subclass to represent bond indices in a molecule,
    stored in a sorted, unique order to ensure consistent comparison and hashing.
    """

    def __new__(cls, *bond_id: int, tag: str = ''):
        return super().__new__(cls, tuple(sorted(bond_id)))

    def __repr__(self):
        return f"BondPosition{tuple(self)}"

    def __eq__(self, other):
        if isinstance(other, BondPosition):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == tuple(sorted(other))
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self))
    
if __name__ == "__main__":
    # molecule = Molecule("OC1CCC(OC2CCC(O)OC2)OC1") 
    # molecule = Molecule("ClC1=NC=CC(=C1)NC(O)=NC=2C=CC=CC2") 
    # molecule = Molecule("O=C1OC(C(=O)OC)(C(C=2C=CC(O)=CC2)=C1OC)CC3=CC=C(O)C(=C3)CC=C(C)C") 
    molecule = Molecule("OCC(=CCNC=1N=CNC2=NC=NC21)C") 
    print(molecule)

    # fragmenter = Fragmenter()
    # fragmenter.split_simple_bond(molecule.mol, 4)
    fragmenter = Fragmenter((AdductType.M_PLUS_H_POS,), max_depth=2)

    fragment_tree = fragmenter.create_fragment_tree(molecule)
    
    all_formulas = fragment_tree.get_all_formulas()

    for formula in all_formulas:
        print(formula)
        print(formula.exact_mass)
        print()

    with open('tree.txt', 'w') as f:
        f.write(fragment_tree.root.tree_str())
    

    # smiles = Chem.MolFromSmiles('[NH3+][CH2]CO[NH2+][CH2]CO')
    # formula = rdMolDescriptors.CalcMolFormula(smiles)

    

