from __future__ import annotations

from typing import Union, List, Dict, Tuple, Literal
from collections import defaultdict
from bidict import bidict
from enum import Enum
import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ...ms.constants import AdductType
from ...ms.Adduct import Adduct
from ..mol.Molecule import Molecule
from ..mol.Fragment import Fragment
from ..mol.Formula import Formula
from .BondPosition import BondPosition
from .FragmentTree import FragmentTree, FragmentNode, FragmentEdge

class FragmentationMode(Enum):
    SINGLE_NON_RING = "single_non_ring"
    SINGLE_RING = "single_ring"

class Fragmenter:
    """
    Fragmenter class to fragment a molecule into smaller parts.
    """
    def __init__(
            self, adduct_type: Tuple[AdductType], 
            max_depth: int,
            fragmentation_modes: Tuple[FragmentationMode, ...] = (FragmentationMode.SINGLE_NON_RING, FragmentationMode.SINGLE_RING)
            ):
        self.adduct_type = adduct_type
        self.max_depth = max_depth
        self.fragmentation_modes = tuple(set(fragmentation_modes))

    def _to_adducts(self, molecule_or_smiles: Union[Molecule, str]) -> Tuple[Adduct]:
        """
        Convert an AdductType to an Adduct object.
        """
        if isinstance(molecule_or_smiles, Molecule):
            molecule = molecule_or_smiles
        elif isinstance(molecule_or_smiles, str):
            molecule = Molecule(molecule_or_smiles)
        else:
            raise TypeError(f"Unsupported type for molecule_or_smiles: {type(molecule_or_smiles)}")
        
        if self.adduct_type == AdductType.M_PLUS_H_POS:
            if molecule.charge == 0:
                return (Adduct.from_str('[M+H]+'),)
            elif molecule.charge == 1:
                return (Adduct.from_str('[M]+'),)
            else:
                raise ValueError(f"Unsupported charge state for adduct type {self.adduct_type} with molecule {molecule.smiles}")
        else:
            raise ValueError(f"Unsupported adduct type: {self.adduct_type}")

    def create_fragment_tree(self, molecule: Molecule, timeout_seconds: float = float('inf')) -> FragmentTree:
        """
        Create a fragment tree from the molecule.

        Parameters:
            molecule (Molecule): The input molecule.
            timeout_seconds (float): The maximum time allowed for processing. If None, no timeout is applied.

        Raises:
            TimeoutError: Raised if the processing time exceeds the specified timeout.

        Returns:
            FragmentTree: The resulting fragment tree.
        """
        start_time = time.time()

        def check_timeout():
            if (time.time() - start_time) > timeout_seconds:
                raise TimeoutError("Fragmentation process timed out.")

        nodes: list[FragmentNode] = []
        edges: list[FragmentEdge] = []
        node_frags: list[bool] = []

        aromatic_bonds = molecule.get_aromatic_ring_bonds()

        precursor_fragment = Fragment(molecule, adduct_type=self.adduct_type)
        node = FragmentNode(precursor_fragment.adducted_ion.molecule.smiles)
        node.adducts = self._to_adducts(molecule)
        nodes.append(node)
        node_frags.append(False)
        next_node_ids = [0]
        origin_atom_id_to_current_id_maps = [bidict({atom.GetAtomMapNum(): atom.GetAtomMapNum() for atom in molecule._mol.GetAtoms()})]

        for depth in range(1, self.max_depth + 2):
            check_timeout()
            # print(f"\n{'-' * 40}")
            # print(f"Fragmenting at depth {depth}...")
            # print(f"Current nodes: {len(nodes)}")
            # print(f"Current edges: {len(edges)}")
            # print(f"Next node IDs: {len(next_node_ids)}")
            if depth > self.max_depth:
                break
            new_node_ids = []
            for current_node_id in next_node_ids:
                check_timeout()
                origin_atom_id_to_current_id_map = origin_atom_id_to_current_id_maps.pop(0)
                if node_frags[current_node_id]:
                    continue
                smiles = nodes[current_node_id].smiles
                molecule_base = Molecule(smiles)
                ignore_bond_positions = tuple(
                    BondPosition(origin_atom_id_to_current_id_map[bond_pos[0]], origin_atom_id_to_current_id_map[bond_pos[1]])
                    for bond_pos in aromatic_bonds if ((bond_pos[0] in origin_atom_id_to_current_id_map) and (bond_pos[1] in origin_atom_id_to_current_id_map))
                )
                fragmented_list = self._fragment_all(molecule_base, ignore_bond_positions=ignore_bond_positions)

                for (bond_pos, fragment_index), fragment_info in fragmented_list.items():
                    check_timeout() 
                    molecule_sub = fragment_info['molecule']
                    atom_map_dict_sub = bidict(fragment_info['atom_map_dict'])
                    smiles_sub = molecule_sub.smiles
                    attr = fragment_info['attr']
                    for id, node in enumerate(nodes):
                        if node.smiles == smiles_sub:
                            edge = FragmentEdge(
                                source_id=current_node_id,
                                target_id=id,
                                bond_positions=bond_pos,
                                fragment_index=fragment_index,
                                attribute=attr
                                )
                            edges.append(edge)
                            break
                    else:
                        # If no existing node found, create a new one
                        new_node = FragmentNode(smiles_sub)
                        new_node.adducts = self._to_adducts(smiles_sub)
                        edge = FragmentEdge(
                            source_id=current_node_id,
                            target_id=len(nodes),
                            bond_positions=bond_pos,
                            fragment_index=fragment_index,
                            attribute=attr
                        )
                        nodes.append(new_node)
                        edges.append(edge)
                        node_frags.append(False)
                        new_node_ids.append(len(nodes) - 1)
                        new_atommap_to_idx = molecule_sub.atom_map_to_idx
                        origin_atom_id_to_current_id_maps.append(
                            bidict({origin_atom_id_to_current_id_map.inv[atom_map_dict_sub.inv[atom.GetAtomMapNum()]]: new_atommap_to_idx[atom.GetAtomMapNum()]+1 for atom in molecule_sub._mol.GetAtoms()})
                        )

                node_frags[current_node_id] = True

            next_node_ids = list(new_node_ids)
        
        # Create the FragmentTree
        fragment_tree = FragmentTree(
            molecule=molecule,
            nodes=nodes,
            edges=edges,
        )
        return fragment_tree
                
    def _fragment_all(self, molecule: Molecule, ignore_bond_positions: Tuple[BondPosition] = ()) -> Dict[Tuple[BondPosition, int], Dict[str]]:
        fragments_by_bond: Dict[Tuple[BondPosition, int], Dict[str]] = defaultdict(lambda: {"attr": {'fragment_type':[]}, "fragment": None})

        if FragmentationMode.SINGLE_NON_RING in self.fragmentation_modes:
            for bond_pos, frag_molecules in molecule.fragment_non_ring_single_bonds(ignore_bond_positions).items():
                for i, frag_mol in enumerate(frag_molecules):
                    key = (bond_pos, i)
                    if fragments_by_bond[key]['fragment'] is not None:
                        raise ValueError(f"Duplicate bond position {bond_pos} found in non-ring single bond fragmentation.")
                    fragments_by_bond[key]['fragment'] = frag_mol
                    fragments_by_bond[key]['attr']['fragment_type'].append('SINGLE_NON_RING')

        if FragmentationMode.SINGLE_RING in self.fragmentation_modes:
            for bond_pos, frag_molecules in molecule.fragment_ring_single_bonds(ignore_bond_positions).items():
                for i, frag_mol in enumerate(frag_molecules):
                    key = (bond_pos, i)
                    if fragments_by_bond[key]['fragment'] is not None:
                        raise ValueError(f"Duplicate bond position {bond_pos} found in ring single bond fragmentation.")
                    fragments_by_bond[key]['fragment'] = frag_mol
                    fragments_by_bond[key]['attr']['fragment_type'].append('SINGLE_RING')

        fragments_by_bond = dict(fragments_by_bond)

        fragment_list = {}
        for key, fragment_info in fragments_by_bond.items():
            fragment_type_attr = tuple(set(fragment_info['attr']['fragment_type']))
            frag = Fragment(fragment_info['fragment'], self.adduct_type)
            fragment_list[key] = {
                'molecule': frag.adducted_ion.molecule,
                'atom_map_dict': frag.adducted_ion.atom_map_dict,
                'attr': {'fragment_type': fragment_type_attr, 'adduct': frag.adduct},}
            
        return fragment_list

    def copy(self) -> Fragmenter:
        """
        Create a copy of the current Fragmenter instance.

        Returns:
            Fragmenter: A new instance with the same adduct_types and max_depth.
        """
        return Fragmenter(
            adduct_type=self.adduct_type,
            max_depth=self.max_depth,
            fragmentation_modes=self.fragmentation_modes
        )
    

    
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

    
