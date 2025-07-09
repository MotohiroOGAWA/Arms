import unittest
from lib.chem.mol.Molecule import Molecule
from lib.chem.tree.Fragmenter import Fragmenter
from lib.chem.tree.FragmentTree import FragmentTree
from lib.ms.constants import AdductType


class TestFragmenter(unittest.TestCase):
    # [node.smiles for node in fragment_tree.nodes]
    def setUp(self):
        self.adduct_type = AdductType.M_PLUS_H_POS

    def test1(self):
        molecule = Molecule('Cc1ccccc1')
        fragmenter = Fragmenter(self.adduct_type, max_depth=10)
        fragment_tree = fragmenter.create_fragment_tree(molecule)
        pass

    def test2(self):
        molecule = Molecule('CC(NC(=O)CC1=CNC2=C1C=CC=C2)C(O)=O')
        fragmenter = Fragmenter(self.adduct_type, max_depth=10)
        fragment_tree = fragmenter.create_fragment_tree(molecule)
        pass

    def test3(self):
        molecule = Molecule('CC1=CC(O)=CC(O)=C1C')
        fragmenter = Fragmenter(self.adduct_type, max_depth=10)
        fragment_tree = fragmenter.create_fragment_tree(molecule)
        # fragment_tree.save('/home/user/workspace/mnt/app/data/mol/experiments_tree/test_fragment_tree.dill')
        pass

    def test31(self):
        smiles = 'CC1=C(O)C=C(O)C=[C+]1'
        molecule = Molecule(smiles)
        from rdkit import Chem
        bonds = Molecule.get_resonance_bonds(molecule)
        print(bonds)
        

    def test4(self):
        molecule = Molecule('O=C1OC(C)CCCC(O)CCCC=CC=2C=C(O)C=C(O)C12')
        fragmenter = Fragmenter(self.adduct_type, max_depth=10)
        fragment_tree = fragmenter.create_fragment_tree(molecule)
        fragment_tree.save('/home/user/workspace/mnt/app/data/mol/experiments_tree/test_fragment_tree.dill')
        pass


if __name__ == "__main__":
    fragment_tree = FragmentTree.load('/home/user/workspace/mnt/app/data/mol/experiments_tree/test_fragment_tree.dill')
    fragment_tree.get_all_formulas(sources=True)
    unittest.main()
