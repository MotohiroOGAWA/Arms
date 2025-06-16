import unittest
from lib.chem.mol.Fragmenter import Fragmenter
from lib.chem.mol.Molecule import Molecule
from lib.ms.constants import AdductType


class TestFragmenter(unittest.TestCase):
    def setUp(self):
        self.adduct_types = [AdductType.M_PLUS_H_POS]

    def test1(self):
        molecule = Molecule('CC(NC(=O)CC1=CNC2=C1C=CC=C2)C(O)=O')
        fragmenter = Fragmenter(self.adduct_types, max_depth=5)
        fragment_tree = fragmenter.create_fragment_tree(molecule)
        pass


if __name__ == "__main__":
    unittest.main()
