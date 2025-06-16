import unittest
from lib.chem.mol.Fragment import Fragment
from lib.chem.mol.Molecule import Molecule
from lib.ms.constants import AdductType


class TestReconstruct(unittest.TestCase):
    def setUp(self):
        self.adduct_types = [AdductType.M_PLUS_H_POS]

    def tearDown(self):
        """
        This method is called after each test method.
        """
        for fragment in self.last_fragments:
            self.assertIsNotNone(fragment)
            self.assertIsInstance(fragment, Fragment)
        


    def test1(self):
        molecule = Molecule('CC(NC(=O)CC1=CNC2=C1C=CC=C2)C(O)=O')
        self.last_fragments = []
        for adduct_type in self.adduct_types:
            fragment = Fragment(molecule, adduct_type)
            self.last_fragments.append(fragment)

    def test2(self):
        molecule = Molecule('[*]CC1=CNC2=C1C=CC=C2')
        self.last_fragments = []
        for adduct_type in self.adduct_types:
            fragment = Fragment(molecule, adduct_type)
            self.last_fragments.append(fragment)


if __name__ == "__main__":
    unittest.main()
