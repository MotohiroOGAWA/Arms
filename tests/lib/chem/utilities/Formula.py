import unittest
from lib.chem.mol.Formula import Formula


class TestFormula(unittest.TestCase):
    def test_parse_formula(self):
        f = Formula("C6H12O6")
        self.assertEqual(f.elements, {"C": 6, "H": 12, "O": 6})
        self.assertEqual(f.charge, 0)

    def test_charge_parsing(self):
        cases = [
            ("C6H5O7-3", {"C": 6, "H": 5, "O": 7}, -3),
            ("C6H5O7-", {"C": 6, "H": 5, "O": 7}, -1),
            ("C6H5O7+3", {"C": 6, "H": 5, "O": 7}, 3),
            ("C6H5O7+", {"C": 6, "H": 5, "O": 7}, 1),
        ]
        for formula_str, expected_elements, expected_charge in cases:
            with self.subTest(formula=formula_str):
                f = Formula(formula_str)
                self.assertEqual(f.elements, expected_elements)
                self.assertEqual(f.charge, expected_charge)

    def test_exact_mass(self):
        f = Formula("H2O")
        expected = 1.007825 * 2 + 15.994915  # Monoisotopic mass
        self.assertAlmostEqual(f.exact_mass, expected, places=4)

    def test_addition(self):
        f1 = Formula("H2O")
        f2 = Formula("Na+")
        f3 = f1 + f2
        self.assertEqual(str(f3), "H2NaO+")
        self.assertEqual(f3.charge, 1)

    def test_subtraction(self):
        f1 = Formula("C6H12O6")
        f2 = Formula("H2O")
        f3 = f1 - f2
        self.assertEqual(str(f3), "C6H10O5")

    def test_diff(self):
        f1 = Formula("C2H4O2")
        f2 = Formula("CH2O")
        diff, charge = f1.diff(f2)
        self.assertEqual(diff, {"C": 1, "H": 2, "O": 1})
        self.assertEqual(charge, 0)

        f3 = Formula("C2H4O2+")
        f4 = Formula("CH2O")
        diff, charge = f3.diff(f4)
        self.assertEqual(diff, {"C": 1, "H": 2, "O": 1})
        self.assertEqual(charge, 1)

        f5 = Formula("C2H4O2-2")
        f6 = Formula("CH2O-1")
        diff, charge = f5.diff(f6)
        self.assertEqual(diff, {"C": 1, "H": 2, "O": 1})
        self.assertEqual(charge, -1)

    def test_str_and_repr(self):
        f = Formula("C2H5OH")
        self.assertEqual(str(f), "C2H6O")
        self.assertIn("Formula(C2H6O)", repr(f))

    def test_from_mol(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("OC1CCC(OC2CCC(O)OC2)OC1")
        f = Formula.from_mol(mol)
        self.assertEqual(f.elements, {"C": 10, "H": 18, "O": 5})
        self.assertEqual(f.charge, 0)

    def test_enumerate_possible_sub_formulas(self):
        f = Formula("C3H8O")
        sub_formulas = f.get_possible_sub_formulas()
        self.assertIsInstance(sub_formulas, list)
        self.assertGreater(len(sub_formulas), 0)


if __name__ == "__main__":
    unittest.main()
