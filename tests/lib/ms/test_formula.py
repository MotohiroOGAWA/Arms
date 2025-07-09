import unittest
from lib.chem.mol.Formula import Formula


class TestFormula(unittest.TestCase):
    def test_basic_parsing(self):
        f = Formula("C6H12O6")
        self.assertEqual(f.elements["C"], 6)
        self.assertEqual(f.elements["H"], 12)
        self.assertEqual(f.elements["O"], 6)
        self.assertEqual(f.charge, 0)
        self.assertEqual(f.to_string(), "C6H12O6")

    def test_charge_parsing(self):
        f = Formula("H2O+")
        self.assertEqual(f.charge, 1)
        f = Formula("NaCl-")
        self.assertEqual(f.charge, -1)
        f = Formula("Fe+3")
        self.assertEqual(f.charge, 3)
        f = Formula("Fe-2")
        self.assertEqual(f.charge, -2)

    def test_mass_calculation(self):
        f = Formula("H2O")
        self.assertAlmostEqual(f.exact_mass, 18.0106, places=2)

    def test_addition(self):
        f1 = Formula("H2")
        f2 = Formula("O")
        f3 = f1 + f2
        self.assertEqual(f3.to_string(), "H2O")

    def test_subtraction(self):
        f1 = Formula("H2O")
        f2 = Formula("H")
        f3 = f1 - f2
        self.assertEqual(f3.to_string(), "HO")

    def test_equality(self):
        f1 = Formula("C6H12O6")
        f2 = Formula("C6H12O6")
        self.assertEqual(f1, f2)

    def test_to_string_no_charge(self):
        f = Formula("H2O+")
        self.assertEqual(f.to_string(no_charge=True), "H2O")
        self.assertEqual(f.plain, "H2O")

    def test_diff(self):
        f1 = Formula("H2O")
        f2 = Formula("HO")
        diff, charge_diff = f1.diff(f2)
        self.assertEqual(diff, {"H": 1})
        self.assertEqual(charge_diff, 0)


if __name__ == "__main__":
    # cd ~/workspace/mnt/app ; python -m unittest tests. ~
    # tests.lib.ms.test_formula
    unittest.main()
