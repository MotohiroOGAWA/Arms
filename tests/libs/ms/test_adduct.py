import unittest
from libs.ms.Adduct import Adduct
from libs.chem.mol.Formula import Formula


class TestAdduct(unittest.TestCase):
    def test_standard_adducts(self):
        """Test normal adduct strings with charge and mass shift."""
        test_cases = [
            ("[M]",            0, 0.0),
            ("[M]+",           +1, 0.0),
            ("[M]-",           -1, 0.0),
            ("[M+H]+",         +1, Formula("H").exact_mass),
            ("[M-H]-",         -1, -Formula("H").exact_mass),
            ("[M+HCOOH-H]-",   -1, Formula("HCOOH").exact_mass - Formula("H").exact_mass),
            ("[M-H2O+H]+",     +1, -Formula("H2O").exact_mass + Formula("H").exact_mass),
            ("[M+NH4]+",       +1, Formula("NH4").exact_mass),
            ("[M-C6H10H5+H]+", +1, -Formula("C6H10H5").exact_mass + Formula("H").exact_mass),
            ("[M+2Na-2H]+",    +1, Formula("Na").exact_mass * 2 - Formula("H").exact_mass * 2),
            ("[M+NaCl]+",      +1, Formula("NaCl").exact_mass),
        ]

        for adduct_str, expected_charge, expected_mass in test_cases:
            with self.subTest(adduct_str=adduct_str):
                adduct = Adduct.from_str(adduct_str)
                # Check that round-trip from string and back matches
                reconstructed = Adduct.from_str(str(adduct))
                self.assertEqual(adduct, reconstructed)
                # Check charge
                self.assertEqual(adduct.charge, expected_charge)
                # Check mass shift
                self.assertAlmostEqual(adduct.mass_shift, expected_mass, places=4)

    def test_string_normalization(self):
        """Test that multiple H+H is normalized to 2H in string output."""
        adduct = Adduct.from_str("[M+H+H]+")
        self.assertEqual(str(adduct), "[M+2H]+")  # normalized string
        self.assertEqual(adduct.charge, 1)
        self.assertAlmostEqual(adduct.mass_shift, Formula("H").exact_mass * 2, places=4)

    def test_invalid_adduct_strings(self):
        # List of malformed adduct strings that should raise exceptions
        invalid_inputs = [
            "M+H]",       # Missing opening bracket
            "[M+H",       # Missing closing bracket
            "[M+?]+",     # Invalid formula
            # "[M+]+",      # No formula after '+'
            # "[M+2]+",     # Number with no formula
        ]

        for adduct_str in invalid_inputs:
            with self.subTest(adduct_str=adduct_str):
                with self.assertRaises(Exception):
                    Adduct.from_str(adduct_str)



if __name__ == "__main__":
    # cd ~/workspace/mnt/app ; python -m unittest tests. ~
    # tests.lib.ms.test_adduct
    unittest.main()
    pass