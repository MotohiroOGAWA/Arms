import unittest
from lib.io.msp_reader import read_msp_file
from lib.ms.MassSpectrum import MassSpectrum

class TestMspReader(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test method.
        It loads the MSP file and prepares the MassSpectrum object.
        """
        self.msp_file = './tests/data/msp/MSMS_Public_EXP_NEG_VS17_test.msp'
        self.cols = read_msp_file(self.msp_file)

    # def test_read_msp_file(self):
    #     """
    #     Test whether the MSP file is read correctly and contains at least one spectrum.
    #     """
    #     self.assertGreater(len(self.mass_spectra), 0, "No spectra loaded from the MSP file.")

    def test_extract_multiple_peaks(self):
        """
        Test whether peaks can be correctly extracted from multiple indices.
        """
        mass_spectra = MassSpectrum(self.cols)
        peaks = mass_spectra[6:8]

        mass_spectrum1 = mass_spectra['Name']
        mass_spectrum2 = mass_spectra[['Name', 'Formula']]

        # Check if exactly two peak arrays are returned
        self.assertEqual(len(peaks), 2, "Expected 2 sets of peaks to be extracted.")

if __name__ == "__main__":
    # cd ~/workspace/mnt/app ; python -m unittest tests.lib.io.test_msp_reader
    unittest.main()
