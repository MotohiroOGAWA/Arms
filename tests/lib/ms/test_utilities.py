import unittest
from lib.ms.utilities import *

class TestMspReader(unittest.TestCase):
    def test_calc_coverage(self):
        """
        Test the calc_coverage function with a simple example.
        """
        src_ms = np.array([100.0, 200.0, 300.0])
        tgt_ms = np.array([[100.0, 1.0], [200.0, 2.0], [400.0, 3.0]])
        
        # Expected coverage with weights
        expected_coverage = 3.0 / 6.0
        coverage = calc_coverage(src_ms, tgt_ms, wt_intensity=True)
        
        self.assertAlmostEqual(coverage, expected_coverage, places=2, msg="Coverage calculation is incorrect.")
        
        # Expected coverage without weights
        expected_coverage = 2.0 / 3.0
        coverage = calc_coverage(src_ms, tgt_ms[:, 0], wt_intensity=False)
        
        self.assertAlmostEqual(coverage, expected_coverage, places=2, msg="Coverage calculation is incorrect.")

if __name__ == "__main__":
    # cd ~/workspace/mnt/app ; python -m unittest tests.lib.ms.test_utilities
    unittest.main()
