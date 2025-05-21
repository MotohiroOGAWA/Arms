import unittest
from lib.ms.MassSpectrum import MassSpectrum
from lib.ms.Peak import Peak, PeakSeries, PeakEntry, Formula

class TestTemplate(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test method.
        """
        data = {
            'Name': ['Test1', 'Test2', 'Test3'],
            'Value': [1, 2, 3],
            'Description': ['First test', 'Second test', 'Third test'],
            'Peak': ['17.02655,100;180.06339,200;3.45,300', '11.12,110;21.34,210;31.45,310', '12.12,120;22.34,220;32.45,320'],
        }
        self.mass_spectrum = MassSpectrum(data)

    def testMassSpectrum(self):
        self.mass_spectrum[0]
        self.mass_spectrum[[0, 1]]
        self.mass_spectrum[0:2]
        self.mass_spectrum['Name']
        self.mass_spectrum[['Name', 'Value']]

    def testMassSpectrumSeries(self):
        mass_spectrum_series = self.mass_spectrum['Name']
        mass_spectrum_series[0]
        mass_spectrum_series[[0, 1]]
        mass_spectrum_series[0:2]
        mass_spectrum_series.value_counts()
    
    def testPeak(self):
        peak: Peak = self.mass_spectrum[0]
        peak[0]
        peak[[0, 1]]
        peak[0:2]
        peak['Name']
        peak[['Name', 'Value']]
        peak.is_int_mz
        str(peak)
        peak.normalize_intensity()

    def testPeakSeries(self):
        peak_series: PeakSeries = self.mass_spectrum[0][:]
        peak_series[0]
        peak_series[[0, 1]]
        peak_series[0:2]
        peak_series.format_peak()

    def testPeakEntry(self):
        peak_entry: PeakEntry = self.mass_spectrum[0][0]
        peak_entry.mz
        peak_entry.intensity
        str(peak_entry)

    def testAssignFormula(self):
        formula1 = Formula("C6H12O6")
        formula2 = Formula("NH3")

        # peaks = self.mass_spectrum[0].peaks
        # peaks.assign_formula([formula1, formula2])
        self.mass_spectrum[0].peaks.assign_formula([formula1, formula2])
        assigned_formulas = self.mass_spectrum[0].peaks.fragment_formulas
        assert any(formula1 == f for f in assigned_formulas), "Formula assignment failed"
        assert any(formula2 == f for f in assigned_formulas), "Formula assignment failed"

if __name__ == "__main__":
    # cd ~/workspace/mnt/app ; python -m unittest tests. ~
    # unittest.main()
    pass