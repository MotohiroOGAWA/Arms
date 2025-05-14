import unittest
from lib.ms.MassSpectrum import MassSpectrum
from lib.ms.Peak import Peak, PeakSeries, PeakEntry

class TestTemplate(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test method.
        """
        data = {
            'Name': ['Test1', 'Test2', 'Test3'],
            'Value': [1, 2, 3],
            'Description': ['First test', 'Second test', 'Third test'],
            'Peak': ['10.12,100;20.34,200;3.45,300', '11.12,110;21.34,210;31.45,310', '12.12,120;22.34,220;32.45,320'],
        }
        self.mass_spectrum = MassSpectrum(data)

    def testMassSpectrum(self):
        self.mass_spectrum[0]
        self.mass_spectrum[[0, 1]]
        self.mass_spectrum[0:2]
        self.mass_spectrum['Name']
        self.mass_spectrum[['Name', 'Value']]
    
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

if __name__ == "__main__":
    # cd ~/workspace/mnt/app ; python -m unittest tests. ~
    # unittest.main()
    pass
