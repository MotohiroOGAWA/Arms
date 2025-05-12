from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class Peak:
    """
    A class to represent a collection of mass spectral peaks.

    Each peak is a pair of [m/z, intensity], and the full data is a 2D numpy array
    of shape (n_peaks, 2).
    """

    def __init__(self, data: Dict, normalize: bool = False):
        """
        Initialize the Peak object with peak data.

        Parameters:
            data (np.ndarray): 2D array with shape (n_peaks, 2) representing [m/z, intensity] pairs.
            normalize (bool): If True, normalize the intensity values to a maximum of 1.0.

        Raises:
            AssertionError: If data is not a 2D array or does not have shape (n, 2).
        """
        assert isinstance(data, dict), "data must be a dictionary"
        assert "Peak" in data, "data must contain a 'Peak' key"
        peak_str = data['Peak']
        self._peak = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
        self._data = self._peak
        if normalize:
            self.normalize_intensity()

    def __len__(self) -> int:
        return self._peak.shape[0]
    
    def __str__(self) -> str:
        return self.format_peak()

    def __repr__(self) -> str:
        return str(self)
    
    def __getattr__(self, name):
        """
        Allow dynamic attribute access for numpy array methods.
        """
        if name == 'Peak':
            return self._peak
        return getattr(self._data, name)
    
    def __getitem__(self, index: int) -> PeakEntry:
        """
        Return a PeakEntry at the given index.
        """
        mz, intensity = self._peak[index]
        return PeakEntry(mz, intensity)
    
    def __iter__(self):
        """
        Iterate over all peaks as PeakEntry instances.
        """
        for mz, intensity in self._peak:
            yield PeakEntry(mz, intensity)
    
    def format_peak(self, decimals: int = 4, width: int = 12) -> str:
        """
        Format the peak matrix into a string with aligned columns.

        Args:
            decimals (int): Number of digits after the decimal point.
            width (int): Total width of each field (must be at least decimals + 2).

        Returns:
            str: Formatted string with aligned m/z and intensity columns.
        """
        assert width >= decimals + 2, f"Width must be at least decimals + 2 (got width={width}, decimals={decimals})"
        
        format_str = f"{{:>{width}.{decimals}f}}\t{{:>{width}.{decimals}f}}"
        lines = [format_str.format(mz, intensity) for mz, intensity in self._peak]
        return "\n".join(lines)


    def normalize_intensity(self, to: float = 1.0) -> None:
        """
        Normalizes intensity values so that the maximum becomes the given value.

        Parameters:
            to (float): The value to scale the maximum intensity to.
        """
        assert isinstance(to, (int, float)), "to must be a number"
        assert to > 0, "to must be greater than 0"
        assert len(self) > 0, "No peaks to normalize"
        
        max_intensity = np.max(self.data[:, 1])
        if max_intensity > 0:
            self.data[:, 1] = self.data[:, 1] / max_intensity * to

    @property
    def is_int_mz(self) -> bool:
        """
        Check if all m/z values are integers.

        Returns:
            bool: True if all m/z values are integers, False otherwise.
        """
        all_integers = np.all(self.data[:, 0] % 1 == 0)
        return all_integers


class PeakEntry:
    """
    Represents a single mass spectral peak with m/z and intensity.
    """

    def __init__(self, mz: float, int: float):
        self.mz = mz
        self.intensity = int

    def __repr__(self):
        return f"PeakEntry(mz={self.mz}, intensity={self.intensity})"
