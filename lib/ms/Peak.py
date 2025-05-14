from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from collections.abc import Sequence


class Peak:
    """
    A class to represent a collection of mass spectral peaks.

    Each peak is a pair of [m/z, intensity], and the full data is a 2D numpy array
    of shape (n_peaks, 2).
    """

    def __init__(self, data: Dict, normalize: bool = True):
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
        peaks = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
        self._peak = PeakSeries(peaks)
        self._data = data
        if normalize:
            self.normalize_intensity()

    def __len__(self) -> int:
        return len(self._peak)
    
    def __str__(self) -> str:
        return self._peak.format_peak()

    def __repr__(self) -> str:
        return str(self)
    
    def __getitem__(self, i: int | slice | List[int] | str | List[str]) -> PeakEntry | PeakSeries:
        """
        Return a PeakEntry at the given index.
        """
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for Peak with {len(self)} peaks."
            return self._peak[i]
        elif isinstance(i, str):
            if i in self._data:
                return self._data[i]
            else:
                raise KeyError(f"Key '{i}' not found in Peak data.")
        elif isinstance(i, slice):
            return PeakSeries(self._peak[i])
        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                return PeakSeries(self._peak[i])
            elif all(idx in self._data for idx in i):
                return (self._data[k] for k in i)
            else:
                raise IndexError(f"Indices {i} out of range for Peak with {len(self)} peaks.")
        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of int.")
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        """
        Iterate over all peaks as PeakEntry instances.
        """
        for p in self._peak:
            yield p


    def normalize_intensity(self, to: float = 1.0) -> None:
        """
        Normalizes intensity values so that the maximum becomes the given value.

        Parameters:
            to (float): The value to scale the maximum intensity to.
        """
        self._peak.normalize_intensity(to)

    @property
    def is_int_mz(self) -> bool:
        """
        Check if all m/z values are integers.

        Returns:
            bool: True if all m/z values are integers, False otherwise.
        """
        all_integers = np.all(self._peak._data[:, 0] % 1 == 0)
        return all_integers

class PeakSeries:
    """
    Represents a series of mass spectral peaks.
    """

    def __init__(self, data: np.ndarray):
        assert isinstance(data, np.ndarray) or isinstance(data, PeakSeries), "PeakSeries data must be a numpy array or PeakSeries"
        if isinstance(data, PeakSeries):
            data = data._data
        assert data.ndim == 2 and data.shape[1] == 2, "data must be a 2D array with shape (n_peaks, 2)"
        self._data = data

    def __len__(self) -> int:
        return self._data.shape[0]
    
    def __repr__(self):
        return f"PeakSeries(n_peaks={len(self)})"
    
    def __str__(self):
        return self.format_peak()
    
    def __getitem__(self, i: int | slice | Sequence) -> PeakEntry | PeakSeries:
        """
        Return a single Peak object (for int index) or a new PeakSeries object (for slice or list of indices).
        """
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for PeakSeries with {len(self)} peaks."
            mz, intensity = self._data[i]
            return PeakEntry(mz, intensity)
        elif isinstance(i, slice):
            return PeakSeries(self._data[i])
        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                return PeakSeries(self._data[i])
            else:
                raise IndexError(f"Indices {i} out of range for PeakSeries with {len(self)} peaks.")
        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of int.")
        
    def __iter__(self):
        """
        Iterate over all peaks as tuples of (m/z, intensity).
        """
        for mz, intensity in self._data:
            yield mz, intensity

    
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
        lines = [format_str.format(mz, intensity) for mz, intensity in self._data]
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
        
        max_intensity = np.max(self._data[:, 1])
        if max_intensity > 0:
            self._data[:, 1] = self._data[:, 1] / max_intensity * to
    

class PeakEntry:
    """
    Represents a single mass spectral peak with m/z and intensity.
    """

    def __init__(self, mz: float, int: float):
        self.mz = mz
        self.intensity = int

    def __repr__(self):
        return f"PeakEntry(mz={self.mz}, intensity={self.intensity})"
    
    def __str__(self):
        return f"m/z: {self.mz}, Intensity: {self.intensity}"
    
    def __iter__(self):
        """
        Iterate over the m/z and intensity values.
        """
        yield self.mz
        yield self.intensity
