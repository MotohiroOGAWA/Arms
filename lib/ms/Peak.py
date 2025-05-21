from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional
from collections.abc import Sequence
from ..chem.mol.Molecule import Molecule
from ..chem.mol.Formula import Formula
from ..ms.Adduct import Adduct
from .constants import MIN_ABS_TOLERANCE


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
        assert isinstance(data["Peak"], PeakSeries), "data['Peak'] must be a string"
        self._peak = data["Peak"]
        self._data: dict = data
        if normalize:
            self.normalize_intensity()

    def __len__(self) -> int:
        return len(self._peak)
    
    def __str__(self) -> str:
        res = ''
        max_len = max(len(k) for k in self._data.keys())
        for d in self._data:
            if d == "Peak":
                continue
            else:
                res += f"{d:<{max_len+1}}:\t{self._data[d]}\n"
        res += f"{'Peak':<{max_len+1}}: \t{self._peak.to_str()}"
        return res
        

    def __repr__(self) -> str:
        contents = [f'\t{line}' for line in str(self).splitlines()]
        content = "\n".join(contents)
        return f"Peak(n_peaks={len(self)},\n{content}\n)"
    
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
            return self._peak[i]
        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                return self._peak[i]
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

    def assign_formula(self, fragmenter) -> None:
        """
        Assigns the closest formula from the given list to each peak within the given m/z tolerance.

        Parameters:
            formulas (List[Formula]): A list of Formula objects to assign.
            mz_tol (float): Maximum allowed absolute m/z difference to consider a match.
        """
        smiles = self["SMILES"]
        molecule = Molecule(smiles)
        fragment_tree = fragmenter.create_fragment_tree(molecule)

        formulas = fragment_tree.get_all_formulas()
        
        self._peak.assign_formula(formulas)

        pass


    @property
    def is_int_mz(self) -> bool:
        """
        Check if all m/z values are integers.

        Returns:
            bool: True if all m/z values are integers, False otherwise.
        """
        all_integers = np.all(self._peak.np[:, 0] % 1 == 0)
        return all_integers
    
    @property
    def peaks(self) -> PeakSeries:
        """
        Return the underlying PeakSeries object.
        """
        return self._peak

class PeakSeries:
    """
    Represents a series of mass spectral peaks.
    """

    def __init__(self, data: np.ndarray, precursor_formula: Formula = None, fragment_formulas: List[Formula] = None):
        assert isinstance(data, np.ndarray), "PeakSeries data must be a numpy array or PeakSeries"
        assert data.ndim == 2 and data.shape[1] == 2, "data must be a 2D array with shape (n_peaks, 2)"
        assert fragment_formulas is None or (all((isinstance(f, Formula) or f is None) for f in fragment_formulas) and len(fragment_formulas) == data.shape[0]), "fragment_formulas must be a list of Formula objects with the same length as data"

        self._data:List[PeakEntry] = [PeakEntry(mz, intensity) for mz, intensity in data]
        self._precursor_formula = precursor_formula
        if fragment_formulas is not None:
            for i, formula in enumerate(fragment_formulas):
                if formula is not None:
                    self._data[i].formula = formula
        
        self._data.sort(key=lambda x: x.mz)
        pass

    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self):
        contents = [f'\t{line}' for line in str(self).splitlines()]
        content = "\n".join(contents)
        if self._precursor_formula:
            return f"PeakSeries(n_peaks={len(self)}, precursor={self._precursor_formula.exact_mass}({self._precursor_formula}),\n{content}\n)"
        else:
            return f"PeakSeries(n_peaks={len(self)},\n{content}\n)"
    
    def __str__(self):
        return self.format_peak()
    
    def __getitem__(self, i: int | slice | Sequence) -> PeakEntry | PeakSeries:
        """
        Return a single Peak object (for int index) or a new PeakSeries object (for slice or list of indices).
        """
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for PeakSeries with {len(self)} peaks."
            return self._data[i]
        elif isinstance(i, slice):
            return PeakSeries(self.np[i], precursor_formula=self._precursor_formula, fragment_formulas=[d.formula for d in self._data[i]])
        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                return PeakSeries(self.np[i], precursor_formula=self._precursor_formula, fragment_formulas=[self._data[d].formula for d in i])
            else:
                raise IndexError(f"Indices {i} out of range for PeakSeries with {len(self)} peaks.")
        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of int.")
        
    def __iter__(self) -> Iterator[PeakEntry]:
        """
        Iterate over all peaks as tuples of (m/z, intensity).
        """
        for p in self._data:
            yield p

    @property
    def precursor_formula(self) -> Optional[Formula]:
        """
        Formula of the precursor ion (before fragmentation).
        """
        return self._precursor_formula

    @precursor_formula.setter
    def precursor_formula(self, formula: Formula):
        self._precursor_formula = formula

    @property
    def fragment_formulas(self) -> List[Formula]:
        """
        List of formulas for the fragment ions.
        """
        return [peak.formula for peak in self._data]
    
    @property
    def np(self) -> np.ndarray:
        """
        Return the underlying numpy array of m/z and intensity values.
        """
        value = np.array([list(p) for p in self._data])
        return value
    
    @staticmethod
    def parse(peak_str: str) -> PeakSeries:
        """
        Create a PeakSeries object from a string with optional formulas.

        Supports format:
            "mz,intensity;..." or
            "mz,intensity,formula;..."

        Args:
            peak_str (str): String like "100.0,200.0,C6H12O6;150.0,300.0,C7H14O2"

        Returns:
            PeakSeries: A new PeakSeries instance with optional formulas.
        """
        assert isinstance(peak_str, str), "peak_str must be a string"

        peak_list = []
        formula_list = []

        for entry in peak_str.strip().split(";"):
            parts = entry.strip().split(",")
            assert len(parts) >= 2, f"Invalid peak entry: {entry}"
            mz = float(parts[0])
            intensity = float(parts[1])
            peak_list.append([mz, intensity])

            if len(parts) >= 3:
                formula = Formula(parts[2].strip())  # Assume Formula can be constructed from string
            else:
                formula = None
            formula_list.append(formula)

        data = np.array(peak_list)
        return PeakSeries(data, fragment_formulas=formula_list)
    
    def to_str(self, include_formula: bool = True, decimals: int = 6) -> str:
        """
        Convert the PeakSeries into a string of the format:
            "mz,intensity" or "mz,intensity,formula" for each peak.

        Args:
            include_formula (bool): Whether to include formula in the output if available.
            decimals (int): Number of decimal places for mz and intensity.

        Returns:
            str: The string representation.
        """
        fmt = f"{{:.{decimals}f}},{{:.{decimals}f}}"
        lines = []
        for peak in self._data:
            base = fmt.format(peak.mz, peak.intensity)
            if include_formula and peak.formula is not None:
                line = f"{base},{str(peak.formula)}"
            else:
                line = base
            lines.append(line)
        return ";".join(lines)

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
        
        format_str1 = f"{{:>{width}.{decimals}f}}\t{{:>{width}.{decimals}f}}"
        format_str2 = f"{{:>{width}.{decimals}f}}\t{{:>{width}.{decimals}f}}\t{{}}"
        lines = [format_str2.format(p.mz, p.intensity, p.formula) if p.formula else format_str1.format(p.mz, p.intensity) for p in self._data]
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
        
        peaks_np = self.np
        max_intensity = np.max(peaks_np[:, 1])
        if max_intensity > 0:
            normalized = peaks_np[:, 1] / max_intensity * to
            for i, peak in enumerate(self._data):
                peak.intensity = normalized[i]

    def assign_formula(self, formulas: List[Formula], mz_tol: float = MIN_ABS_TOLERANCE) -> None:
        """
        Assigns the closest formula from the given list to each peak within the given m/z tolerance.

        Parameters:
            formulas (List[Formula]): A list of Formula objects to assign.
            mz_tol (float): Maximum allowed absolute m/z difference to consider a match.
        """
        assert isinstance(formulas, list), "formulas must be a list of Formula objects"
        assert all(isinstance(f, Formula) for f in formulas), "All elements in formulas must be Formula objects"

        src_masses = np.array([f.exact_mass for f in formulas])  # shape: (n_formulas,)
        peak_mzs = np.array([p.mz for p in self._data])          # shape: (n_peaks,)

        # Compute the absolute difference matrix between peaks and formulas
        diff_matrix = np.abs(peak_mzs[:, np.newaxis] - src_masses[np.newaxis, :])  # shape: (n_peaks, n_formulas)

        # Find best match (min diff) for each peak
        best_match_idx = np.argmin(diff_matrix, axis=1)
        best_match_diff = diff_matrix[np.arange(diff_matrix.shape[0]), best_match_idx]

        for i, (peak, diff) in enumerate(zip(self._data, best_match_diff)):
            if diff <= mz_tol:
                peak.formula = formulas[best_match_idx[i]]
            else:
                peak.formula = None
        pass

    def assigned_formula_coverage(self, weighted: bool = True) -> float:
        """
        Calculate the proportion of peaks that have an assigned formula.

        Args:
            weighted (bool): If True, weight by intensity. If False, count peaks equally.

        Returns:
            float: The coverage ratio (0.0 to 1.0).
        """
        assigned_weight = 0.0
        total_weight = 0.0

        for peak in self._data:
            weight = peak.intensity if weighted else 1.0
            if peak.formula is not None:
                assigned_weight += weight
            total_weight += weight

        return assigned_weight / total_weight if total_weight > 0 else 0.0

        

            

        

    

class PeakEntry:
    """
    Represents a single mass spectral peak with m/z and intensity.
    """

    def __init__(self, mz: float, int: float, formula: Formula = None):
        self.mz = mz
        self.intensity = int
        self._formula = formula

    def __repr__(self):
        if self.formula:
            return f"PeakEntry(mz={self.mz}, intensity={self.intensity}, formula={self.formula})"
        else:
            return f"PeakEntry(mz={self.mz}, intensity={self.intensity})"
    
    def __str__(self):
        if self.formula:
            return f"m/z: {self.mz}, Intensity: {self.intensity}, Formula: {self.formula}"
        else:
            return f"m/z: {self.mz}, Intensity: {self.intensity}"
    
    def __iter__(self):
        """
        Iterate over the m/z and intensity values.
        """
        yield self.mz
        yield self.intensity

    @property
    def formula(self) -> Optional[Formula]:
        """
        Formula of the peak.
        """
        return self._formula
    
    @formula.setter
    def formula(self, value: Formula):
        assert (value is None) or isinstance(value, Formula), "formula must be a Formula object or None"
        self._formula = value


