from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional, Any, Literal
from collections.abc import Sequence
from ..chem.mol.Molecule import Molecule
from ..chem.mol.Formula import Formula
from ..ms.Adduct import Adduct
from .constants import MIN_ABS_TOLERANCE
from ..common.structures import NamedField


class Peak:
    """
    A class to represent a collection of mass spectral peaks.

    Each peak is a pair of [m/z, intensity], and the full data is a 2D numpy array
    of shape (n_peaks, 2).
    """

    def __init__(self, data: Dict, index:int, normalize: bool = True):
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
        assert isinstance(data["Peak"][index], PeakSeries), "data['Peak'] must be a string"
        self._peak = data["Peak"][index]
        self._data: dict[str, list[Any]] = data
        self._index = index
        if normalize:
            self.normalize_intensity()
    
    def _datai(self, key: str):
        """
        Get the data for a specific key at the current index.
        """
        return self._data[key][self._index]

    def __len__(self) -> int:
        return len(self._peak)
    
    def __str__(self) -> str:
        res = ''
        max_len = max(len(k) for k in self._data.keys())
        for d in self._data:
            if d == "Peak":
                continue
            else:
                res += f"{d:<{max_len+1}}:\t{self._datai(d)}\n"
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
                return self._datai(i)
            else:
                raise KeyError(f"Key '{i}' not found in Peak data.")
        elif isinstance(i, slice):
            return self._peak[i]
        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                return self._peak[i]
            elif all(idx in self._data for idx in i):
                return (self._datai(k) for k in i)
            else:
                raise IndexError(f"Indices {i} out of range for Peak with {len(self)} peaks.")
        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of int.")
    
    def __setitem__(self, key, value):
        self._data[key][self._index] = value

    def __delitem__(self, key):
        raise NotImplementedError("Deleting items from Peak is not supported.")
        del self._data[key][self._index]

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


    def assign_formula(self, fragmenter, column_name:str, mz_tol=MIN_ABS_TOLERANCE) -> None:
        """
        Assigns formulas to the peaks using a fragmenter.

        Parameters:
            formulas (List[Formula]): A list of Formula objects to assign.
            column_name (str): Metadata key to store assigned formulas.
            mz_tol (float): Maximum allowed absolute m/z difference to consider a match.
        """
        smiles = self["SMILES"]
        molecule = Molecule(smiles)
        fragment_tree = fragmenter.create_fragment_tree(molecule)

        formulas = fragment_tree.get_all_formulas()
        
        self._peak.assign_formula(formulas, column_name=column_name, mz_tol=mz_tol),

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
    
    def copy(self) -> Peak:
        """
        Create a copy of the Peak object.

        Returns:
            Peak: A new Peak instance with the same data.
        """
        return Peak({k:[v[self._index]] for k, v in self._data.items()}, 0)

class PeakSeries:
    """
    Represents a series of mass spectral peaks.
    """

    def __init__(self, data: np.ndarray, metadata: Dict[str, List] = {}, is_sorted: bool = True):
        assert isinstance(data, np.ndarray), "PeakSeries data must be a numpy array or PeakSeries"
        assert data.ndim == 2 and data.shape[1] == 2, "data must be a 2D array with shape (n_peaks, 2)"
        assert all([(len(v)==data.shape[0]) for v in metadata.values()]), "metadata must be a dictionary"

        self._data:List[PeakEntry] = [PeakEntry(mz, intensity) for mz, intensity in data]
        self._metadata = metadata
        
        if is_sorted:
            sort_indices = sorted(range(len(self._data)), key=lambda i: self._data[i].mz)
            self._data = [self._data[i] for i in sort_indices]
            self._metadata = {
                key: [values[i] for i in sort_indices]
                for key, values in metadata.items()
            }

    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self):
        contents = [f'\t{line}' for line in str(self).splitlines()]
        content = "\n".join(contents)
        return f"PeakSeries(n_peaks={len(self)},\n{content}\n)"
    
    def __str__(self):
        return self.format_peak()
    
    def __getitem__(self, i: int | slice | Sequence | str) -> PeakEntry | PeakSeries:
        """
        Return a single Peak object (for int index) or a new PeakSeries object (for slice or list of indices).
        """
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for PeakSeries with {len(self)} peaks."
            return self._data[i]
        elif isinstance(i, slice):
            return PeakSeries(self.np[i], metadata={key: values[i] for key, values in self._metadata.items()})
        elif isinstance(i, str):
            if i in self._metadata:
                return self._metadata[i]
            else:
                raise KeyError(f"Key '{i}' not found in PeakSeries metadata.")
        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                return PeakSeries(self.np[i], metadata={key: values[i] for key, values in self._metadata.items()})
            else:
                raise IndexError(f"Indices {i} out of range for PeakSeries with {len(self)} peaks.")
        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of int.")
        
    def __iter__(self) -> Iterator[PeakEntry]:
        """
        Iterate over all peaks as tuples of (m/z, intensity).
        """
        for i, p in enumerate(self._data):
            metadatas = {key: self._data[key][i] for key in self._metadata}
            yield p, metadatas
    
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
        metadata = {}
        meta_specs = []

        entries = peak_str.strip().split(";")
        if len(entries) >= 1 and entries[0].startswith("(") and entries[0].endswith(")"):
            # Strip the parentheses and split by ','
            inside = entries[0][1:-1]  # remove surrounding parentheses
            meta_specs = [
                NamedField(name.strip(), type_str.strip())
                for name, type_str in (pair.split(":") for pair in inside.split(","))
            ]
            entries = entries[1:]  # Skip the first entry which is metadata
            metadata = {name: [] for name in meta_specs}

        for entry in entries:
            parts = entry.strip().split(",")
            assert len(parts) == 2+len(meta_specs), f"Invalid peak entry: {entry}"
            mz = float(parts[0])
            intensity = float(parts[1])
            peak_list.append([mz, intensity])

            for i, name in enumerate(meta_specs):
                metadata[name].append(PeakSeries.convert_metadata(parts[2 + i].strip(), name.type))

        data = np.array(peak_list)
        return PeakSeries(data, metadata=metadata)
    
    @staticmethod
    def convert_metadata(value: str, type_str: str) -> Any:
        """
        Convert a string value to the specified type.

        Args:
            value (str): The string value to convert.
            type_str (str): The type to convert to (e.g. "int", "float", "str").

        Returns:
            Any: The converted value.
        """
        if value is None:
            return None
        
        if value == "None":
            return None
        
        if type_str == "int":
            return int(value)
        elif type_str == "float":
            return float(value)
        elif type_str == "str":
            return value
        elif type_str == "bool":
            if value.lower() in ["true", "1"]:
                return True
            elif value.lower() in ["false", "0"]:
                return False
            else:
                raise ValueError(f"Invalid boolean value: {value}")
        elif type_str == "Formula":
            return Formula(value)
        elif type_str == "List[Formula]":
            return [Formula(v) for v in value.split("|") if v.strip()]
        else:
            raise ValueError(f"Unsupported type: {type_str}")
    
    def to_str(self, include_formula: bool = True, decimals: int = 6) -> str:
        """
        Convert the PeakSeries into a string of the format:
            "mz,intensity" or "mz,intensity,metadata1" for each peak.

        Args:
            include_formula (bool): Whether to include formula in the output if available.
            decimals (int): Number of decimal places for mz and intensity.

        Returns:
            str: The string representation.
        """
        fmt = f"{{:.{decimals}f}},{{:.{decimals}f}}"
        lines = []
        meta_specs: List[NamedField] = list(self._metadata.keys())
        if len(self._metadata) > 0:
            lines.append(f"({','.join([f'{spec}:{spec.type}' for spec in meta_specs])})")

        for i, peak in enumerate(self._data):
            base = fmt.format(peak.mz, peak.intensity)
            if len(self._metadata) > 0:
                m_list = []
                for spec in meta_specs:
                    value = self._metadata[spec][i]
                    if value is None:
                        m_list.append("None")
                    elif isinstance(value, list):
                        m_list.append("|".join(str(v) for v in value))
                    else:
                        m_list.append(str(value))
                m = ",".join(m_list)
                line = f"{base},{m}"
            else:
                line = base
            lines.append(line)
        return ";".join(lines)
    
    def set_metadata(self, key: NamedField, values: List[str]) -> None:
        """
        Set metadata for the PeakSeries.

        Args:
            key (NamedField): The metadata key to set.
            values (List[str]): List of values to assign to the key.
        """
        assert isinstance(key, NamedField), "key must be a NamedField"
        assert len(values) == len(self), f"Length of values ({len(values)}) must match number of peaks ({len(self)})"
        
        self._metadata[key] = [PeakSeries.convert_metadata(v, key.type) for v in values]

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
        
        meta_specs: List[NamedField] = list(self._metadata.keys())
        format_str1 = f"{{:>{width}.{decimals}f}}\t{{:>{width}.{decimals}f}}"
        format_str2 = f"{{:>{width}.{decimals}f}}\t{{:>{width}.{decimals}f}}\t{{}}"

        lines = []
        for i, p in enumerate(self._data):
            if len(self._metadata) > 0:
                meta_strs = []
                for m in meta_specs:
                    val = self._metadata[m][i]
                    if val is None:
                        meta_strs.append('None')
                    elif isinstance(val, list):
                        meta_strs.append(','.join(str(_) for _ in val))
                    else:
                        meta_strs.append(str(val))
                line = format_str2.format(p.mz, p.intensity, '\t'.join(meta_strs))
            else:
                line = format_str1.format(p.mz, p.intensity)
            lines.append(line)

        lines = ['\t'.join(['m/z', 'intensity'] + [str(m) for m in meta_specs])] + lines
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

    def sorted_by_intensity(self, descending: bool = True) -> 'PeakSeries':
        """
        Return a new PeakSeries instance with peaks sorted by intensity.

        Args:
            descending (bool): If True, sort from highest to lowest intensity.
                            If False, sort from lowest to highest.

        Returns:
            PeakSeries: A new PeakSeries sorted by intensity.
        """
        sort_indices = sorted(range(len(self._data)), key=lambda i: self._data[i].intensity, reverse=descending)

        sorted_data = [self._data[i] for i in sort_indices]

        sorted_metadata = {
            key: [values[i] for i in sort_indices]
            for key, values in self._metadata.items()
        }

        sorted_array = np.array([[p.mz, p.intensity] for p in sorted_data])
        return PeakSeries(sorted_array, metadata=sorted_metadata, is_sorted=False)

    def assign_formula(
            self, 
            formulas: List[Formula], 
            column_name: str, 
            mz_tol: float = MIN_ABS_TOLERANCE,
            mode: Literal['best', 'all'] = "best"
            ) -> None:
        """
        Assign one or multiple formulas to each peak based on m/z tolerance.

        Parameters:
            formulas (List[Formula]): A list of Formula objects to assign.
            column_name (str): Metadata key to store assigned formulas.
            mz_tol (float): Maximum allowed absolute m/z difference.
            mode (str): "best" for single closest match, "all" for all within tolerance.
        """
        assert all(isinstance(f, Formula) for f in formulas), "All items must be Formula objects"
        assert isinstance(column_name, str), "column_name must be a NamedField"
        assert mode in ["best", "all"], f"Invalid mode '{mode}'"

        src_masses = np.array([f.exact_mass for f in formulas])
        peak_mzs = np.array([p.mz for p in self._data])

        diff_matrix = np.abs(peak_mzs[:, np.newaxis] - src_masses[np.newaxis, :])

        if mode == "best":
            best_match_idx = np.argmin(diff_matrix, axis=1)
            best_match_diff = diff_matrix[np.arange(len(self._data)), best_match_idx]
            assigned = [
                formulas[best_match_idx[i]] if best_match_diff[i] <= mz_tol else None
                for i in range(len(self._data))
            ]
            column_name_filed = NamedField(column_name, "Formula")
        elif mode == "all":
            assigned = []
            for i in range(len(self._data)):
                matched = [
                    (diff_matrix[i, j], formulas[j])
                    for j in range(len(formulas))
                    if diff_matrix[i, j] <= mz_tol
                ]

                if matched:
                    matched_sorted = sorted(matched, key=lambda x: x[0])
                    assigned.append("|".join(str(f) for _, f in matched_sorted))
                else:
                    assigned.append(None)
            column_name_filed = NamedField(column_name, "List[Formula]")
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        self._metadata[column_name_filed] = assigned


    def assigned_formula_coverage(self, column_name:str, weighted: bool = True) -> float:
        """
        Calculate the proportion of peaks that have an assigned formula.

        Args:
            weighted (bool): If True, weight by intensity. If False, count peaks equally.

        Returns:
            float: The coverage ratio (0.0 to 1.0).
        """
        assert column_name in self._metadata, f"Column '{column_name}' not found in metadata"
        
        assigned_weight = 0.0
        total_weight = 0.0

        for i, peak in enumerate(self._data):
            weight = peak.intensity if weighted else 1.0
            if self._metadata[column_name][i] is not None:
                assigned_weight += weight
            total_weight += weight

        return assigned_weight / total_weight if total_weight > 0 else 0.0


class PeakEntry:
    """
    Represents a single mass spectral peak with m/z and intensity.
    """

    def __init__(self, mz: float, intensity: float):
        self.mz = mz
        self.intensity = intensity

    def __repr__(self):
        return f"PeakEntry({self.__str__()})"
    
    def __str__(self):
        return f"m/z={self.mz}, intensity={self.intensity}"
    
    def __iter__(self):
        """
        Iterate over the m/z and intensity values.
        """
        yield self.mz
        yield self.intensity
