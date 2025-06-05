from __future__ import annotations

import numpy as np
from typing import Dict, List, Any
from collections.abc import Sequence
from ..chem.mol.Molecule import Molecule
from ..ms.PeakSeries import PeakSeries, PeakEntry
from .constants import MIN_ABS_TOLERANCE


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


