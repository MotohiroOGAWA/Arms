from __future__ import annotations

import os

from typing import Tuple, Dict, List
from collections.abc import Sequence
import numpy as np
import pandas as pd
import dill

from .Peak import Peak, PeakSeries
from ..io.msp_reader import read_msp_file

class MassSpectrum:
    def __init__(self, peak_data: Dict[str, List]):
        assert isinstance(peak_data, dict), "peak_data must be a dictionary"
        assert all(isinstance(v, list) for v in peak_data.values()), "All values in peak_data must be lists"
        assert all(isinstance(k, str) for k in peak_data.keys()), "All keys in peak_data must be strings"
        assert "Peak" in peak_data, "peak_data must contain a 'Peak' key"
        assert len({len(v) for v in peak_data.values()}) == 1, "All lists in peak_data must have the same length"

        self._data = peak_data
        self._peak_series_indices = set()

    def __repr__(self):
        return f"MassSpectrum(rows={len(self)}, columns={list(self._data.keys())})"

    def __len__(self):
        return len(self._data["Peak"])
    
    def __getitem__(self, i: int | slice | List[int] | str | List[str]) -> Peak | MassSpectrum:
        """
        Return a single Peak object (for int index) or a new MassSpectrum object (for slice or list of indices).
        """
        is_row_dir = True
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for MassSpectrum with {len(self)} peaks."
            if isinstance(self._data["Peak"][i], str):
                self._data["Peak"][i] = PeakSeries.parse(self._data["Peak"][i])
                self._peak_series_indices.add(i)
            res = {key: value[i] for key, value in self._data.items()}
            return Peak(res)
        elif isinstance(i, str):
            if i in self._data:
                indices = [i]
                is_row_dir = False
            else:
                raise KeyError(f"Key '{i}' not found in MassSpectrum data.")

        elif isinstance(i, slice):
            indices = range(*i.indices(len(self)))

        elif isinstance(i, Sequence):
            if all(isinstance(idx, int) for idx in i):
                if all(0 <= idx < len(self) for idx in i):
                    indices = i
                else:
                    raise IndexError(f"Indices {i} out of range for MassSpectrum with {len(self)} peaks.")
                indices = i
            elif all(idx in self._data for idx in i):
                indices = i
                is_row_dir = False
            else:
                raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of ints.")
        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of ints.")

        # Build new dict with sliced lists
        if is_row_dir:
            new_data = {key: [value[j] for j in indices] for key, value in self._data.items()}
            return MassSpectrum(new_data)
        else:
            if 'Peak' not in indices:
                indices.append('Peak')
            new_data = {key: self._data[key] for key in indices}
            return MassSpectrum(new_data)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, item):
        return item in self._data
    
    def __iter__(self):
        """
        Iterate over all peaks as Peak instances.
        """
        for i in range(len(self)):
            yield self[i]

    def save(self, file:str, overwrite=True) -> None:
        # Check if the directory already exists and handle overwrite option
        if not overwrite and os.path.exists(file):
            raise FileExistsError(f"File '{file}' already exists. Set overwrite=True to overwrite the file.")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file), exist_ok=True)

        for i in self._peak_series_indices:
            self._data["Peak"][i] = self._data["Peak"][i].to_str()

        assert all(isinstance(v, str) for v in self._data["Peak"]), "All Peak values must be strings before saving."

        # Save as dill file
        with open(file, 'wb') as f:
            dill.dump(self._data, f)

    @staticmethod
    def load(file:str) -> MassSpectrum:
        assert os.path.exists(file), f"File '{file}' does not exist."
        with open(file, 'rb') as f:
            data = dill.load(f)
        return MassSpectrum(data)
    
    @staticmethod
    def from_msp(filepath, encoding='utf-8', save_file=None, overwrite=True) -> MassSpectrum:
        """
        Read an msp file and return a MassSpectrum object.
        """
        cols = read_msp_file(filepath, encoding=encoding)
        mass_spectrum = MassSpectrum(cols)

        if save_file:
            mass_spectrum.save(save_file, overwrite=overwrite)\
            
        return mass_spectrum