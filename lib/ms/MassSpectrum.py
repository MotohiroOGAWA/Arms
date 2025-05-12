from __future__ import annotations

import os

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import dill

from .Peak import Peak

class MassSpectrum:
    def __init__(self, peak_data: Dict[str, List]):
        assert isinstance(peak_data, dict), "peak_data must be a dictionary"
        assert "Peak" in peak_data, "peak_data must contain a 'Peak' key"
        assert len({len(v) for v in peak_data.values()}) == 1, "All lists in peak_data must have the same length"
        self._data = peak_data

    def __repr__(self):
        return f"MassSpectrum(rows={len(self)}, columns={list(self._data.keys())})"

    def __len__(self):
        return len(self._data["Peak"])

    def __getattr__(self, name) -> List:
        if name not in self._data:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return self._data[name]
    
    def __getitem__(self, i: int | slice) -> Peak | List[Peak]:
        """
        Return either a single Peak object (for int index)
        or a list of Peak objects (for slice).
        """
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for MassSpectrum with {len(self)} peaks."
            res = {key: value[i] for key, value in self._data.items()}
            return Peak(res)

        elif isinstance(i, slice):
            # slice → list of Peak
            indices = range(*i.indices(len(self)))
            return [self[j] for j in indices]
        
        elif isinstance(i, list):
            # list of indices → list of Peak
            assert all(0 <= idx < len(self) for idx in i), f"Indices {i} out of range for MassSpectrum with {len(self)} peaks."
            return [self[idx] for idx in i]

        else:
            raise TypeError(f"Invalid index type: {type(i)}. Must be int or slice.")

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
            yield Peak(self._data)[i]


    def save(self, file:str, overwrite=True) -> None:
        # Check if the directory already exists and handle overwrite option
        if not overwrite and os.path.exists(file):
            raise FileExistsError(f"File '{file}' already exists. Set overwrite=True to overwrite the file.")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # Save as dill file
        with open(file, 'wb') as f:
            dill.dump(self._data, f)

    @staticmethod
    def load(file:str) -> MassSpectrum:
        assert os.path.exists(file), f"File '{file}' does not exist."
        with open(file, 'rb') as f:
            data = dill.load(f)
        return MassSpectrum(data)