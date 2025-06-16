from __future__ import annotations

import os

from typing import Tuple, Dict, List, overload
from collections.abc import Sequence
from tqdm import tqdm
import numpy as np
import pandas as pd
import dill
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import signal
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

from lib import Molecule
from .Peak import Peak, PeakSeries
from .PeakConditions import PeakCondition
from ..io.msp_reader import read_msp_file

class MassSpectrum:
    def __init__(self, peak_data: Dict[str, List]):
        assert isinstance(peak_data, dict), "peak_data must be a dictionary"
        assert all(isinstance(v, list) or isinstance(v, pd.Series) for v in peak_data.values()), "All values in peak_data must be lists"
        assert all(isinstance(k, str) for k in peak_data.keys()), "All keys in peak_data must be strings"
        assert "Peak" in peak_data, "peak_data must contain a 'Peak' key"
        assert len({len(v) for v in peak_data.values()}) == 1, "All lists in peak_data must have the same length"

        self._data = peak_data

    def __repr__(self):
        return f"MassSpectrum(rows={len(self)}, columns={list(self._data.keys())})"

    def __len__(self):
        return len(self._data["Peak"])
    
    @overload
    def __getitem__(self, i: int) -> Peak:
        pass

    @overload
    def __getitem__(self, i: slice) -> MassSpectrum:
        pass

    @overload
    def __getitem__(self, i: List) -> MassSpectrum:
        pass

    @overload
    def __getitem__(self, i: str) -> pd.Series:
        pass

    def __getitem__(self, i: int | slice | List[int] | str | List[str]):
        """
        Return a single Peak object (for int index) or a new MassSpectrum object (for slice or list of indices).
        """
        is_row_dir = True
        if isinstance(i, int):
            assert 0 <= i < len(self), f"Index {i} out of range for MassSpectrum with {len(self)} peaks."
            if isinstance(self._data["Peak"][i], str):
                self._data["Peak"][i] = PeakSeries.parse(self._data["Peak"][i])
            # res = {key: value[i] for key, value in self._data.items()}
            return Peak(self._data, i)
        elif isinstance(i, str):
            if i in self._data:
                indices = [i]
                is_row_dir = False
                return pd.Series(self._data[i], name=i)
            else:
                raise KeyError(f"Key '{i}' not found in MassSpectrum data.")

        elif isinstance(i, slice):
            indices = range(*i.indices(len(self)))

        elif isinstance(i, Sequence) or isinstance(i, pd.Series):
            if isinstance(i, pd.Series):
                if i.dtype == 'bool':
                    indices = i[i].index.tolist()
                else:
                    raise TypeError(f"Invalid index type: {type(i)}. Must be int, slice, or list of ints.")
            elif all(isinstance(idx, int) for idx in i):
                if all(0 <= idx < len(self) for idx in i):
                    indices = i
                else:
                    raise IndexError(f"Indices {i} out of range for MassSpectrum with {len(self)} peaks.")
                indices = i
            elif all(isinstance(flag, bool) for flag in i):
                assert len(i) == len(self), f"Boolean index must be the same length as MassSpectrum with {len(self)} peaks."
                indices = [idx for idx, flag in enumerate(i) if flag]
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

    def save(self, file:str, overwrite=True, preview=False) -> None:
        # Check if the directory already exists and handle overwrite option
        if not overwrite and os.path.exists(file):
            raise FileExistsError(f"File '{file}' already exists. Set overwrite=True to overwrite the file.")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file), exist_ok=True)

        original_peaks = self._data["Peak"]
        self._data["Peak"] = [
            p.to_str() if isinstance(p, PeakSeries) else p
            for p in tqdm(original_peaks, desc="Converting peaks to string")
        ]

        # Save as dill file
        with open(file, 'wb') as f:
            dill.dump(self._data, f)

        if preview:
            df = pd.DataFrame(self._data)
            df.head(n=100).to_csv(file + ".preview.tsv", index=False, sep="\t")
        
        self._data["Peak"] = original_peaks

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
            mass_spectrum.save(save_file, overwrite=overwrite)
            
        return mass_spectrum

    def copy(self) -> MassSpectrum:
        """
        Return a deep copy of the MassSpectrum object.
        """
        return MassSpectrum(deepcopy(self._data))

    @staticmethod
    def _process_chunk(chunk_with_indices, fragmenter, timeout_seconds=10):
        signal.signal(signal.SIGALRM, timeout_handler)

        results = []
        for i, peak in chunk_with_indices:
            try:
                signal.alarm(timeout_seconds)  # Set timeout

                smiles = peak['SMILES']
                molecule = Molecule(smiles)
                formula = molecule.formula

                try:
                    peak.assign_formula(fragmenter, 'Formula')
                    assign_cov = peak.peaks.assigned_formula_coverage('Formula')
                except Exception:
                    assign_cov = -1

                try:
                    possible_formulas = formula.get_possible_sub_formulas(hydrogen_delta=3)
                    peak.peaks.assign_formula(possible_formulas, 'PossibleFormula', mode='all')
                    possible_cov = peak.peaks.assigned_formula_coverage('PossibleFormula')
                except Exception:
                    possible_cov = -1

            except TimeoutException as e:
                # print(f"Timeout for index {i}: {e}")
                assign_cov = -2
                possible_cov = -2
            except Exception as e:
                # print(f"Error for index {i}: {e}")
                assign_cov = -1
                possible_cov = -1
            finally:
                signal.alarm(0)  # Cancel alarm

            results.append((i, peak.peaks.to_str(), assign_cov, possible_cov))

        return results

    def parallel_assign_formula(self, fragmenter, num_workers=4, chunk_size=100, resume=False, timeout_seconds=10):
        n = len(self)

        if not 'AssignFormulaCov' in self:
            self['AssignFormulaCov'] = [None] * n
        if not 'PossibleFormulaCov' in self:
            self['PossibleFormulaCov'] = [None] * n

        indices = list(range(n))
        if resume:
            assign_covs = np.asarray(self['AssignFormulaCov'])
            possible_covs = np.asarray(self['PossibleFormulaCov'])
            assign_covs = pd.Series(assign_covs).replace({None: np.nan})
            possible_covs = pd.Series(possible_covs).replace({None: np.nan})
            mask = (assign_covs.isna() | (assign_covs == -1)) | (possible_covs.isna() | (possible_covs == -1))
            indices = np.where(mask)[0].tolist()

        chunks = [
            [(i, self[i].copy()) for i in indices[start:start + chunk_size]]
            for start in tqdm(range(0, len(indices), chunk_size), desc='Creating chunks')
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(MassSpectrum._process_chunk, chunk, fragmenter.copy(), timeout_seconds) for chunk in chunks]
            for f in tqdm(as_completed(futures), total=len(futures), desc='Assigning formulas'):
                results = f.result()
                for i, peaks_str, assign_cov, possible_cov in results:
                    self[i]['Peak'] = peaks_str
                    self[i]['AssignFormulaCov'] = assign_cov
                    self[i]['PossibleFormulaCov'] = possible_cov

    def peaks_any(self, *conditions: PeakCondition, progress: bool = False) -> MassSpectrum:
        """
        Return a new MassSpectrum containing only spectra whose PeakSeries satisfies
        at least one of the provided conditions.

        Parameters:
            *conditions (PeakCondition): One or more conditions to evaluate.
            progress (bool): Whether to show a progress bar. Default is False.

        Returns:
            MassSpectrum: A new MassSpectrum with spectra satisfying any of the conditions.
        """
        assert all(isinstance(c, PeakCondition) for c in conditions), \
            "All arguments must be instances of PeakCondition"

        iterator = range(len(self))
        if progress:
            iterator = tqdm(iterator, desc="Filtering spectra")

        indices = [
            i for i in iterator
            if self[i].peaks.any(*conditions)
        ]

        return self[indices]
    
    def peaks_all(self, *conditions: PeakCondition, progress: bool = False) -> MassSpectrum:
        """
        Return a new MassSpectrum containing only spectra where all given conditions 
        are satisfied by the corresponding PeakSeries.

        Parameters:
            *conditions (PeakCondition): One or more conditions to evaluate.
            progress (bool): Whether to show a progress bar during filtering.

        Returns:
            MassSpectrum: A new MassSpectrum object with filtered spectra.
        """
        assert all(isinstance(c, PeakCondition) for c in conditions), \
            "All conditions must be instances of PeakCondition"

        iterator = range(len(self))
        if progress:
            iterator = tqdm(iterator, desc="Filtering spectra (all conditions)")

        indices = [
            i for i in iterator
            if self[i].peaks.all(*conditions)
        ]

        return self[indices]
    
