from __future__ import annotations

import os

from typing import Tuple
import numpy as np
import pandas as pd

from .Peak import Peak

class MassSpectrum:
    def __init__(self, peak_df):
        assert isinstance(peak_df, pd.DataFrame), "peak_df must be a pandas DataFrame"
        assert "Peak" in peak_df.columns, "peak_df must contain a 'Peak' column"
        self._df: pd.DataFrame = peak_df

    def __repr__(self):
        return f"MassSpectrum(rows={len(self)}, columns={self._df.columns})"

    def __len__(self):
        return len(self._df)
    

    def extract_peaks(self, indices, normalize: bool = False) -> Tuple[Peak]:
        """
        Extracts mass spectral peaks from an MSP DataFrame for multiple rows.

        Parameters:
            indices (list, np.ndarray, tuple): A list, numpy array, or tuple of integer indices specifying 
                                            the rows to extract peaks from.
            normalize (bool): If True, normalize the intensity values to a maximum of 1.0.

        Returns:
            tuple[Peak]: A tuple of numpy arrays, where each array contains the peaks for a specific row.
                            Each array is a 2D array with shape (n, 2), where n is the number of peaks
                            and each row contains [mz, intensity].

        Raises:
            ValueError: If `indices` is not a list, numpy array, or tuple of integers.
        """
        if isinstance(indices, (list, np.ndarray, tuple)):
            peaks = []
            for i in indices:
                _peak = self.extract_peak(i, normalize=normalize)
                peaks.append(_peak)
            return tuple(peaks)
        else:
            raise ValueError("indices must be a list/np.ndarray/tuple of int.")
        
    def extract_peak(self, idx, normalize: bool = False) -> Peak:
        """
        Extracts mass spectral peaks from an MSP DataFrame for a single row.

        Parameters:
            idx (int): An integer index specifying the row to extract peaks from.
            normalize (bool): If True, normalize the intensity values to a maximum of 1.0.

        Returns:
            Peak: A numpy array containing the peaks for the specified row.
                        The array is a 2D array with shape (n, 2), where n is the number of peaks,
                        and each row contains [mz, intensity].

        Raises:
            ValueError: If `idx` is not an integer.
        """
        if isinstance(idx, (int, np.integer)):  # Single index
            peak_str = self._df.loc[idx, "Peak"]
            peak = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
            peak = Peak(peak, normalize=normalize)
            return peak
        else:
            raise ValueError("idx must be an int.")
        

    def save(self, file:str, overwrite=True) -> None:
        # Check if the directory already exists and handle overwrite option
        if not overwrite and os.path.exists(file):
            raise FileExistsError(f"File '{file}' already exists. Set overwrite=True to overwrite the file.")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # Save metadata as parquet file
        self._df.to_parquet(file, index=False)
        
    def save_tsv(self, file:str, overwrite=True) -> None:
        # Check if the directory already exists and handle overwrite option
        if not overwrite and os.path.exists(file):
            raise FileExistsError(f"File '{file}' already exists. Set overwrite=True to overwrite the file.")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # Save metadata as tsv file
        self._df.to_csv(file, index=False, sep='\t')

    @staticmethod
    def load(file:str) -> MassSpectrum:
        assert os.path.exists(file), f"File '{file}' does not exist."
        df = pd.read_parquet(file)
        return MassSpectrum(df)
    
    @staticmethod
    def load_tsv(file:str) -> MassSpectrum:
        assert os.path.exists(file), f"File '{file}' does not exist."
        df = pd.read_csv(file, sep='\t')
        return MassSpectrum(df)