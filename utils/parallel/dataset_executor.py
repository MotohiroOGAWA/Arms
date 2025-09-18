import os
import tempfile
import dill
import importlib
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable, Any, Dict, List, Tuple, Union, Optional
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset



def split_dataset(
    dataset: MSDataset,
    chunk_size: int,
    column: Optional[str] = None,
    output_dir: str = None,
    return_outputs: bool = False,
    output_prefix: str = "processed_",
    show_progress: bool = True,
) -> Union[Tuple[str, List[str]], Tuple[str, List[str], List[str]]]:
    """
    Split MSDataset into chunks and save each chunk as HDF5 in a temporary directory.

    Args:
        dataset (MSDataset): Input dataset.
        chunk_size (int): Number of spectra per chunk.
        by (str): "rows" = split by row count,
                  "column" = split by unique values in a column.
        column (str, optional): Column name for grouping (used only if by="column").
        output_dir (str, optional): Directory where processed files will be placed.
        return_outputs (bool): If True, also return corresponding output file paths.
        output_prefix (str): Prefix for output files (when return_outputs=True).
        show_progress (bool): If True, display tqdm progress bar.

    Returns:
        Tuple:
            (temp_dir, input_files) if return_outputs=False
            (temp_dir, input_files, output_files) if return_outputs=True
    """
    n = len(dataset)

    base_tmp_dir = os.path.join("data", "tmp")
    os.makedirs(base_tmp_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="msdataset_split_", dir=base_tmp_dir)
    input_files: List[str] = []

    if column is None:
        iterator = range(0, n, chunk_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Splitting dataset by rows", unit="chunk")

        for start in iterator:
            end = min(start + chunk_size, n)
            subset = dataset[start:end]
            out_path = os.path.join(temp_dir, f"chunk_{start:06d}_{end:06d}.h5")
            subset.to_hdf5(out_path)
            input_files.append(out_path)

    else:
        if column is None:
            raise ValueError("column must be specified when by='column'")

        col_series = dataset[column]  # pandas.Series
        codes, uniques = pd.factorize(col_series, sort=True)

        col_to_indices = {}
        for code, val in enumerate(uniques):
            indices = np.where(codes == code)[0]
            col_to_indices[val] = indices

        unique_values = list(col_to_indices.keys())
        iterator = range(0, len(unique_values), chunk_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Splitting dataset by column '{column}'", unit="chunk")

        for start in iterator:
            end = min(start + chunk_size, len(unique_values))
            chunk_values = unique_values[start:end]

            indices = [i for val in chunk_values for i in col_to_indices[val]]
            subset = dataset[indices]
            out_path = os.path.join(temp_dir, f"chunk_{column}_{start:06d}_{end:06d}.h5")
            subset.to_hdf5(out_path)
            input_files.append(out_path)

    if return_outputs:
        if output_dir is None:
            output_dir = temp_dir
        os.makedirs(output_dir, exist_ok=True)
        output_files = [
            os.path.join(output_dir, f"{output_prefix}{os.path.basename(f)}")
            for f in input_files
        ]
        return temp_dir, input_files, output_files
    else:
        return temp_dir, input_files


def merge_datasets(
    files: List[str],
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> MSDataset:
    """
    Merge multiple MSDataset files (HDF5) into a single dataset.

    Args:
        files (List[str]): List of HDF5 file paths to merge.
        device (torch.device, optional): Device where the merged dataset will be placed.
                                         Defaults to the device of the first dataset.
        save_path (str, optional): If provided, save the merged dataset to this HDF5 file.

    Returns:
        MSDataset: A single merged dataset.
    """
    if not files:
        raise ValueError("No files provided for merging")
    
    if device is None:
        device = torch.device("cpu")

    # Load all datasets from HDF5
    datasets = [MSDataset.from_hdf5(f, device=device) for f in files]

    # Concatenate datasets
    merged = MSDataset.concat(datasets, device=device)

    # Optionally save to file
    if save_path is not None:
        merged.to_hdf5(save_path)

    return merged