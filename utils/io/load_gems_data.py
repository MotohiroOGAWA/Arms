import h5py
import numpy as np

def print_hdf5_structure(file_path):
    """Print the group/dataset structure of an HDF5 file."""

    def print_group(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_group)

def load_hdf5_dataset(file_path: str, dataset_name: str) -> np.ndarray:
    """
    Load a specific dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset inside the file (e.g., "spectra/mz").

    Returns:
        np.ndarray: The dataset contents as a numpy array.
    """
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}")
        return f[dataset_name][:]  # Convert to numpy array

def load_gems_data(file_path):
    """
    Load GeMS data from an HDF5 file.

    [Dataset] RT, shape=(23517534,), dtype=float32
    [Dataset] charge, shape=(23517534,), dtype=int8
    [Dataset] instrument accuracy est., shape=(23517534,), dtype=float32
    [Dataset] lsh, shape=(23517534,), dtype=int64
    [Dataset] name, shape=(23517534,), dtype=object
    [Dataset] precursor_mz, shape=(23517534,), dtype=float32
    [Dataset] spectrum, shape=(23517534, 2, 128), dtype=float64
    """
    precursor_mz = load_hdf5_dataset(file_path, "precursor_mz")
    mz = load_hdf5_dataset(file_path, "spectrum")
    pass


if __name__ == "__main__":
    file_path = "data/raw/DreaMS/GeMS/GeMS_A10.hdf5"
    file_path = "/home/user/workspace/mnt/app/data/raw/DreaMS/GeMS/GeMS_A10.hdf5"
    print_hdf5_structure(file_path)

    load_gems_data(file_path)
