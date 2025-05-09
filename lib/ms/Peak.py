import numpy as np


class Peak:
    """
    A class to represent a collection of mass spectral peaks.

    Each peak is a pair of [m/z, intensity], and the full data is a 2D numpy array
    of shape (n_peaks, 2).
    """

    def __init__(self, data: np.ndarray, normalize: bool = False):
        """
        Initialize the Peak object with peak data.

        Parameters:
            data (np.ndarray): 2D array with shape (n_peaks, 2) representing [m/z, intensity] pairs.
            normalize (bool): If True, normalize the intensity values to a maximum of 1.0.

        Raises:
            AssertionError: If data is not a 2D array or does not have shape (n, 2).
        """
        assert isinstance(data, np.ndarray), "data must be a NumPy array"
        assert data.ndim == 2 and data.shape[1] == 2, "data must have shape (n, 2)"
        self.data: np.ndarray = data
        if normalize:
            self.normalize_intensity()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return f"{self.data}"
    
    def __call__(self) -> np.ndarray:
        """
        When the instance is called like a function, return the peak data.

        Returns:
            np.ndarray: The peak data array with shape (n_peaks, 2).
        """
        return self.data
    
    def __getattr__(self, name):
        """
        Allow dynamic attribute access for numpy array methods.
        """
        return getattr(self.data, name)

    def __getitem__(self, key):
        """
        Allows indexing and slicing like a numpy array.
        """
        return self.data[key]

    def __setitem__(self, key, value):
        """
        Allows setting values like a numpy array.
        """
        self.data[key] = value

    def __iter__(self):
        """
        Allows iteration over the peaks.
        """
        return iter(self.data)

    def __array__(self):
        """
        Allows automatic conversion to a numpy array when needed.
        """
        return self.data


    def normalize_intensity(self, to: float = 1.0) -> None:
        """
        Normalizes intensity values so that the maximum becomes the given value.

        Parameters:
            to (float): The value to scale the maximum intensity to.
        """
        assert isinstance(to, (int, float)), "to must be a number"
        assert to > 0, "to must be greater than 0"
        assert len(self) > 0, "No peaks to normalize"
        
        max_intensity = np.max(self.data[:, 1])
        if max_intensity > 0:
            self.data[:, 1] = self.data[:, 1] / max_intensity * to

    @property
    def is_int_mz(self) -> bool:
        """
        Check if all m/z values are integers.

        Returns:
            bool: True if all m/z values are integers, False otherwise.
        """
        all_integers = np.all(self.data[:, 0] % 1 == 0)
        return all_integers
