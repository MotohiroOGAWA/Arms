

from rdkit import Chem
import numpy as np
from .constants import MIN_ABS_TOLERANCE

def calc_coverage(src_ms: np.ndarray, tgt_ms: np.ndarray, wt_intensity=True, mz_tol=MIN_ABS_TOLERANCE) -> float:
    if src_ms.ndim != 1:
        raise ValueError("src_ms must be a 1D array (m/z array).")
    if wt_intensity and tgt_ms.ndim != 2:
        raise ValueError("tgt_ms must be a 2D array (m/z and intensity pairs).")
    if not wt_intensity and tgt_ms.ndim != 1:
        raise ValueError("tgt_ms must be a 1D array (m/z array).")
    
    # Extract m/z and intensity
    if wt_intensity:
        tgt_mz = tgt_ms[:, 0]
        tgt_int = tgt_ms[:, 1]
    else:
        tgt_mz = tgt_ms
        tgt_int = np.ones_like(tgt_mz)

    # Normalize intensity weights
    tgt_int = tgt_int / np.sum(tgt_int) if np.sum(tgt_int) > 0 else tgt_int

    matched_weight = 0.0
    for mz, weight in zip(tgt_mz, tgt_int):
        if np.any(np.abs(src_ms - mz) <= mz_tol):
            matched_weight += weight

    return matched_weight

