import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable, Union, List, Tuple, Dict, Any
from cores.MassMolKit.Mol.Formula import Formula
from cores.MassMolKit.Mol.formula_utils import get_possible_sub_formulas
from cores.MassMolKit.Mol.Compound import Compound
from cores.MassMolKit.MS.AdductIon import AdductIon
from cores.MassMolKit.MS.constants import AdductType
from cores.MassMolKit.Fragment.Fragmenter import Fragmenter
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset, SpectrumRecord

from ..parallel.dataset_executor import *
from ..parallel.run_parallel import *

import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def _process_assign_formula_chunk(
    chunk_with_smiles: List[str],
    fragmenter: Fragmenter,
    timeout_seconds: int = 10,
) -> Dict[str, Union[List[AdductIon], int]]:
    results = {}
    # for smi in tqdm(chunk_with_smiles):
    for smi in chunk_with_smiles:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            compound = Compound.from_smiles(smi)
            if fragmenter is None:
                formula_list = get_possible_sub_formulas(compound.formula, hydrogen_delta=3)
                results[smi] = {str(formula): str(formula) for formula in formula_list}
            else:
                tree = fragmenter.create_fragment_tree(compound, timeout_seconds=timeout_seconds)
                res = tree.get_all_adduct_ions()
                adduct_ions_str = {str(formula): ','.join([str(adduct_ion) for adduct_ion in adduct_ions]) for formula, adduct_ions in res.items()}
                results[smi] = adduct_ions_str
        except TimeoutError:
            print(f"Timeout for SMILES: {smi}")
            results[smi] = -2
        except Exception as e:
            print(f"Error processing SMILES {smi}: {e}")
            results[smi] = -1
        finally:
            signal.alarm(0)  # Disable the alarm

    return results


def assign_formula(
    input_file: str,
    output_file: str,
    fragmenter: Fragmenter = None,
    mass_tolerance: float = 0.01,
    resume: bool = False,
    timeout_seconds: int = 10,
    smiles_column: str = "SMILES",
    # cov_value_column: str = "AssignFormulaCov",
    # assign_column: str = "AssignedFormula",
):
    """
    Parallel annotation of formulas on an MSDataset-like object.

    Args:
        dataset: list-like or DataFrame-like object with SMILES and peaks.
        fragmenter: fragmenter object with copy() method.
        num_workers: number of processes.
        chunk_size: records per process chunk.
        resume: whether to skip already processed rows.
        timeout_seconds: timeout for formula assignment.
        smiles_column: column name for SMILES strings.
        assign_cov_column: column name for assigned formula coverage.
        possible_cov_column: column name for possible formula coverage.
        formula_func: function to generate possible formulas from SMILES.
        formula_kwargs: kwargs passed to formula_func.
    """
    if fragmenter is None:
        cov_value_column = "PossibleFormulaCov"
        assign_column = "PossibleFormula"
    else:
        cov_value_column = "FragFormulaCov"
        assign_column = "FragFormula"

    dataset = MSDataset.from_hdf5(input_file)

    if cov_value_column not in dataset._columns:
        dataset[cov_value_column] = None

    # --- Step 1: unique SMILES → index list ---
    smi_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(dataset[smiles_column]):
        smi_to_indices[smi].append(i)

    # --- Step 2: resume mask (optional) ---
    if resume:
        assign_covs = np.asarray(dataset[cov_value_column])
        assign_covs = pd.Series(assign_covs).replace({None: np.nan})
        mask = (assign_covs.isna() | (assign_covs == -1))
        valid_indices = np.where(mask)[0].tolist()

        # filter smi_to_indices
        smi_to_indices = {
            smi: [i for i in idxs if i in valid_indices]
            for smi, idxs in smi_to_indices.items()
            if any(i in valid_indices for i in idxs)
        }

    # --- Step 3: chunking by SMILES ---
    unique_smiles = list(smi_to_indices.keys())

    # --- Step 4: execution ---
    results = _process_assign_formula_chunk(unique_smiles, fragmenter.copy() if fragmenter is not None else None, timeout_seconds)

    for smi, adduct_ions_dict in results.items():
        indices = smi_to_indices[smi]
        assigned_list = None
        if fragmenter is None:
            if isinstance(adduct_ions_dict, dict):
                assigned_list = [Formula.parse(formula_str) for formula_str in adduct_ions_dict.keys()]
        else:
            assigned_list: List[AdductIon] = []
            if isinstance(adduct_ions_dict, dict):
                for adduct_ions_str in adduct_ions_dict.values():
                    assigned_list.extend([
                        AdductIon.parse(adduct_str)
                        for adduct_str in adduct_ions_str.split(',')
                    ])
        for i in indices:
            if assigned_list is not None:
                annotate_peaks_with_formulas(
                    dataset[i],
                    assigned_list,
                    mass_tolerance=mass_tolerance,
                    coverage_column=cov_value_column,
                    annotation_column=assign_column,
                )
            else:
                dataset[i][cov_value_column] = adduct_ions_dict  # -1 or -2 for error/timeout
    dataset.to_hdf5(output_file)

def annotate_peaks_with_formulas(
    spectrum: SpectrumRecord,
    candidates: Union[List[Formula], List[AdductIon]],
    mass_tolerance: float,
    coverage_column: str,
    annotation_column: str,
):
    """
    Annotate peaks in a SpectrumRecord with candidate formulas/adduct ions
    within a given mass tolerance, and compute coverage.

    Args:
        spectrum (SpectrumRecord): Spectrum to annotate.
        candidates (List[Formula|AdductIon]): Candidate formulas or adduct ions.
        mass_tolerance (float): Allowed absolute mass difference.
        coverage_column (str): Column name to store coverage value.
        annotation_column (str): Column name to store annotations per peak.

    Returns:
        List[str]: List of annotations (comma-separated candidates) for each peak.
    """
    # Peak m/z and intensity
    mzs = spectrum.peaks.mz
    intensities = spectrum.peaks.intensity

    # Candidate exact masses
    candidate_masses = torch.tensor(
        [c.exact_mass if isinstance(c, Formula) else c.mz for c in candidates],
        dtype=mzs.dtype,
        device=mzs.device
    )

    # Difference matrix: [n_peaks, n_candidates]
    diff_matrix = torch.abs(mzs[:, None] - candidate_masses[None, :])

    # Boolean match mask
    match_mask = diff_matrix <= mass_tolerance

    # Build peak → annotation string mapping
    peak_matches = {
        i: ','.join([str(candidates[j]) for j in torch.where(match_mask[i])[0].tolist()])
        for i in range(len(mzs))
        if match_mask[i].any()
    }

    # Annotation list aligned with all peaks
    annotations = [peak_matches.get(i, '') for i in range(len(mzs))]
    spectrum.peaks[annotation_column] = annotations

    # Coverage calculation
    total_intensity = intensities.sum().item()
    matched_intensity = intensities[list(peak_matches.keys())].sum().item()
    coverage = matched_intensity / total_intensity if total_intensity > 0 else 0.0
    spectrum[coverage_column] = coverage

# python -m utils.assign.assign_formula -i data/raw/MoNA/positive/filtered_mona_positive.hdf5 -o data/raw/MoNA/positive/output_assign.hdf5 --mode PossibleFormula
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assign formulas/adducts to MSDataset spectra")

    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Input HDF5 file path"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=True,
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["PossibleFormula", "FragFormula"],
        help="Annotation mode: PossibleFormula (no fragmenter) or FragFormula (with fragmenter)"
    )
    parser.add_argument(
        "--mass_tolerance",
        type=float,
        default=0.01,
        help="Mass tolerance for peak-to-formula matching (default: 0.01)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip already processed spectra"
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=10,
        help="Timeout per SMILES for fragmentation (default: 10)"
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        default="SMILES",
        help="Column name containing SMILES strings (default: 'SMILES')"
    )

    args = parser.parse_args()

    fragmenter = None if args.mode == "PossibleFormula" else Fragmenter(adduct_type=(AdductType.M_PLUS_H_POS,), max_depth=3)

    assign_formula(
        input_file=args.input_file,
        output_file=args.output_file,
        fragmenter=fragmenter,
        mass_tolerance=args.mass_tolerance,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
        smiles_column=args.smiles_column,
    )
