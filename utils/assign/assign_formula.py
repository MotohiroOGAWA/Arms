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
from cores.MassMolKit.Fragment.Fragmenter import Fragmenter
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset, SpectrumRecord


def _process_assign_formula_chunk(
    chunk_with_smiles: List[str],
    fragmenter: Fragmenter,
    timeout_seconds: int = 10,
) -> Dict[str, Union[List[AdductIon], int]]:
    results = {}
    for smi in chunk_with_smiles:
        try:
            try:
                compound = Compound.from_smiles(smi)
                if fragmenter is None:
                    formula_list = get_possible_sub_formulas(compound.formula)
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

        except Exception as e:
            print(f"Unexpected error for SMILES {smi}: {e}")
            results[smi] = -1

    return results


def parallel_assign_formula(
    dataset: MSDataset,
    fragmenter: Fragmenter,
    mass_tolerance: float,
    num_workers: int = 4,
    chunk_size: int = 100,
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

    n = len(dataset)
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
    chunks = [
        unique_smiles[start:start + chunk_size]
        for start in tqdm(range(0, len(unique_smiles), chunk_size), desc="Creating chunks")
    ]

    # --- Step 4: parallel execution ---
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_assign_formula_chunk,
                chunk,
                fragmenter.copy() if fragmenter is not None else None,
                timeout_seconds,
            )
            for chunk in chunks
        ]
        with tqdm(total=len(futures), desc="Assigning formulas") as pbar:
            for f in as_completed(futures):
                results = f.result()
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
                            match_formulas_to_peaks(
                                dataset[i],
                                assigned_list,
                                mass_tolerance=mass_tolerance,
                                cov_value_column=cov_value_column,
                                assign_column=assign_column,
                            )
                pbar.update(1)
            pass

def match_formulas_to_peaks(
    spectrum_record: SpectrumRecord,
    assigned_list: Union[List[Formula], List[AdductIon]],
    mass_tolerance: float,
    cov_value_column: str,
    assign_column: str,
) -> Dict[int, List[int]]:
    """
    Match peaks in a SpectrumRecord to candidate formulas within a given mass tolerance.

    Args:
        spectrum_record (SpectrumRecord): Input spectrum with peaks.
        formula_list (List[Formula]): List of candidate formulas.
        mass_tolerance (float): Allowed absolute mass difference.

    Returns:
        Dict[int, List[int]]: Mapping {peak_index: [formula_indices]} where
            each peak index maps to the indices of formulas within tolerance.
    """
    # [num_peaks]
    mzs = spectrum_record.peaks.mz
    intensities = spectrum_record.peaks.intensity
    # [num_formulas]
    formula_exact_mass = torch.tensor(
        [f.exact_mass if isinstance(f, Formula) else f.mz for f in assigned_list],
        dtype=mzs.dtype,
        device=mzs.device
    )

    # Compute difference matrix: [num_peaks, num_formulas]
    diff_matrix = torch.abs(mzs[:, None] - formula_exact_mass[None, :])

    # Boolean mask for matches
    match_mask = diff_matrix <= mass_tolerance

    # Build result dict
    matches = {
        i: ','.join([str(assigned_list[j]) for j in torch.where(match_mask[i])[0].tolist()])
        for i in range(len(mzs))
        if match_mask[i].any()
    }
    items = [matches.get(i, '') for i in range(len(mzs))]
    spectrum_record.peaks[assign_column] = items

    matched_intensities = sum(intensities[i] for i in matches.keys()).item()
    total_intensity = intensities.sum().item()
    coverage = matched_intensities / total_intensity if total_intensity > 0 else 0.0
    spectrum_record[cov_value_column] = coverage
    return items