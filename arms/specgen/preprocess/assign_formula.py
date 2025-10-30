import os
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm
import torch
import pandas as pd
import time

from ...cores.MassEntity.msentity.core import MSDataset, PeakSeries
from ...cores.MassEntity.msentity.io import read_msp, write_msp, read_mgf, write_mgf
from ...cores.MassMolKit.mmkit.chem import Compound
from ...cores.MassMolKit.mmkit.mass import Adduct, MassTolerance, PpmTolerance, DaTolerance
from ...cores.MassMolKit.mmkit.chem.formula_utils import get_possible_sub_formulas, assign_formulas_to_peaks, Formula

from ...io.utils import derive_file_path

def assign_formulas(
        input_file:str, 
        mass_tolerance:MassTolerance,
        file_type:str=None, 
        hdf5_output_file:str=None,
        msp_output_file:str=None,
        mgf_output_file:str=None,
        calc_formula_col_name:str='CalcFormula',
        adduct_type_col_name:str='AdductType', 
        smiles_col_name:str='SMILES',
        ion_mode_col_name:str='IonMode',
        precursor_mz_col_name:str='PrecursorMZ',
        overwrite:bool=False,
        save_interval_sec:float=float('inf')
        ) -> MSDataset:
    if file_type is None:
        if input_file.endswith('.msp'):
            file_type = 'msp'
        elif input_file.endswith('.mgf'):
            file_type = 'mgf'
        elif input_file.endswith('hdf5') or input_file.endswith('.h5'):
            file_type = 'hdf5'
        else:
            raise ValueError("Cannot determine file type from extension. Please specify 'file_type' parameter.")
        
    if file_type == 'msp':
        dataset = read_msp(input_file)
    elif file_type == 'mgf':
        dataset = read_mgf(input_file)
    elif file_type == 'hdf5':
        dataset = MSDataset.from_hdf5(input_file)
    else:
        raise ValueError("Unsupported file type. Use 'msp' or 'mgf'.")
    
    if adduct_type_col_name not in dataset.columns:
        raise ValueError(f"Adduct type column '{adduct_type_col_name}' not found in dataset.")
    if smiles_col_name not in dataset.columns:
        raise ValueError(f"SMILES column '{smiles_col_name}' not found in dataset.")
    if ion_mode_col_name not in dataset.columns:
        raise ValueError(f"Ion mode column '{ion_mode_col_name}' not found in dataset.")
    if precursor_mz_col_name not in dataset.columns:
        raise ValueError(f"Precursor m/z column '{precursor_mz_col_name}' not found in dataset.")

    if hdf5_output_file is None and msp_output_file is None and mgf_output_file is None:
        hdf5_output_file = derive_file_path(input_file, suffix='_assigned_formula', ext='.hdf5')
        print(f"No output file specified. Using default HDF5 output file: {hdf5_output_file}")
        if os.path.exists(hdf5_output_file) and not overwrite:
            confirm = input(f"Output file '{hdf5_output_file}' already exists. Overwrite? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return None

    calc_formula_col_name = calc_formula_col_name.strip()
    calc_formula_cov_col_name = calc_formula_col_name+'Cov'
    # if calc_formula_cov_col_name in dataset._spectrum_meta_ref.columns:
    #     print(f"Column '{calc_formula_cov_col_name}' already exists.")
    #     cand_i = 1
    #     col_name = f"{calc_formula_cov_col_name}_{cand_i}"
    #     while col_name in dataset._spectrum_meta_ref.columns:
    #         cand_i += 1
    #         col_name = f"{calc_formula_cov_col_name}_{cand_i}"
    #     calc_formula_cov_col_name = col_name
    #     calc_formula_col_name = calc_formula_col_name+f'_{cand_i}'
    #     print(f"Using new column name '{calc_formula_cov_col_name}' for calculated formulas.")
    if overwrite:
        dataset[calc_formula_cov_col_name] = ''  # Initialize the column
        dataset.peaks[calc_formula_col_name] = ''  # Initialize peak metadata column
    else:
        if calc_formula_cov_col_name not in dataset.columns:
            dataset[calc_formula_cov_col_name] = ''  # Initialize the column
        if calc_formula_col_name not in dataset.peaks._metadata_ref.columns:
            dataset.peaks[calc_formula_col_name] = ''  # Initialize peak metadata column

    pbar = tqdm(total=len(dataset), desc="Grouping by precursor formula", mininterval=1.0)
    precursor_formula_groups = defaultdict(list)
    success_count = 0
    progress_count = 0
    for idx, spectrum_record in enumerate(dataset):
        try:
            compound_smiles = spectrum_record[smiles_col_name]
            adduct_str = spectrum_record[adduct_type_col_name]
            ion_mode = spectrum_record[ion_mode_col_name]
            precursor_mz = float(spectrum_record[precursor_mz_col_name])
            compound = Compound.from_smiles(compound_smiles)
            adduct = Adduct.parse(adduct_str)
            precursor_formula = adduct.calc_formula(compound.formula)
            calc_precursor_mz = adduct.calc_mz(compound.exact_mass)
            if mass_tolerance.within(calc_precursor_mz, precursor_mz):
                precursor_formula_groups[precursor_formula.value].append(idx)
                success_count += 1
            else:
                print(f"Warning: Precursor m/z {precursor_mz} not within tolerance for adduct '{adduct_str}' and formula '{compound.formula.plain}'")
        except Exception as e:
            spectrum_record[calc_formula_cov_col_name] = f"PrecursorMZ mismatch: calc {calc_precursor_mz}"
            print(f"Error processing spectrum index {idx}: {e}")
        finally:
            pbar.update(1)
            progress_count += 1
            pbar.set_postfix({"Success": f'{success_count}/{progress_count}({success_count/progress_count*100:.1f}%)'})
    pbar.close()


    pbar = tqdm(total=len(dataset), desc="Assigning formulas to peaks", mininterval=1.0)
    last_save_time = time.time()
    success_count = 0
    progress_count = 0
    for precursor_formula_value, record_indices in precursor_formula_groups.items():
        precursor_formula = Formula.parse(precursor_formula_value)
        possible_formulas = None
        for idx in record_indices:
            try:
                spectrum_record = dataset[idx]
                # --- Skip if already processed ---
                if spectrum_record[calc_formula_cov_col_name] != '':
                    continue
                if possible_formulas is None:
                    possible_formulas = get_possible_sub_formulas(precursor_formula, hydrogen_delta=1)
                peaks_mz = [p.mz for p in spectrum_record.peaks]
                peak_intensities = [p.intensity for p in spectrum_record.peaks]
                total_intensity = sum(peak_intensities)

                assigned_formula_infoes = assign_formulas_to_peaks(
                    peaks_mz=peaks_mz,
                    formula_candidates=possible_formulas,
                    mass_tolerance=mass_tolerance
                )
                assigned_formulas = [','.join(info['matched_formulas']) for info in assigned_formula_infoes]

                # --- coverage calculation ---
                matched_intensity = sum(peak_intensities[idx] for idx, info in enumerate(assigned_formula_infoes) if len(info["matched_formulas"]) > 0)
                coverage = matched_intensity / total_intensity

                spectrum_record.peaks[calc_formula_col_name] = assigned_formulas
                spectrum_record[calc_formula_cov_col_name] = str(coverage)
                success_count += 1

            except Exception as e:
                print(f"Error assigning formula for spectrum: {e}")
                spectrum_record[calc_formula_cov_col_name] = 'ErrorAssigningFormula'
            finally:
                pbar.update(1)
                progress_count += 1
                pbar.set_postfix({"Success": f'{success_count}/{progress_count}({success_count/progress_count*100:.1f}%)'})

                if hdf5_output_file is not None:
                    current_time = time.time()
                    if current_time - last_save_time > save_interval_sec:  # Save every 'save_interval_sec' seconds
                        try:
                            os.makedirs(os.path.dirname(hdf5_output_file), exist_ok=True)
                            dataset.to_hdf5(hdf5_output_file, mode='w')
                            print(f"\nIntermediate HDF5 file saved to '{hdf5_output_file}'.")
                        except Exception as e:
                            print(f"\nError saving intermediate HDF5 file: {e}")
                        last_save_time = current_time
    pbar.close()
    
    if hdf5_output_file is not None:
        try:
            os.makedirs(os.path.dirname(hdf5_output_file), exist_ok=True)
            dataset.to_hdf5(hdf5_output_file, mode='w')
            print(f"HDF5 file saved to '{hdf5_output_file}'.")
        except Exception as e:
            print(f"Error saving HDF5 file: {e}")

    if msp_output_file is not None:
        try:
            os.makedirs(os.path.dirname(msp_output_file), exist_ok=True)
            write_msp(dataset, msp_output_file)
            print(f"MSP file saved to '{msp_output_file}'.")
        except Exception as e:
            print(f"Error saving MSP file: {e}")

    if mgf_output_file is not None:
        try:
            os.makedirs(os.path.dirname(mgf_output_file), exist_ok=True)
            write_mgf(dataset, mgf_output_file)
            print(f"MGF file saved to '{mgf_output_file}'.")
        except Exception as e:
            print(f"Error saving MGF file: {e}")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Assign possible subformulas to MS/MS peaks based on precursor formula and mass tolerance."
    )

    # --- Input / Output ---
    parser.add_argument(
        "input_file", type=str,
        help="Input file (.msp, .mgf, or .hdf5)"
    )
    parser.add_argument(
        "-ftype", "--file_type", dest="file_type",
        type=str, choices=["msp", "mgf", "hdf5"], default=None,
        help="Explicitly specify input file type"
    )
    parser.add_argument(
        "-o_h5", "--hdf5_output_file", dest="hdf5_output_file",
        type=str, default=None,
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "-o_msp", "--msp_output_file", dest="msp_output_file",
        type=str, default=None,
        help="Output MSP file path"
    )
    parser.add_argument(
        "-o_mgf", "--mgf_output_file", dest="mgf_output_file",
        type=str, default=None,
        help="Output MGF file path"
    )

    # --- Column names ---
    parser.add_argument(
        "-col_calc", "--calc_formula_col_name", dest="calc_formula_col_name",
        type=str, default="CalcFormula",
        help="Column name for assigned (calculated) formulas in peak metadata"
    )
    parser.add_argument(
        "-col_adduct", "--adduct_type_col_name", dest="adduct_type_col_name",
        type=str, default="AdductType",
        help="Column name for adduct type"
    )
    parser.add_argument(
        "-col_smiles", "--smiles_col_name", dest="smiles_col_name",
        type=str, default="SMILES",
        help="Column name for SMILES"
    )
    parser.add_argument(
        "-col_ion_mode", "--ion_mode_col_name", dest="ion_mode_col_name",
        type=str, default="IonMode",
        help="Column name for ion mode"
    )
    parser.add_argument(
        "-col_precursor_mz", "--precursor_mz_col_name", dest="precursor_mz_col_name",
        type=str, default="PrecursorMZ",
        help="Column name for precursor m/z"
    )

    # --- Mass tolerance ---
    parser.add_argument(
        "-tol", "--tolerance_value", dest="tolerance_value",
        type=float, required=True,
        help="Mass tolerance value (e.g., 10 for 10 ppm or 0.01 for 0.01 Da)"
    )
    parser.add_argument(
        "-tol_unit", "--tolerance_unit", dest="tolerance_unit",
        type=str, choices=["ppm", "da"], default="ppm",
        help="Mass tolerance unit (default: ppm)"
    )

    # --- Other ---
    parser.add_argument(
        "--overwrite", "-ow", dest="overwrite",
        action="store_true",
        help="Overwrite existing output files if present"
    )
    parser.add_argument(
        "-save_interval", "--save_interval_sec", dest="save_interval_sec",
        type=float, default=float('inf'),
        help="Interval in seconds for saving intermediate HDF5 results (default: no periodic saving)"
    )

    args = parser.parse_args()

    # --- Initialize tolerance ---
    if args.tolerance_unit.lower() == "ppm":
        mass_tolerance = PpmTolerance(args.tolerance_value)
    elif args.tolerance_unit.lower() == "da":
        mass_tolerance = DaTolerance(args.tolerance_value)
    else:
        raise ValueError(f"Unsupported tolerance unit: {args.tolerance_unit}")

    # --- Run process ---
    start_time = time.time()
    print("Starting formula assignment process...\n")

    dataset = assign_formulas(
        input_file=args.input_file,
        mass_tolerance=mass_tolerance,
        file_type=args.file_type,
        hdf5_output_file=args.hdf5_output_file,
        msp_output_file=args.msp_output_file,
        mgf_output_file=args.mgf_output_file,
        calc_formula_col_name=args.calc_formula_col_name,
        adduct_type_col_name=args.adduct_type_col_name,
        smiles_col_name=args.smiles_col_name,
        ion_mode_col_name=args.ion_mode_col_name,
        precursor_mz_col_name=args.precursor_mz_col_name,
        overwrite=args.overwrite,
        save_interval_sec=args.save_interval_sec,
    )

    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
