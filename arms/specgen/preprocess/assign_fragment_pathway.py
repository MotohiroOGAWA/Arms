import os
import sys
from typing import List, Tuple
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
import torch
import pandas as pd
import time

from arms.utils.run_parallel import *
from ...io.utils import derive_file_path
from ...cores.MassEntity.msentity.core import MSDataset, PeakSeries
from ...cores.MassEntity.msentity.io import read_msp, write_msp, read_mgf, write_mgf
from ...cores.MassMolKit.mmkit.chem import Compound
from ...cores.MassMolKit.mmkit.mass import Adduct, MassTolerance, PpmTolerance, DaTolerance

from ...cores.MassMolKit.mmkit.fragment.Fragmenter import Fragmenter
from ...cores.MassMolKit.mmkit.fragment.FragmentPathway import AdductedFragmentPathway
from ...cores.MassMolKit.mmkit.fragment.Fragmenter import Fragmenter
from ...cores.MassMolKit.mmkit.fragment.AdductedFragmentTree import AdductedFragmentTree
from ...cores.MassMolKit.mmkit.fragment.CleavagePatternLibrary import CleavagePatternLibrary
from ...cores.MassMolKit.mmkit.chem.Compound import Compound
from ...cores.MassMolKit.mmkit.mass.constants import parse_ion_mode
from ...cores.MassMolKit.mmkit.mass.Adduct import Adduct
from ...cores.MassMolKit.mmkit.mass.Tolerance import PpmTolerance, DaTolerance
from ...cores.MassMolKit.mmkit.chem.formula_utils import get_isotopic_masses


def _prepare_dataset_io(input_file:str, file_type:str=None, hdf5_output_file:str=None, msp_output_file:str=None, mgf_output_file:str=None, overwrite:bool=False) -> Tuple[str, MSDataset, str, str, str]:
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
    
    if hdf5_output_file is None and msp_output_file is None and mgf_output_file is None:
        hdf5_output_file = derive_file_path(input_file, suffix='_assigned_formula', ext='.hdf5')
        print(f"No output file specified. Using default HDF5 output file: {hdf5_output_file}")
        if os.path.exists(hdf5_output_file) and not overwrite:
            confirm = input(f"Output file '{hdf5_output_file}' already exists. Overwrite? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return None
    return file_type, dataset, hdf5_output_file, msp_output_file, mgf_output_file

def assign_fragment_pathways(
        input_file:str, 
        max_depth:int,
        mass_tolerance:MassTolerance,
        timeout_seconds:float=float('inf'),
        file_type:str=None, 
        hdf5_output_file:str=None,
        msp_output_file:str=None,
        mgf_output_file:str=None,
        pathway_col_name:str='FragmentPathway',
        adduct_type_col_name:str='AdductType', 
        smiles_col_name:str='SMILES',
        ion_mode_col_name:str='IonMode',
        precursor_mz_col_name:str='PrecursorMZ',
        overwrite:bool=False,
        save_interval_sec:float=float('inf')
) -> MSDataset:
    file_type, dataset, hdf5_output_file, msp_output_file, mgf_output_file = _prepare_dataset_io(
        input_file=input_file,
        file_type=file_type,
        hdf5_output_file=hdf5_output_file,
        msp_output_file=msp_output_file,
        mgf_output_file=mgf_output_file,
        overwrite=overwrite
    )

    if adduct_type_col_name not in dataset.columns:
        raise ValueError(f"Adduct type column '{adduct_type_col_name}' not found in dataset.")
    if smiles_col_name not in dataset.columns:
        raise ValueError(f"SMILES column '{smiles_col_name}' not found in dataset.")
    if ion_mode_col_name not in dataset.columns:
        raise ValueError(f"Ion mode column '{ion_mode_col_name}' not found in dataset.")
    if precursor_mz_col_name not in dataset.columns:
        raise ValueError(f"Precursor m/z column '{precursor_mz_col_name}' not found in dataset.")
            
    pathway_col_name = pathway_col_name.strip()
    pathway_cov_col_name = pathway_col_name + 'Cov'

    if overwrite:
        dataset[pathway_cov_col_name] = ''  # Initialize the column
        dataset.peaks[pathway_col_name] = ''  # Initialize peak metadata column
    else:
        if pathway_cov_col_name not in dataset.columns:
            dataset[pathway_cov_col_name] = ''  # Initialize the column
        if dataset.peaks._metadata_ref is None or pathway_col_name not in dataset.peaks._metadata_ref.columns:
            dataset.peaks[pathway_col_name] = ''  # Initialize peak metadata column
    
    
    print("grouping spectra by SMILES...", end="")
    smiles_index_map = dataset.meta.groupby(smiles_col_name).apply(lambda x: x.index.tolist()).to_dict()
    print("done.")

    pbar = tqdm(total=len(dataset), desc="Assigning fragment pathway to peaks", mininterval=1.0)
    last_save_time = time.time()
    success_count = 0
    progress_count = 0
    for smiles, indices in smiles_index_map.items():
        try:
            compound = Compound.from_smiles(smiles)
            ms_dataset_subset:MSDataset = dataset[indices]
            valid_record_sub_idx = defaultdict(lambda: defaultdict(list))
            for i, spectrum_record in enumerate(ms_dataset_subset):
                precursor_type_str = spectrum_record[adduct_type_col_name]
                ion_mode_str = spectrum_record[ion_mode_col_name]
                if precursor_type_str is None or ion_mode_str is None:
                    continue

                ion_mode = parse_ion_mode(ion_mode_str)

                valid_record_sub_idx[ion_mode][precursor_type_str].append(i)

            if len(valid_record_sub_idx) == 0:
                continue

            fragmenter = Fragmenter(
                max_depth=max_depth,
                cleavage_pattern_lib=CleavagePatternLibrary.load_default_positive()
            )
            
            for ion_mode, precursor_type_dict in valid_record_sub_idx.items():
                fragment_tree = None

                valid_precursor_type_dict = defaultdict(list)
                for precursor_type_str, idx_list in precursor_type_dict.items():
                    try:
                        adduct = Adduct.parse(precursor_type_str)
                        for i, spectrum_record in enumerate(ms_dataset_subset[idx_list]):
                            precursor_mz = float(spectrum_record[precursor_mz_col_name])
                            if mass_tolerance.within(precursor_mz, adduct.calc_mz(compound.formula.exact_mass)):
                                valid_precursor_type_dict[precursor_type_str].append(i)
                            # elif 'Br' in compound.formula or 'Cl' in compound.formula:
                            #     isotopic_masses_res = get_isotopic_masses(compound.formula)
                            #     for iso_mass, n_heavy_cl, n_heavy_br in isotopic_masses_res:
                            #         if mass_tolerance.within(precursor_mz, adduct.calc_mz(iso_mass)):
                            #             n_neutrons = n_heavy_cl + n_heavy_br
                            #             valid_precursor_type_dict[precursor_type_str].append(i)
                            #             break
                            else:
                                print(f"Precursor m/z {precursor_mz} not within tolerance for adduct {precursor_type_str} and compound {smiles}")
                    except Exception as e:
                        print(f"Error parsing adduct '{precursor_type_str}': {e}")
                        progress_count += 1
                        continue
                
                for precursor_type_str, idx_list in precursor_type_dict.items():
                    ms_dataset_subsubset = ms_dataset_subset[idx_list]
                    precursor_type = Adduct.parse(precursor_type_str)
                    adducted_tree = None
                    for spectrum_record in ms_dataset_subsubset:
                        try:
                            if spectrum_record[pathway_cov_col_name] != '' and not overwrite:
                                if spectrum_record[pathway_cov_col_name] != '-1':
                                    success_count += 1
                                continue
                            if fragment_tree is None:
                                fragment_tree = fragmenter.create_fragment_tree(compound, ion_mode=ion_mode, timeout_seconds=timeout_seconds)
                            if adducted_tree is None:
                                adducted_tree = AdductedFragmentTree(fragment_tree)
                            fragment_pathways_by_peak = AdductedFragmentPathway.build_pathways_by_peak(
                                adducted_tree=adducted_tree,
                                precursor_type=precursor_type,
                                peaks_mz=[p.mz for p in spectrum_record.peaks],
                                mass_tolerance=mass_tolerance,
                            )
                            
                            total_intensities = sum(p.intensity for p in spectrum_record.peaks)
                            matched_intensity = 0.0
                            fragment_pathways_strs = []
                            for i, fragment_pathways in enumerate(fragment_pathways_by_peak):
                                if len(fragment_pathways) == 0:
                                    fragment_pathways_strs.append("")
                                    continue
                                fragment_pathways_str = AdductedFragmentPathway.list_to_str(fragment_pathways)
                                fragment_pathways_parsed = AdductedFragmentPathway.parse_list(fragment_pathways_str)
                                fragment_pathways_strs.append(fragment_pathways_str)
                                matched_intensity += spectrum_record.peaks[i].intensity
                            coverage = matched_intensity / total_intensities if total_intensities > 0 else 0.0
                            spectrum_record[pathway_cov_col_name] = str(coverage)
                            spectrum_record.peaks[pathway_col_name] = fragment_pathways_strs
                            success_count += 1
                        except Exception as e:
                            print(f"Error assigning fragment pathways for spectrum index {spectrum_record.index}: {e}")
                            spectrum_record[pathway_cov_col_name] = '-1'
                            continue
                        finally:
                            progress_count += 1
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue
        finally:
            pbar.update(len(indices))
            pbar.set_postfix({"Success": f'{success_count}/{progress_count}({success_count/progress_count*100:.1f}%)'})

            
            if hdf5_output_file is not None:
                current_time = time.time()
                if current_time - last_save_time > save_interval_sec:  # Save every 'save_interval_sec' seconds
                    try:
                        os.makedirs(os.path.dirname(hdf5_output_file), exist_ok=True)
                        print(f"\nIntermediate HDF5 file saved to '{hdf5_output_file}'...", end="")
                        dataset.to_hdf5(hdf5_output_file, mode='w')
                        print("done.")
                    except Exception as e:
                        print(f"\nError saving intermediate HDF5 file: {e}")
                    last_save_time = current_time
    pbar.close()
    
    if hdf5_output_file is not None:
        try:
            os.makedirs(os.path.dirname(hdf5_output_file), exist_ok=True)
            print(f"HDF5 file saved to '{hdf5_output_file}'...", end="")
            dataset.to_hdf5(hdf5_output_file, mode='w')
            print("done.")
        except Exception as e:
            print(f"Error saving HDF5 file: {e}")

    if msp_output_file is not None:
        try:
            os.makedirs(os.path.dirname(msp_output_file), exist_ok=True)
            print(f"MSP file saved to '{msp_output_file}'...", end="")
            write_msp(dataset, msp_output_file)
            print("done.")
        except Exception as e:
            print(f"Error saving MSP file: {e}")

    if mgf_output_file is not None:
        try:
            os.makedirs(os.path.dirname(mgf_output_file), exist_ok=True)
            print(f"MGF file saved to '{mgf_output_file}'...", end="")
            write_mgf(dataset, mgf_output_file)
            print("done.")
        except Exception as e:
            print(f"Error saving MGF file: {e}")

    return dataset

def parallel_assign_fragment_pathways(
        executable:str, 
        argv:list[str], 
        num_workers:int, 
        chunk_size:int,
        
        input_file:str, 
        file_type:str=None, 
        hdf5_output_file:str=None,
        msp_output_file:str=None,
        mgf_output_file:str=None,
        overwrite:bool=False,

        smiles_col_name:str='SMILES',
        ) -> None:
    file_type, dataset, hdf5_output_file, msp_output_file, mgf_output_file = _prepare_dataset_io(
        input_file=input_file,
        file_type=file_type,
        hdf5_output_file=hdf5_output_file,
        msp_output_file=msp_output_file,
        mgf_output_file=mgf_output_file,
        overwrite=overwrite
    )
    
    if smiles_col_name not in dataset.columns:
        raise ValueError(f"SMILES column '{smiles_col_name}' not found in dataset.")

    print("Creating Chunks by SMILES...")
    smiles_index_map = dataset.meta.groupby(smiles_col_name).apply(lambda x: x.index.tolist()).to_dict()
    iterator = iter(smiles_index_map.items())
    smiles_chunks:List[List[int]] = []
    while True:
        part = list(islice(iterator, chunk_size))
        if not part:
            break
        smiles_chunks.append([i for _, sub in part for i in sub])
    print(f"Split into {len(smiles_chunks)} chunks (chunk size = {chunk_size})")

    base_dir = os.path.dirname(hdf5_output_file)
    temp_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(hdf5_output_file))[0] + '_temp_parallel_assign_fragment_pathways')
    os.makedirs(temp_dir, exist_ok=True)

    parallel_infoes = []
    pbar = tqdm(total=len(smiles_chunks), desc="Preparing parallel tasks", mininterval=1.0)
    for i, indices in enumerate(smiles_chunks):
        temp_input = os.path.join(temp_dir, f"part_{i}.hdf5")
        temp_output = os.path.join(temp_dir, f"part_{i}_done.hdf5")
        
        subset = dataset[indices]

        cmd_args = argv.copy()
        cmd_args[cmd_args.index(input_file)] = temp_input # Update input file

        def replace_arg(flag_name: str, new_value: str):
            if flag_name in cmd_args:
                idx = cmd_args.index(flag_name)
                cmd_args[idx + 1] = new_value

        def delete_arg(flag_name: str):
            if flag_name in cmd_args:
                idx = cmd_args.index(flag_name)
                cmd_args.pop(idx)
                if len(cmd_args) > idx and not cmd_args[idx].startswith('-'):
                    cmd_args.pop(idx)  # Remove value only if next arg is not another flag

        replace_arg("-o_h5", temp_output)
        replace_arg("--hdf5_output_file", temp_output)
        delete_arg("-o_msp")
        delete_arg("--msp_output_file")
        delete_arg("-o_mgf")
        delete_arg("--mgf_output_file")
        delete_arg("-ftype")
        delete_arg("--file_type")
        replace_arg("--num_workers", "1")
        replace_arg("-n_workers", "1")

        commands = [executable, "-m", "arms.specgen.preprocess.assign_fragment_pathway"] + cmd_args[1:]
        
        subset.to_hdf5(temp_input, mode='w')

        parallel_infoes.append({
            "commands": commands,
            "temp_input": temp_input,
            "temp_output": temp_output,
        })
        pbar.update(1)
    pbar.close()

    commands_list = [info["commands"] for info in parallel_infoes]
    run_parallel_subprocesses(commands_list=commands_list, max_workers=num_workers)

    print("Merging chunk results...")
    output_datasets = []
    for info in parallel_infoes:
        output_dataset = MSDataset.from_hdf5(info["temp_output"])
        output_datasets.append(output_dataset)
    merged_dataset = MSDataset.concat(output_datasets)
    print("Saving merged results...")
    if hdf5_output_file is not None:
        print(f"Merged HDF5 file saved to '{hdf5_output_file}'", end="...")
        merged_dataset.to_hdf5(hdf5_output_file, mode='w')
        print("done.")
    if msp_output_file is not None:
        print(f"Merged MSP file saved to '{msp_output_file}'", end="...")
        write_msp(merged_dataset, msp_output_file)
        print("done.")
    if mgf_output_file is not None:
        write_mgf(merged_dataset, mgf_output_file)
        print(f"Merged MGF file saved to '{mgf_output_file}'", end="...")
        print("done.")

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

    # --- Fragmenter parameters ---
    parser.add_argument(
        "-max_depth", "--max_depth", dest="max_depth",
        type=int, required=True,
        help="Maximum fragmentation depth for generating fragment trees (default: 1)"
    )
    parser.add_argument(
        "-timeout", "--timeout_seconds", dest="timeout_seconds",
        type=float, default=float('inf'),
        help="Maximum time in seconds to wait for a response (default: no timeout)"
    )

    # --- Column names ---
    parser.add_argument(
        "-col_pathway", "--pathway_col_name", dest="fragment_pathway_col_name",
        type=str, default="FragmentPathway",
        help="Column name for assigned fragment pathways in peak metadata"
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

    # --- Parallel processing ---
    parser.add_argument(
        "--num_workers", "-n_workers", dest="num_workers",
        type=int, default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--chunk_size", "-chunk_size", dest="chunk_size",
        type=int, default=-1,
        help="Number of spectra per chunk for parallel processing (default: -1 for no chunking)"
    )

    args = parser.parse_args()

    if args.num_workers > 1:
        if args.chunk_size <= 0:
            print("Error: --chunk_size must be a positive integer when using multiple workers.", file=sys.stderr)
            sys.exit(1)
        executable = sys.executable
        argv = sys.argv
        parallel_assign_fragment_pathways(
            executable=executable,
            argv=argv,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,

            input_file=args.input_file,
            file_type=args.file_type,
            hdf5_output_file=args.hdf5_output_file,
            msp_output_file=args.msp_output_file,
            mgf_output_file=args.mgf_output_file,
            overwrite=args.overwrite,

            smiles_col_name=args.smiles_col_name,
        )
    else:
        # --- Initialize tolerance ---
        if args.tolerance_unit.lower() == "ppm":
            mass_tolerance = PpmTolerance(args.tolerance_value)
        elif args.tolerance_unit.lower() == "da":
            mass_tolerance = DaTolerance(args.tolerance_value)
        else:
            raise ValueError(f"Unsupported tolerance unit: {args.tolerance_unit}")

        # --- Run process ---
        start_time = time.time()
        print("Starting fragment pathway assignment process...\n")

        dataset = assign_fragment_pathways(
            input_file=args.input_file,
            max_depth=args.max_depth,
            mass_tolerance=mass_tolerance,
            timeout_seconds=args.timeout_seconds,
            file_type=args.file_type,
            hdf5_output_file=args.hdf5_output_file,
            msp_output_file=args.msp_output_file,
            mgf_output_file=args.mgf_output_file,
            pathway_col_name=args.fragment_pathway_col_name,
            adduct_type_col_name=args.adduct_type_col_name,
            smiles_col_name=args.smiles_col_name,
            ion_mode_col_name=args.ion_mode_col_name,
            precursor_mz_col_name=args.precursor_mz_col_name,
            overwrite=args.overwrite,
            save_interval_sec=args.save_interval_sec,
        )

        end_time = time.time()
        print(f"\nCompleted in {end_time - start_time:.2f} seconds.")