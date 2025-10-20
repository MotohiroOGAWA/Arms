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

from ...io.utils import derive_file_path

def collect_peaks_by_condition(
        input_file:str, 
        intensity_threshold:float, 
        mass_tolerance:MassTolerance,
        file_type:str=None, 
        hdf5_output_file:str=None,
        msp_output_file:str=None,
        mgf_output_file:str=None,
        adduct_type_col_name:str='AdductType', 
        smiles_col_name:str='SMILES',
        ion_mode_col_name:str='IonMode',
        precursor_mz_col_name:str='PrecursorMZ',
        overwrite:bool=False
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
        hdf5_output_file = derive_file_path(input_file, suffix='_grouped_peak', ext='.hdf5')
        print(f"No output file specified. Using default HDF5 output file: {hdf5_output_file}")
        if os.path.exists(hdf5_output_file) and not overwrite:
            confirm = input(f"Output file '{hdf5_output_file}' already exists. Overwrite? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return None

    record_groups, spectrum_groups = _group_compound_adduct(
        dataset=dataset,
        adduct_type_col_name=adduct_type_col_name,
        smiles_col_name=smiles_col_name,
        ion_mode_col_name=ion_mode_col_name,
        precursor_mz_col_name=precursor_mz_col_name,
        intensity_threshold=intensity_threshold,
        mass_tolerance=mass_tolerance,
    )

    spectrum_metadata = defaultdict(list)
    mz_int_data = []
    offsets = [0]
    peak_metadata   = defaultdict(list)

    pbar = tqdm(total=len(record_groups), desc="Merging records by precursor ion", mininterval=1.0)

    for group_key, record_indices in record_groups.items():
        compound_smiles, adduct_str, ion_mode = group_key
        compound = Compound.from_smiles(compound_smiles)
        adduct = Adduct.parse(adduct_str)
        mz_int_pairs = spectrum_groups[group_key]
        merged_peaks = _merge_close_peaks(mz_int_pairs, mass_tolerance)

        spectrum_metadata['SMILES'].append(compound_smiles)
        spectrum_metadata['PrecursorMZ'].append(adduct.calc_mz(compound.exact_mass))
        spectrum_metadata['AdductType'].append(adduct_str)
        spectrum_metadata['IonMode'].append(ion_mode)
        spectrum_metadata['Formula'].append(compound.formula.plain)
        spectrum_metadata['GroupedRecordIndices'].append(','.join(map(str, record_indices)))
        spectrum_metadata['NumSpectraInGroup'].append(len(record_indices))

        offset = offsets[-1]
        for spec_idx, merged_peak in enumerate(merged_peaks):
            mz, intensity, original_indices = merged_peak

            mz_int_data.append((mz, intensity))
            peak_metadata["NumMerged"].append(len(original_indices.split(',')))
            peak_metadata["SpectrumIndex"].append(original_indices)
            offset += 1
        offsets.append(offset)
        pbar.update(1)
            
    pbar.close()


    mz_int_data_tensor = torch.tensor(mz_int_data, dtype=torch.float32)
    peak_metadata = pd.DataFrame(peak_metadata)
    offsets_tensor = torch.tensor(offsets)
    peaks = PeakSeries(data=mz_int_data_tensor, metadata=peak_metadata, offsets=offsets_tensor)

    spectrum_df = pd.DataFrame(spectrum_metadata)
    output_dataset = MSDataset(spectrum_meta=spectrum_df, peak_series=peaks)

    if hdf5_output_file is not None:
        try:
            os.makedirs(os.path.dirname(hdf5_output_file), exist_ok=True)
            output_dataset.to_hdf5(hdf5_output_file, mode='w')
            print(f"HDF5 file saved to '{hdf5_output_file}'.")
        except Exception as e:
            print(f"Error saving HDF5 file: {e}")

    if msp_output_file is not None:
        try:
            os.makedirs(os.path.dirname(msp_output_file), exist_ok=True)
            write_msp(output_dataset, msp_output_file)
            print(f"MSP file saved to '{msp_output_file}'.")
        except Exception as e:
            print(f"Error saving MSP file: {e}")

    if mgf_output_file is not None:
        try:
            os.makedirs(os.path.dirname(mgf_output_file), exist_ok=True)
            write_mgf(output_dataset, mgf_output_file)
            print(f"MGF file saved to '{mgf_output_file}'.")
        except Exception as e:
            print(f"Error saving MGF file: {e}")

    return output_dataset


def _group_compound_adduct(
        dataset: MSDataset,
        adduct_type_col_name: str,
        smiles_col_name: str,
        ion_mode_col_name: str,
        precursor_mz_col_name: str,
        intensity_threshold: float,
        mass_tolerance: MassTolerance,
    ) -> Tuple[dict, dict]:

    pbar = tqdm(total=len(dataset), desc="Grouping records by smiles and adduct", mininterval=1.0)
    success_count = 0
    progress_count = 0
    record_groups:dict[(Compound,Adduct), list[int]] = defaultdict(list)
    spectrum_groups:dict[(Compound,Adduct), list[tuple]] = defaultdict(list)
    for idx, record in enumerate(dataset):
        adduct_type = record[adduct_type_col_name]
        smiles = record[smiles_col_name]
        ion_mode = record[ion_mode_col_name]
        precursor_mz = record[precursor_mz_col_name]

        try:
            compound = Compound.from_smiles(smiles)
            adduct = Adduct.parse(adduct_type)
            precursor_mz = float(precursor_mz)
            
            if not mass_tolerance.within(adduct.calc_mz(compound.exact_mass), precursor_mz):
                error_mz = adduct.calc_mz(compound.exact_mass) - precursor_mz
                raise ValueError(f"Precursor m/z {precursor_mz} does not match calculated m/z({error_mz} ) for adduct {adduct} and compound {compound}.")
            r = record.normalize(scale=1.0, in_place=False)
            mz_int_pairs = []
            for mz, intensity in r.peaks:
                if intensity < intensity_threshold:
                    continue
                mz_int_pairs.append((mz, intensity, idx))
            ckey = (compound.smiles, str(adduct), ion_mode)
            record_groups[ckey].append(idx)
            spectrum_groups[ckey].extend(mz_int_pairs)
            success_count += 1
        except Exception as e:
            print(f"Error parsing compound or adduct for record {idx}: {e}")
            continue
        finally:
            progress_count += 1
            pbar.update(1)
            pbar.set_postfix({"Groups": len(record_groups), "Success": f'{success_count}/{progress_count}({success_count/progress_count*100:.1f}%)'})
    pbar.close()

    return record_groups, spectrum_groups



def _merge_close_peaks(
    mz_intensity_index_list: List[Tuple[float, float, int]],
    mass_tolerance: MassTolerance
) -> List[Tuple[float, float, str]]:
    # --- Sort peaks by m/z value ---
    mz_intensity_index_list.sort(key=lambda x: x[0])

    merged_peaks = []
    current_mz = float('-inf')
    mz_group = []
    intensity_group = []
    spectrum_index_group = []

    # --- Iterate through all peaks ---
    for mz, intensity, spectrum_index in mz_intensity_index_list:
        # If current peak is close to the previous group → merge it
        if (mass_tolerance / 2).within(mz, current_mz):
            mz_group.append(mz)
            intensity_group.append(intensity)
            spectrum_index_group.append(spectrum_index)
            # Update current representative m/z as mean of group
            current_mz = sum(mz_group) / len(mz_group)
        else:
            # Save the current group before starting a new one
            if len(mz_group) > 0:
                avg_mz = sum(mz_group) / len(mz_group)
                avg_intensity = sum(intensity_group) / len(intensity_group)
                involved_indices = sorted(set(spectrum_index_group))
                merged_peaks.append(
                    (avg_mz, avg_intensity, ','.join(map(str, involved_indices)))
                )

            # Start a new group
            mz_group = [mz]
            intensity_group = [intensity]
            spectrum_index_group = [spectrum_index]
            current_mz = mz

    # --- Add the last group if it exists ---
    if len(mz_group) > 0:
        avg_mz = sum(mz_group) / len(mz_group)
        avg_intensity = sum(intensity_group) / len(intensity_group)
        involved_indices = sorted(set(spectrum_index_group))
        merged_peaks.append(
            (avg_mz, avg_intensity, ','.join(map(str, involved_indices)))
        )

    return merged_peaks

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect and merge peaks by compound/adduct/ion mode with mass tolerance filtering."
    )

    parser.add_argument(
        "input_file", type=str,
        help="Input file (.msp, .mgf, or .hdf5)"
    )
    parser.add_argument(
        "-int_thres", "--intensity_threshold", dest="intensity_threshold",
        type=float, default=0.0,
        help="Minimum intensity threshold for peaks"
    )
    parser.add_argument(
        "-tol", "--tolerance_value", dest="tolerance_value",
        type=float, required=True,
        help="Mass tolerance value"
    )
    parser.add_argument(
        "-tol_unit", "--tolerance_unit", dest="tolerance_unit",
        type=str, choices=["ppm", "da"], default="ppm",
        help="Mass tolerance unit (default: ppm)"
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
    parser.add_argument(
        "--overwrite", "-ow", dest="overwrite",
        action="store_true",
        help="Overwrite existing output files if present"
    )

    args = parser.parse_args()

    if args.tolerance_unit.lower() == "ppm":
        mass_tolerance = PpmTolerance(args.tolerance_value)
    elif args.tolerance_unit.lower() == "da":
        mass_tolerance = DaTolerance(args.tolerance_value)

    start_time = time.time()
    print("Starting peak collection process...\n")
    dataset = collect_peaks_by_condition(
        input_file=args.input_file,
        intensity_threshold=args.intensity_threshold,
        mass_tolerance=mass_tolerance,
        file_type=args.file_type,
        hdf5_output_file=args.hdf5_output_file,
        msp_output_file=args.msp_output_file,
        mgf_output_file=args.mgf_output_file,
        adduct_type_col_name=args.adduct_type_col_name,
        smiles_col_name=args.smiles_col_name,
        ion_mode_col_name=args.ion_mode_col_name,
        precursor_mz_col_name=args.precursor_mz_col_name,
        overwrite=args.overwrite
    )
    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
