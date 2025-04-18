import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
from typing import Tuple

import re
import dill

from item_parser import ItemParser
    

def read_msp_file(filepath, encoding='utf-8', save_file=None, overwrite=True) -> pd.DataFrame:
    file_size = os.path.getsize(filepath)
    processed_size = 0
    line_count = 1
    item_parser = ItemParser()

    cols = {} # Create a data list for each column
    # peaks = []
    peak = []
    max_peak_cnt = 0
    record_cnt = 1
    text = ""
    error_text = ""
    error_flag = False
    
    with open(filepath, 'r', encoding=encoding) as f:
        peak_flag = False
        with tqdm(total=file_size, desc="Read msp file", mininterval=0.5) as pbar:
            for line in f.readlines():
                try:
                    if not peak_flag and line == '\n':
                        continue

                    text += line

                    if peak_flag and line == '\n':
                        peak_flag = False

                        if not error_flag:
                            #　エラーが生じなかった場合はデータを保存する
                            if "Peak" not in cols:
                                cols["Peak"] = [""] * record_cnt
                            cols["Peak"][-1] = ";".join([f"{mz},{intensity}" for mz, intensity in peak])
                            max_peak_cnt = max(max_peak_cnt, len(peak))
                        else:
                            # エラーが生じた場合はエラーデータとして保存する
                            error_text += f"Record: {record_cnt}\n" + f"Rows: {line_count}\n"
                            error_text += text + '\n\n'
                            error_flag = False
                            for k in cols:
                                if len(cols[k]) == record_cnt:
                                    cols[k].pop()
                                elif len(cols[k]) > record_cnt:
                                    error_text += f"Error: '{k}' has more data than the record count.\n"
                        text = ""
                        peak = []
                        record_cnt += 1
                        for k in cols:
                            cols[k].append("")
                    elif peak_flag:
                        # Handling cases where peaks are tab-separated or space-separated
                        if len(line.strip().split('\t')) == 2:
                            mz, intensity = line.strip().split('\t')
                        elif len(line.strip().split(' ')) == 2:
                            mz, intensity = line.strip().split(' ')
                        else:
                            raise ValueError(f"Error: '{line.strip()}' was not split correctly.")
                        mz, intensity = float(mz), float(intensity)
                        peak.append([mz, intensity])
                    else:
                        k,v = item_parser.parse(line)
                        if k not in cols:
                            cols[k] = [""] * record_cnt

                        if k == "Comments":
                            # Extract computed SMILES from comments
                            pattern = r'"computed SMILES=([^"]+)"'
                            match = re.search(pattern, v)
                            if match:
                                if "SMILES" not in cols:
                                    cols["SMILES"] = [""] * record_cnt
                                cols["SMILES"][-1] = match.group(1)
                        else:
                            cols[k][-1] = v
                        if k == "NumPeaks":
                            peak_flag = True
                    
                    line_count += 1
                    processed_size = len(line.encode(encoding)) + 1
                    pbar.update(processed_size)
                except Exception as e:
                    text = 'Error: ' + str(e) + '\n' + text
                    error_flag = True
                    pass

        # Append last peak data if file doesn't end with a blank line
        if line != '\n':
            cols["Peak"] = ";".join([f"{mz},{intensity}" for mz, intensity in peak])
            # peaks.append(np.array(peak))
            max_peak_cnt = max(max_peak_cnt, len(peak))

        # Remove last empty rows in metadata
        for k in cols:
            if cols[k][-1] != "":
                break
        else:
            for k in cols:
                del cols[k][-1]
        df = pd.DataFrame(data=cols, columns=cols.keys())

        # Convert data types according to the predefined types
        for c in df.columns:
            if c in msp_column_types:
                if msp_column_types[c] != "str":
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype(msp_column_types[c])

    df['IdxOri'] = df.index

    if save_file is not None:
        try:
            save_msp_data(df, save_file, overwrite=overwrite)
        except FileExistsError as e:
            warnings.warn(str(e))

    if error_text != '':
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(os.path.splitext(filepath)[0] + f"_error_{now}.txt", "w") as f:
            f.write(error_text)
            
    return df

def extract_peaks(msp_df, indices) -> Tuple[np.ndarray]:
    """
    Extracts mass spectral peaks from an MSP DataFrame for multiple rows.

    Parameters:
        msp_df (pandas.DataFrame): A DataFrame containing mass spectral data, 
                                   where the "Peak" column stores peak information as strings.
                                   Each string contains peaks formatted as "mz,intensity;...".
        indices (list, np.ndarray, tuple): A list, numpy array, or tuple of integer indices specifying 
                                           the rows to extract peaks from.

    Returns:
        tuple[np.ndarray]: A tuple of numpy arrays, where each array contains the peaks for a specific row.
                           Each array is a 2D array with shape (n, 2), where n is the number of peaks
                           and each row contains [mz, intensity].

    Raises:
        ValueError: If `indices` is not a list, numpy array, or tuple of integers.
    """
    if isinstance(indices, (list, np.ndarray, tuple)):
        peaks = []
        for i in indices:
            peak_str = msp_df.loc[i, "Peak"]
            peak = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
            peaks.append(peak)
        return tuple(peaks)
    else:
        raise ValueError("indices must be a list/np.ndarray/tuple of int.")
    
def extract_peak(msp_df, idx) -> np.ndarray:
    """
    Extracts mass spectral peaks from an MSP DataFrame for a single row.

    Parameters:
        msp_df (pandas.DataFrame): A DataFrame containing mass spectral data, 
                                   where the "Peak" column stores peak information as strings.
                                   Each string contains peaks formatted as "mz,intensity;...".
        idx (int): An integer index specifying the row to extract peaks from.

    Returns:
        np.ndarray: A numpy array containing the peaks for the specified row.
                    The array is a 2D array with shape (n, 2), where n is the number of peaks,
                    and each row contains [mz, intensity].

    Raises:
        ValueError: If `idx` is not an integer.
    """
    if isinstance(idx, (int, np.integer)):  # Single index
        peak_str = msp_df.loc[idx, "Peak"]
        peak = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
        return peak
    else:
        raise ValueError("idx must be an int.")

def save_msp_data(metadata_df:pd.DataFrame, save_file:str, overwrite=True) -> None:
    # Check if the directory already exists and handle overwrite option
    if not overwrite and os.path.exists(save_file):
        raise FileExistsError(f"Directory '{save_file}' already exists. Set overwrite=True to overwrite the directory.")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    # Save metadata as parquet file
    if save_file.endswith(".parquet"):
        metadata_df.to_parquet(save_file, index=False)
        print(f"peak data saved to '{save_file}'(parquet).")
    elif save_file.endswith(".tsv") or save_file.endswith(".txt"):
        metadata_df.to_csv(save_file, index=False, sep='\t')
        print(f"peak data saved to '{save_file}'(tsv).")
    else:
        raise ValueError(f"Unsupported file format for saving: '{save_file}'.")

def load_msp_data(load_file:str) -> pd.DataFrame:
    if load_file.endswith(".parquet"):
        metadata_df = pd.read_parquet(load_file)
        print(f"peak data loaded from '{load_file}'(parquet).")
    elif load_file.endswith(".tsv") or load_file.endswith(".txt"):
        metadata_df = pd.read_csv(load_file, sep='\t')
        print(f"peak data loaded from '{load_file}'(tsv).")
    else:
        raise ValueError(f"Unsupported file format for loading: '{load_file}'.")
    return metadata_df
    

def normalize_column(value:str) -> str:
    result = value.replace("_", "").replace("/", "").lower()
    return result

# MSP column mappings for parsing and column naming
msp_column_names = {
    "Name": [],
    "Formula": [],
    "InChIKey": [],
    "PrecursorMZ": [],
    "AdductType": ["PrecursorType"],
    "SpectrumType": [],
    "InstrumentType": [],
    "Instrument": [],
    "IonMode": [],
    "CollisionEnergy": [],
    "ExactMass": [],
}

# Column data types for processing
msp_column_types = {
    "PrecursorMZ": "Float32",
    "ExactMass": "Float32",
}


# Convert columns to their predefined data types (strings if not specified)
def convert_to_types_str(columns):
    column_types = {}
    for c in columns:
        if c in msp_column_types:
            column_types[c] = msp_column_types[c]
        else:
            column_types[c] = "str"
    return column_types

# AdductType column data mapping 
precursor_type_data = {
    "[M]+" : ["M", "[M]"],
    "[M+H]+": ["M+H", "[M+H]"],
    "[M-H]-": ["M-H", "[M-H]"],
    "[M+Na]+": ["M+Na", "[M+Na]"],
    "[M+K]+": ["M+K", "[M+K]"],
    "[M+NH4]+": ["M+NH4", "[M+NH4]"],
    }
to_precursor_type = {}
for precursor_type, data in precursor_type_data.items():
    to_precursor_type[precursor_type] = precursor_type
    for aliases in data:
        to_precursor_type[aliases] = precursor_type


if __name__ == "__main__":
    pass
