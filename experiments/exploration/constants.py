import os
from typing import Literal
from pathlib import Path
import re
import pandas as pd

current_dir = Path().resolve()
root_dir = str(Path(current_dir).parents[1])

def get_mona_msp_file(ion_mode:Literal['positive', 'negative']) -> str:
    msp_file = os.path.join(root_dir, 'data', 'raw', 'MoNA', ion_mode, 'MoNA-export-LC-MS-MS_' + ion_mode.capitalize() + '_Mode.msp')
    return msp_file

def get_mona_hdf5_file(ion_mode:Literal['positive', 'negative']) -> str:
    hdf5_file = os.path.join(os.path.dirname(get_mona_msp_file(ion_mode)), f'mona_{ion_mode}.hdf5')
    return hdf5_file



def get_coconut_root_dir() -> str:
    coconut_root_dir = os.path.join(root_dir, 'data', 'raw', 'COCONUT')
    return coconut_root_dir

def get_latest_coconut_dir_file(date=None) -> str:
    coconut_root_dir = get_coconut_root_dir()
    if date is None:
        # list directories under coconut_root_dir
        dirs = [
            d for d in os.listdir(coconut_root_dir)
            if os.path.isdir(os.path.join(coconut_root_dir, d)) and d.isdigit()
        ]
        if not dirs:
            raise FileNotFoundError(f"No dated directories found under {coconut_root_dir}")

        # sort by name (YYYYMMDD) and pick the latest
        date = max(dirs)

    coconut_dir = os.path.join(coconut_root_dir, date)
    
    
    pattern = re.compile(r"^coconut_csv-\d{2}-\d{4}\.(csv|parquet)$")
    files = [
        f for f in os.listdir(coconut_dir)
        if pattern.match(f)
    ]

    if not files:
        raise FileNotFoundError(f"No coconut_csv-XX-YYYY.(csv|parquet) found in {coconut_dir}")

    # group by base name (without extension)
    base_to_files = {}
    for f in files:
        base, ext = os.path.splitext(f)
        base_to_files.setdefault(base, []).append(f)

    # choose one file per base, preferring parquet
    selected_files = []
    for base, fs in base_to_files.items():
        if any(f.endswith(".parquet") for f in fs):
            chosen = [f for f in fs if f.endswith(".parquet")][0]
        else:
            chosen = fs[0]
        selected_files.append(chosen)

    # pick the latest one if multiple remain
    csv_file = sorted(selected_files)[-1]
    path = os.path.join(coconut_dir, csv_file)

    return coconut_dir, path

def read_coconut_file(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df
