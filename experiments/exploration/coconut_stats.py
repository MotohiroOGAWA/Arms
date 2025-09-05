# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
import os

from pathlib import Path
current_dir = Path().resolve()

root_dir = str(Path(current_dir).parents[1])
print(f'Root: {root_dir}')
sys.path.append(root_dir)

# %%
import os
import re
import pandas as pd

date = None
coconut_root_dir = os.path.join(root_dir, 'data', 'raw', 'COCONUT')

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
print("Using directory:", coconut_dir)

# --- find coconut_csv-XX-YYYY.{csv,parquet} ---
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
csv_path = os.path.join(coconut_dir, csv_file)

print("Using file:", csv_path)

if csv_path.endswith(".parquet"):
    compounds_df = pd.read_parquet(csv_path)
else:
    compounds_df = pd.read_csv(csv_path)
    print("Converting CSV to Parquet for future use...")
    compounds_df.to_parquet(csv_path.replace(".csv", ".parquet"), index=False)

print("DataFrame loaded with shape:", compounds_df.shape)
