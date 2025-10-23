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
from constants import *

print(f'Root: {root_dir}')
sys.path.append(root_dir)

# %%
from arms.cores.MassEntity.msentity.core import *
hdf5_file = os.path.join(root_dir, 'data/raw/MoNA/positive/filtered_mona_positive.hdf5')

# %%
dataset = MSDataset.from_hdf5(hdf5_file)
dataset

# %%
print(dataset[0])

# %%
import re

charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75}
nce_instruments = ["Orbitrap", "LC-ESI-QFT", "LC-APCI-ITFT", "Linear Ion Trap", "LC-ESI-ITFT"] # "Flow-injection QqQ/MS",

def NCE_to_eV(nce, precursor_mz, charge=1):
    return nce * precursor_mz / 500 * charge_factor[charge]

def align_CE(ce, precursor_mz, instrument=None):
    if type(ce) == float:
        if ce > 0.0 and ce < 1.0:
            return str(ce) # REMOVE
        if instrument in nce_instruments:
            return NCE_to_eV(ce, precursor_mz)
        return ce

    # --- NEW: Ramp handling (e.g. "Ramp 5-45 V") ---
    if isinstance(ce, str) and ("-" in ce and any(u in ce for u in ["v", "V", "eV", "ev", "keV", "kev"])):

        try:
            values = re.findall(r"[-+]?\d*\.\d+|\d+", ce)
            if len(values) >= 2:
                v1, v2 = map(float, values[:2])
                if "keV" in ce:
                    v1, v2 = v1 * 1000, v2 * 1000
                elif instrument in nce_instruments:
                    v1, v2 = NCE_to_eV(v1, precursor_mz), NCE_to_eV(v2, precursor_mz)
                return (v1, v2)
        except:
            pass

    if "keV" in ce:
        ce = ce.replace("keV", "")
        return float(ce) * 1000
    if "eV" in ce:
        ce = ce.replace("eV", "")
        try:
            return float(ce)
        except:
            return ce
    elif "V" in ce:
        ce = ce.replace("V", "")
        try:
            return float(ce)
        except:
            return ce
    elif "ev" in ce:
        ce = ce.replace("ev", "")
        try:
            return float(ce)
        except:
            return ce
    elif "% (nominal)" in ce:
        try:
            nce = ce.split('% (nominal)')[0].strip().split(' ')[-1]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "(nominal)" in ce:
        try:
            nce = ce.split('(nominal)')[0].strip().split(' ')[-1]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "(NCE)" in ce:
        try:
            nce = ce.strip().split('(NCE)')[0]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "HCD" in ce:
        try:
            nce = ce.strip().split('HCD')[0]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "%" in ce:
        try:
            nce = ce.split('%')[0].strip().split(' ')[-1]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    else:
        try: 
            ce = float(ce)
            if ce > 0.0 and ce < 1.0:
                return str(ce) # REMOVE
            if instrument in nce_instruments:
                return NCE_to_eV(ce, precursor_mz)
            return ce
        except:
            return ce


# %%
cnt = 0
for i in range(len(dataset)):
    ce = dataset[i]['CollisionEnergy']
    try:
        precursor_mz = dataset[i]['PrecursorMZ']
        precursor_mz = float(precursor_mz)
    except (ValueError, TypeError):
        print(f'Error: Precursor MZ: {precursor_mz}')
        continue
    instrument = dataset[i]['Instrument']
    aligned_CE = align_CE(ce, precursor_mz, instrument=instrument)
    if isinstance(aligned_CE, str):
        print(f'Original CE: {ce}, Error')
    else:
        print(f'Original CE: {ce}, Aligned CE: {aligned_CE}')
        cnt += 1
print(f'Total aligned: {cnt}/{len(dataset)}')
