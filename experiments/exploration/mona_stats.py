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
ion_mode = 'positive'  # 'positive' or 'negative'
msp_file = get_mona_msp_file(ion_mode)
hdf5_file = get_mona_hdf5_file(ion_mode)

# %%
from arms.cores.MassEntity.msentity.core import MSDataset
from arms.cores.MassEntity.msentity.io import msp

from arms.cores.MassMolKit.mmkit.chem.utilities import to_canonical_smiles

def canonical_map(smiles_list):
    smi_to_canonical = {}
    for smiles in smiles_list:
        try:
            can_smi = to_canonical_smiles(smiles)
            smi_to_canonical[smiles] = can_smi
        except:
            print(f'Invalid SMILES: {smiles}')
    return smi_to_canonical


# %%
if os.path.exists(hdf5_file):
    print(f'Loading existing HDF5 file: {hdf5_file}')
    ms_dataset = MSDataset.from_hdf5(hdf5_file)
else:
    print(f'Converting MSP file to HDF5: {hdf5_file}')
    ms_dataset = msp.read_msp(msp_file)
    smi_to_canonical = canonical_map(ms_dataset['SMILES'].unique())
    ms_dataset['Canonical'] = ms_dataset['SMILES'].map(smi_to_canonical)
    ms_dataset.to_hdf5(hdf5_file)
print(ms_dataset)

# %%
ms_dataset.meta_copy.head()
