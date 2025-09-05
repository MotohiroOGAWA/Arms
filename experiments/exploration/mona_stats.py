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
ion_mode = 'negative'  # 'positive' or 'negative'
msp_file = os.path.join(root_dir, 'data', 'raw', 'MoNA', ion_mode, 'MoNA-export-LC-MS-MS_' + ion_mode.capitalize() + '_Mode.msp')
hdf5_file = os.path.join(os.path.dirname(msp_file), f'mona_{ion_mode}.hdf5')

# %%
from cores.MassEntity.MassEntityCore import MSDataset
from cores.MassEntity.MassEntityIO import msp

from cores.MassMolKit.mol.utilities import to_canonical_smiles

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
