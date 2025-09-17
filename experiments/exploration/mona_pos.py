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
from cores.MassEntity.MassEntityCore import MSDataset
from cores.MassEntity.MassEntityIO import msp

print(f'Loading existing HDF5 file: {hdf5_file}')
ms_dataset = MSDataset.from_hdf5(hdf5_file)
print(ms_dataset)

# %%
ms_dataset = ms_dataset[ms_dataset['SMILES'] != '']  # Filter out entries with empty SMILES

# %%
ms_dataset.to('cuda')

# %%
ms_dataset.peak_series._index.device

# %%
# ms_dataset.peak_series.sorted_by_intensity(ascending=False, in_place=True, method='batch')

# %%
ms_dataset.meta_copy.head()

# %%
ms_dataset['AdductType'].value_counts()

# %%
ms_dataset.peak_series.normalize(scale=1.0, in_place=True)

# %%
from cores.MassEntity.MassEntityCore.PeakCondition import *
peak_condition = IntensityThreshold(threshold=0.01) & TopKIntensity(k=100)
filter_ms_dataset = ms_dataset.filter(peak_condition)
filter_ms_dataset

# %%
filter_ms_dataset.peak_series.sort_by_mz(in_place=True)

# %%
print(filter_ms_dataset[0])

# %%
filter_ms_dataset.to_hdf5(os.path.join(os.path.dirname(hdf5_file), f'filtered_{os.path.split(hdf5_file)[-1]}'))

# %%
sorted_ms_dataset = ms_dataset.sort_by('ExactMass')
sorted_ms_dataset

# %%
print(sorted_ms_dataset[0])

# %%
