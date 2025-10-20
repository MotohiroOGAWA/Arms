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
mona_pos_hdf5_file = get_mona_hdf5_file('positive')
mona_neg_hdf5_file = get_mona_hdf5_file('negative')
coconut_dir, coconut_file = get_latest_coconut_dir_file()

# %%
from arms.cores.MassEntity.msentity.core import MSDataset

mona_pos_msds = MSDataset.from_hdf5(mona_pos_hdf5_file)
mona_neg_msds = MSDataset.from_hdf5(mona_neg_hdf5_file)
coconut_compounds_df = read_coconut_file(coconut_file)

print(f'MoNA positive mode: {mona_pos_msds}')
print(f'MoNA negative mode: {mona_neg_msds}')
print(f'COCONUT: {coconut_compounds_df.columns} {len(coconut_compounds_df)} entries')

# %%
mona_pos_unique_smiles = mona_pos_msds['Canonical'].unique()
mona_neg_unique_smiles = mona_neg_msds['Canonical'].unique()
coconut_unique_smiles = coconut_compounds_df['canonical_smiles'].dropna().unique()

# %%
from utils.viz.overlap.venn import plot_venn3

venn3 = ({
    'MoNA positive': set(mona_pos_unique_smiles),
    'MoNA negative': set(mona_neg_unique_smiles),
    'COCONUT': set(coconut_unique_smiles)
})
counts = plot_venn3(venn3, title="Unique SMILES Overlap")
print(counts)
