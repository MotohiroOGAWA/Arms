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
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset
from cores.MassMolKit.Fragment.Fragmenter import Fragmenter, AdductType

# %%
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset
from cores.MassMolKit.Fragment.Fragmenter import Fragmenter, AdductType, Compound

dataset = MSDataset.from_hdf5(os.path.join(root_dir, 'data/raw/MoNA/positive/filtered_mona_positive.hdf5'))
fragmenter = Fragmenter(adduct_type=(AdductType.M_PLUS_H_POS,), max_depth=8)

# %%
dataset = MSDataset.from_hdf5(os.path.join(root_dir, 'data/raw/MoNA/positive/output_assign2_mona_positive.hdf5'))
dataset

# %%
dataset = dataset[dataset['AdductType'] == '[M+H]+']
dataset

# %%
f_dataset = dataset[dataset['PossibleFormulaCov'] > 0.8]
f_dataset

# %%
print(f_dataset[f_dataset['FragFormulaCov'] > 0.5][0])

# %%
ff_dataset = f_dataset[f_dataset['FragFormulaCov'] < 0.1]
ff_dataset

# %%
for i in range(len(ff_dataset)):
    print(i, ff_dataset[i]['Name'], ff_dataset[i]['Canonical'])

# %%

print(ff_dataset[45].sort_by_intensity())

# %%
f_dataset.to_hdf5(os.path.join(root_dir, 'data/raw/MoNA/positive/final_assign_mona_positive.hdf5'))

# %%
tree = fragmenter.create_fragment_tree(Compound.from_smiles(dataset[0]['Canonical']))
tree.get_all_adduct_ions()
