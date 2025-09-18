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
from utils.assign.assign_formula import parallel_assign_formula

if __name__ == "__main__":
    dataset = MSDataset.from_hdf5(os.path.join(root_dir, 'data/raw/MoNA/positive/filtered_mona_positive.hdf5'))
    fragmenter = Fragmenter(adduct_type=(AdductType.M_PLUS_H_POS,), max_depth=8)
    parallel_assign_formula(
        dataset, 
        # fragmenter=fragmenter, 
        fragmenter=None, 
        mass_tolerance=0.01,
        num_workers=1,
        chunk_size=2,
        smiles_column='Canonical',
        )
    pass

# %%
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset
from cores.MassMolKit.Fragment.Fragmenter import Fragmenter, AdductType, Compound

dataset = MSDataset.from_hdf5(os.path.join(root_dir, 'data/raw/MoNA/positive/filtered_mona_positive.hdf5'))
fragmenter = Fragmenter(adduct_type=(AdductType.M_PLUS_H_POS,), max_depth=8)

# %%
dataset = MSDataset.from_hdf5('/workspaces/Arms/mnt/app/data/raw/MoNA/positive/output_assign_mona_positive.hdf5')

# %%
print(dataset[0])

# %%
tree = fragmenter.create_fragment_tree(Compound.from_smiles(dataset[0]['Canonical']))
tree.get_all_adduct_ions()
