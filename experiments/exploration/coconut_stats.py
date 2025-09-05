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
import os
import re
import pandas as pd

date = None

coconut_dir, compounds_file = get_latest_coconut_dir_file(date)
print("Using directory:", coconut_dir)
print("Using file:", compounds_file)

compounds_df = read_coconut_file(compounds_file)
print("DataFrame loaded with shape:", compounds_df.shape)
