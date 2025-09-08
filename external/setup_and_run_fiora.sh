#!/usr/bin/env bash
set -e  # Stop if any command fails

# 1. Create conda environment (if it doesn't already exist)
if conda info --envs | grep -q '^fiora'; then
  echo "[INFO] Conda environment 'fiora' already exists."
else
  echo "[INFO] Creating conda environment 'fiora'..."
  conda create -n fiora python=3.10
fi

# 2. Activate environment
echo "[INFO] Activating environment 'fiora'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fiora

# 3. Install package in editable mode (current project)
echo "[INFO] Installing current project into 'fiora' environment..."
cd "$(dirname "$0")/tools/fiora"
python -m pip install .

# 4. Install pytest
echo "[INFO] Installing pytest..."
conda install -y -n fiora -c conda-forge pytest

# 5. Run tests
echo "[INFO] Running pytest..."
PYTHONPATH=. python -m pytest -v tests

# 6. Run FIORA prediction example
echo "[INFO] Running FIORA prediction example..."
PYTHONPATH=. fiora-predict -i examples/example_input.csv -o examples/example_spec.mgf

echo "[INFO] Done! Output file: examples/example_spec.mgf"
