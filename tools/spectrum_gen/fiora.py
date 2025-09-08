from .base import SpectrumGenerator



class Fiora(SpectrumGenerator):
    """
    Spectrum generator for Fiora.
    """

    def __init__(self):
        super().__init__()

    # conda create -n fiora python=3.10
    # conda activate fiora
    # python -m pip install .
    # conda install -n fiora -c conda-forge pytest
    # PYTHONPATH=. python -m pytest -v tests
    # PYTHONPATH=. fiora-predict -i examples/example_input.csv  -o examples/example_spec.mgf
    def generate(self, molecule):
        """Generate spectrum from a molecule (SMILES or other input)."""
        pass
