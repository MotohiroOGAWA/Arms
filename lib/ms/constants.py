from enum import Enum
from rdkit import Chem
from rdkit.Chem import Descriptors

# Disable RDKit logging
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages



class IonMode(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class AdductType(Enum):
    NONE = "None"
    M_PLUS_H_POS = "[M+H]+"
    M_MINUS_H_NEG = "[M-H]-"


PPM = 1/1000000
DEFAULT_PPM = 100 * PPM
DEFAULT_DALTON = 0.05 # equiv: 100ppm of 500 m/z 
MIN_ABS_TOLERANCE = 0.01 # 0.02 # Tolerance aplied for small fragment when relative PPM gets too small



# source: https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator