from typing import Dict, List, OrderedDict, Literal
import re
from rdkit import Chem
from ..chem.mol.Formula import Formula
from ..ms.utilities import charge_from_str

class Adduct:
    """
    Class representing an adduct ion.
    """

    def __init__(self, mode: Literal["M", "F"], element_diff: Dict[str, int], charge_diff: int):
        """
        Initialize an adduct with element differences and charge difference.
        
        Args:
            mode (str): Mode of the adduct, either "M" or "F". 
            element_diff (Dict[str, int]): Dictionary of element differences.
            charge_diff (int): Charge difference.
        """
        self.mode = mode

        element_diff = OrderedDict({k: element_diff[k] for k in Formula._reorder_element_keys(element_diff.keys()) if element_diff[k] != 0})
                
        self._element_diff = element_diff
        self._charge_diff = charge_diff

        # Calculate the exact mass shift of the adduct
        mass = 0.0
        for elem, count in element_diff.items():
            atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(elem)
            mass += Chem.GetPeriodicTable().GetMostCommonIsotopeMass(atomic_number) * count
        self.exact_mass = mass


    @property
    def charge(self) -> int:
        """
        Get the charge of the adduct.
        
        Returns:
            str: Charge of the adduct.
        """
        return self._charge_diff
    
    def __repr__(self) -> str:
        """
        Get the string representation of the adduct.
        
        Returns:
            str: String representation of the adduct.
        """
        return f'Adduct({self.__str__()})'

    def __str__(self) -> str:
        """
        Get the string representation of the adduct.
        
        Returns:
            str: String representation of the adduct.
        """
        adduct_str = ""
                
        for elem, count in self._element_diff.items():
            if count > 0:
                adduct_str += f"+{count}{elem}" if count > 1 else f"+{elem}"
            elif count < 0:
                adduct_str += f"{count}{elem}" if count < -1 else f"-{elem}"

        # Decide overall charge
        if self._charge_diff > 0:
            charge = f"+{self._charge_diff}" if self._charge_diff > 1 else "+"
        elif self._charge_diff < 0:
            charge = f"{self._charge_diff}" if self._charge_diff < -1 else "-"
        else:
            charge = ""

        return  f"[{self.mode}{adduct_str}]{charge}"
    
    def __eq__(self, value):
        self.__str__() == str(value)

    @staticmethod
    def from_str(adduct_str: str) -> "Adduct":
        """
        Parse an adduct string like "[M+H]+", "[M+2Na-H]-" into an Adduct object.

        Args:
            adduct_str (str): The adduct string.

        Returns:
            Adduct: Parsed Adduct object.
        """
        assert adduct_str.startswith("[") and "]" in adduct_str, f"Invalid adduct format: {adduct_str}"

        # extract content inside brackets and the final charge
        main, charge_part = adduct_str[1:].split("]")
        mode = main[0]  # 'M' or 'F'
        remainder = main[1:]  # e.g., '+H', '+2Na-H'

        # charge part: '+' or '+2' or '-' etc.
        charge = charge_from_str(charge_part)

        # parse element differences using regex
        # pattern: +H, -H, +2Na, -2H2O etc.
        pattern = re.compile(r'([+-])(\d*)([A-Z][a-z]?[0-9]*)')
        element_diff: Dict[str, int] = {}

        for sign, num, elem in pattern.findall(remainder):
            count = int(num) if num else 1
            count = count if sign == '+' else -count
            element_diff[elem] = element_diff.get(elem, 0) + count

        return Adduct(mode=mode, element_diff=element_diff, charge_diff=charge)
                    