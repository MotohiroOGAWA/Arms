from typing import Dict, List, OrderedDict, Literal
from rdkit import Chem
from ..chem.mol.Formula import Formula

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
                