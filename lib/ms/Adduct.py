from typing import Dict, List, OrderedDict, Literal
import re
from rdkit import Chem
from collections import defaultdict
from ..chem.mol.Formula import Formula
from ..ms.utilities import charge_from_str

class Adduct:
    """
    Class representing an adduct ion.
    """
    def __init__(
            self, 
            mode: Literal["M", "F"], 
            adducts_in: List[Formula] = [], 
            adducts_out: List[Formula] = [], 
            charge_diff: int = 0
            ):
        """
        Initialize an adduct with element differences and charge difference.
        
        Args:
            mode (str): Mode of the adduct, either "M" or "F". 
            element_diff (Dict[str, int]): Dictionary of element differences.
            charge_diff (int): Charge difference.
        """
        self.mode = mode
        formula_count_in: dict[str, int] = defaultdict(int)
        charge = 0
        for f in adducts_in:
            formula_count_in[f.raw_formula] += 1
            charge += f.charge
        for f in adducts_out:
            formula_count_in[f.raw_formula] -= 1
            charge -= f.charge
        
        self._adduct_formulas: dict[Formula, int] = {Formula(f): cnt for f, cnt in formula_count_in.items()}
        self.charge = charge + charge_diff

    @property
    def mass_shift(self) -> float:
        """
        Get the mass shift of the adduct.
        
        Returns:
            float: Mass shift of the adduct.
        """
        total_mass = sum(f.exact_mass * cnt for f, cnt in self._adduct_formulas.items() if cnt != 0)
        return total_mass
    
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
        parts = []
        for f, cnt in sorted(self._adduct_formulas.items(), key=lambda x: (-x[0].exact_mass, x[0].raw_formula)):
            if cnt > 0:
                parts.append(f"+{cnt if cnt > 1 else ''}{f.raw_formula}")
            elif cnt < 0:
                parts.append(f"-{abs(cnt) if cnt < -1 else ''}{f.raw_formula}")
        body = "".join(parts)

        if self.charge > 0:
            charge = f"+{self.charge}" if self.charge > 1 else "+"
        elif self.charge < 0:
            charge = f"{self.charge}" if self.charge < -1 else "-"
        else:
            charge = ""
            
        return f"[{self.mode}{body}]{charge}"
    
    def __eq__(self, value):
        return self.__str__() == str(value)
    
    def __hash__(self):
        """
        Get the hash of the adduct.
        
        Returns:
            int: Hash of the adduct.
        """
        return hash(self.__str__())
    
    def copy(self) -> "Adduct":
        """
        Create a copy of the adduct.
        
        Returns:
            Adduct: A new Adduct instance with the same properties.
        """
        adduct = Adduct(
            mode=self.mode,
            adducts_in=[f.copy() for f in self._adduct_formulas.keys() if self._adduct_formulas[f] > 0],
            adducts_out=[f.copy() for f in self._adduct_formulas.keys() if self._adduct_formulas[f] < 0],
            charge_diff=0
        )
        adduct.charge = self.charge
        return adduct

    @staticmethod
    def from_str(adduct_str: str) -> "Adduct":
        """
        Parse an adduct string like "[M+HCOOH-H]-" into Adduct(adducts_in=[HCOOH], adducts_out=[H]).

        Args:
            adduct_str (str): The adduct string.

        Returns:
            Adduct: Parsed Adduct object.
        """
        assert adduct_str.startswith("[") and "]" in adduct_str, f"Invalid adduct format: {adduct_str}"

        # Extract inner content and charge symbol
        main, charge_part = adduct_str[1:].split("]")
        mode = main[0]
        remainder = main[1:]
        charge = charge_from_str(charge_part)

        # Regex to capture: +H, -H, +2Na, -2HCOOH etc.
        pattern = re.compile(r'([+-])(\d*)([A-Z][a-zA-Z0-9]*)')
        adducts_in = []
        adducts_out = []

        for sign, num, formula_str in pattern.findall(remainder):
            count = int(num) if num else 1
            formulas = [Formula(formula_str) for _ in range(count)]

            if sign == "+":
                adducts_in.extend(formulas)
            else:
                adducts_out.extend(formulas)

        tmp = Adduct(mode=mode, adducts_in=adducts_in, adducts_out=adducts_out)
        charge_diff = charge - tmp.charge

        return Adduct(mode=mode, adducts_in=adducts_in, adducts_out=adducts_out, charge_diff=charge_diff)
    
    @property
    def element_diff(self) -> dict[str, int]:
        """
        Return the total difference in element counts caused by the adduct.
        
        Returns:
            dict[str, int]: Dictionary of element symbol to count (can be negative).
        """
        total: dict[str, int] = defaultdict(int)

        for formula, count in self._adduct_formulas.items():
            for elem, elem_count in formula.elements.items():
                total[elem] += elem_count * count

        return dict(total)