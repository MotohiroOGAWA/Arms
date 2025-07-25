from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ..utilities.formula_utils import enumerate_possible_sub_formulas

class Formula:
    def __init__(self, formula_str: str = ""):
        # OrderedDict to preserve Hill order: C, H, then alphabetical
        self.elements: Dict[str, int] = OrderedDict()
        self.charge: int = 0
        self.raw_formula: str = ""

        if formula_str:
            self._parse_formula(formula_str)

    @property
    def exact_mass(self) -> float:
        """
        Calculate the exact mass of the formula.
        """
        mass = 0.0
        for elem, count in self.elements.items():
            atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(elem)
            mass += Chem.GetPeriodicTable().GetMostCommonIsotopeMass(atomic_number) * count
        return mass

    def __repr__(self):
        return f"Formula({self.__str__()})"
    

    def __str__(self) -> str:
        return self.to_string(no_charge=False)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.elements.items()), self.raw_formula, self.charge))

    def _parse_formula(self, formula: str):
        """
        Parse chemical formula and set element order according to Hill system.
        """
        # Extract and remove charge
        charge_match = re.search(r"([+-]+|[+-]\d+)$", formula)
        if charge_match:
            charge_str = charge_match.group(1)
            formula = formula[: -len(charge_str)]
            self.charge = int(charge_str[1:]) if charge_str[1:] else 1
            if charge_str[0] == '-':
                self.charge *= -1

        self.raw_formula = formula  # Store the raw formula for reference

        # Parse element counts
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        temp = {}
        for elem, count in matches:
            temp[elem] = temp.get(elem, 0) + (int(count) if count else 1)
        temp = {k: v for k, v in temp.items() if v != 0}

        # Determine Hill order
        keys = temp.keys()
        ordered = Formula._reorder_element_keys(keys)

        # Store in ordered dict
        self.elements = OrderedDict((k, temp[k]) for k in ordered)

    def __add__(self, other: Formula) -> Formula:
        result = Formula()
        combined = dict(self.elements)

        for elem, count in other.elements.items():
            combined[elem] = combined.get(elem, 0) + count

        result.charge = self.charge + other.charge
        result._reorder_elements(combined)
        return result

    def __sub__(self, other: Formula) -> Formula:
        result = Formula()
        combined = dict(self.elements)

        for elem, count in other.elements.items():
            combined[elem] = combined.get(elem, 0) - count
            if combined[elem] < 0:
                raise ValueError(f"Cannot subtract {other} from {self}: negative element count for {elem}")

        result.charge = self.charge - other.charge
        result._reorder_elements(combined)
        return result
    
    def __eq__(self, other: Formula) -> bool:
        if not isinstance(other, Formula):
            return False
        
        return str(self) == str(other)
    
    @property
    def value(self) -> str:
        """
        Return the formula as a string with charge.
        """
        return self.to_string(no_charge=False)
    
    @property
    def plain(self) -> str:
        """
        Return the formula as a plain string without charge.
        """
        return self.to_string(no_charge=True)

    def diff(self, other: Formula) -> tuple[OrderedDict[str, int], int]:
        """
        Return a human-readable string showing the difference between self and other.
        Example: +H2-C
        """
        parts = []

        # Merge all element keys
        all_elements = set(self.elements) | set(other.elements)
        elements = OrderedDict()
        for elem in sorted(all_elements):
            count_self = self.elements.get(elem, 0)
            count_other = other.elements.get(elem, 0)
            diff = count_self - count_other
            elements[elem] = diff

        elements = self._reorder_element_keys(elements.keys())
        element_diff = OrderedDict()
        for elem in elements:
            diff = self.elements[elem] - other.elements.get(elem, 0)
            if diff > 0:
                parts.append(f"+{elem}{diff if diff != 1 else ''}")
            elif diff < 0:
                parts.append(f"-{elem}{-diff if diff != -1 else ''}")
            if diff != 0:
                element_diff[elem] = diff


        # Charge difference
        charge_diff = self.charge - other.charge
        # if charge_diff > 0:
        #     parts.append(f"+{charge_diff}" if charge_diff != 1 else "+")
        # elif charge_diff < 0:
        #     parts.append(f"{charge_diff}" if charge_diff != -1 else "-")

        return element_diff, charge_diff


    def _reorder_elements(self, element_counts: Dict[str, int]):
        """Apply Hill system ordering to elements and store as OrderedDict."""
        keys = element_counts.keys()
        ordered = Formula._reorder_element_keys(keys)

        self.elements = OrderedDict((k, element_counts[k]) for k in ordered if element_counts[k] != 0)

    @staticmethod
    def _reorder_element_keys(elements: list[str]) -> OrderedDict:
        """
        Reorder elements according to Hill system.
        """
        mol = Chem.RWMol()
        for elem in elements:
            atom = Chem.Atom(elem)
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
        mol = mol.GetMol()

        formula_str = rdMolDescriptors.CalcMolFormula(mol)
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)

        ordered = tuple(m[0] for m in matches)
        return ordered

    def to_string(self, no_charge: bool = False) -> str:
        """
        Return the formula as a string. If no_charge=True, omit the charge.
        """
        formula = "".join(
            f"{elem}{self.elements[elem] if self.elements[elem] != 1 else ''}"
            for elem in self.elements
        )

        if no_charge:
            return formula

        if self.charge > 0:
            return formula + ("+" if self.charge == 1 else f"+{self.charge}")
        elif self.charge < 0:
            return formula + ("-" if self.charge == -1 else f"-{-self.charge}")
        
        return formula

    @staticmethod
    def from_elements(elements: Dict[str, int], charge: int = 0) -> Formula:
        """
        Create a Formula object from a dictionary of elements and their counts.
        """
        formula = Formula()
        elements = {k: v for k, v in elements.items() if v != 0}

        # Determine Hill order
        keys = elements.keys()
        ordered = Formula._reorder_element_keys(keys)

        # Store in ordered dict
        formula.elements = OrderedDict((k, elements[k]) for k in ordered)
        formula.charge = charge

        return formula

    def copy(self) -> Formula:
        return Formula.from_elements(self.elements.copy(), self.charge)
    
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Formula:
        """
        Create a Formula object from an RDKit Mol object.
        """
        formula_str = rdMolDescriptors.CalcMolFormula(mol)
        return cls(formula_str)
    
    def get_possible_sub_formulas(self, hydrogen_delta: int = 0) -> Dict[str, float]:
        """
        Generate possible sub-formulas with their degree of unsaturation.
        Returns a dictionary of sub-formula strings and their DBE values.
        """
        # Add hydrogen delta to original count
        elements = self.elements.copy()
        elements["H"] = elements.get("H", 0) + hydrogen_delta
        elements["H"] = max(elements["H"], 0)  # prevent negative H count

        res = [
            Formula.from_elements(elements)
            for elements, dbe in enumerate_possible_sub_formulas(self.elements)
        ]

        res = sorted(res, key=lambda f: (f.exact_mass, f.plain))

        return res
    


    