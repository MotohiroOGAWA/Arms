from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class Formula:
    def __init__(self, formula_str: str = ""):
        # OrderedDict to preserve Hill order: C, H, then alphabetical
        self.elements: Dict[str, int] = OrderedDict()
        self.charge: int = 0
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
    
    def __hash__(self) -> int:
        return hash((frozenset(self.elements.items()), self.charge))

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

        # Parse element counts
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        temp = {}
        for elem, count in matches:
            temp[elem] = temp.get(elem, 0) + (int(count) if count else 1)

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

        result.charge = self.charge - other.charge
        result._reorder_elements(combined)
        return result
    
    def __eq__(self, other: Formula) -> bool:
        if not isinstance(other, Formula):
            return False
        
        return str(self) == str(other)

    def diff(self, other: Formula) -> str:
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

    def __str__(self) -> str:
        formula = "".join(
            f"{elem}{self.elements[elem] if self.elements[elem] != 1 else ''}"
            for elem in self.elements
        )
        if self.charge > 0:
            return formula + ("+" if self.charge == 1 else f"+{self.charge}")
        elif self.charge < 0:
            return formula + ("-" if self.charge == -1 else f"-{-self.charge}")
        return formula

    def copy(self) -> Formula:
        new = Formula()
        new.elements = self.elements.copy()
        new.charge = self.charge
        return new
    
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Formula:
        """
        Create a Formula object from an RDKit Mol object.
        """
        formula_str = rdMolDescriptors.CalcMolFormula(mol)
        return cls(formula_str)
    

    