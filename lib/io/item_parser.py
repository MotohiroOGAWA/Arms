import re
from typing import Tuple

class ItemParser:
    column_aliases = {
        "Name": [],
        "Formula": [],
        "InChIKey": [],
        "PrecursorMZ": [],
        "AdductType": ["PrecursorType"],
        "SpectrumType": [],
        "InstrumentType": [],
        "IonMode": [],
        "CollisionEnergy": [],
        "ExactMass": [],
        "SMILES": [],
        "Peak": [],
    }

    adduct_type_aliases = {
        "[M]+": [],
        "[M+H]+": [],
        "[M-H]-": [],
        "[M+Na]+": [],
        "[M+K]+": [],
        "[M+NH4]+": [],
    }

    _to_canonical_key = {}
    _to_canonical_adduct_type = {}

    def __init__(self):
        self._initialize()

    @classmethod
    def capitalize(cls, name: str) -> str:
        parts = re.split(r'[_\s]+', name)
        resuslt = ''.join(part.strip().capitalize() for part in parts if part)
        return resuslt

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        capitalized_key = cls.capitalize(key)
        return capitalized_key.replace("/", "").lower()

    @classmethod
    def _normalize_adduct_type(cls, value: str) -> str:
        core_match = re.match(r"\[(.*?)\]", value)
        
        if not core_match:
            return value.replace(" ", "").strip()
        
        core = core_match.group(1)

        charge_match = re.search(r"\](\d*[+-]+)$", value)
        if not charge_match:
            charge_match = re.search(r"\](.*)$", value)
            charge = charge_match.group(1).strip()
        else:
            raw = charge_match.group(1).strip()

            try:
                # Exapmple: "++" → +2, "--" → -2
                if all(c == "+" for c in raw):
                    charge = len(raw)
                elif all(c == "-" for c in raw):
                    charge = -len(raw)
                elif raw[-1] == "+":
                    charge = int(raw[:-1]) if raw[:-1].isdigit() else 1
                elif raw[-1] == "-":
                    charge = -int(raw[:-1]) if raw[:-1].isdigit() else -1
                else:
                    charge = None
            except ValueError:
                charge = None
            
            if charge is None:
                charge = raw
            elif charge == 0:
                charge = ''
            elif charge == 1:
                charge = '+'
            elif charge == -1:
                charge = '-'
            elif charge > 1:
                charge = str(charge) + '+'
            elif charge < -1:
                charge = str(-charge) + '-'
            else:
                charge = raw

        normal_name = f"[{core}]{charge}"
            
        return normal_name.replace(" ", "").strip()

    @classmethod
    def _initialize(cls):
        """
        Initialize the class by creating
        a mapping of canonical keys to their aliases.
        """
        # Create a mapping of canonical keys to their aliases
        for canonical_key, aliases in cls.column_aliases.items():
            normalized = cls._normalize_key(canonical_key)
            cls._to_canonical_key[normalized] = canonical_key
            for alias in aliases:
                normalized_alias = cls._normalize_key(alias)
                cls._to_canonical_key.setdefault(normalized_alias, canonical_key)

        # Create a mapping of canonical adduct types to their aliases
        for canonical_adduct, aliases in cls.adduct_type_aliases.items():
            normalized = cls._normalize_adduct_type(canonical_adduct)
            cls._to_canonical_adduct_type[normalized] = canonical_adduct
            for alias in aliases:
                normalized_alias = cls._normalize_adduct_type(alias)
                cls._to_canonical_adduct_type.setdefault(normalized_alias, canonical_adduct)

    @classmethod
    def to_canonical_key(cls, name: str) -> str:
        normalized_name = cls._normalize_key(name)
        if normalized_name not in cls._to_canonical_key:
            capitalized_name = cls.capitalize(name)
            cls._to_canonical_key[normalized_name] = capitalized_name

        return cls._to_canonical_key[normalized_name]

    @classmethod
    def to_canonical_adduct_type(cls, adduct_type: str) -> str:
        normalized_adduct_type = cls._normalize_adduct_type(adduct_type)
        if normalized_adduct_type not in cls._to_canonical_adduct_type:
            cls._to_canonical_adduct_type[normalized_adduct_type] = normalized_adduct_type
        
        return cls._to_canonical_adduct_type[normalized_adduct_type]

    @classmethod
    def parse(cls, line: str) -> Tuple[str, str]:
        if ":" not in line:
            raise ValueError(f"Line format is invalid: '{line}'")

        parts = line.split(":", 1)
        key = parts[0].strip()
        value = parts[1].strip()

        canonical_key = cls.to_canonical_key(key)
        if canonical_key == "AdductType":
            value = cls.to_canonical_adduct_type(value)

        return canonical_key, value

