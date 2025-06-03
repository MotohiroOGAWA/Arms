from itertools import product
from typing import Generator


def calculate_dbe(elements: dict[str, int]) -> float:
    """
    Calculate the double bond equivalent (DBE) from an element count dictionary.
    Only common organic elements are supported (C, H, N, O, P, S, F, Cl, Br, I).

    Args:
        elements (dict): A dictionary of element counts.

    Returns:
        float: The calculated DBE.

    Raises:
        AssertionError: If unsupported elements are present.
    """
    allowed_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
    unsupported = set(elements.keys()) - allowed_elements
    assert not unsupported, f"Unsupported elements in formula: {unsupported}"

    C = elements.get('C', 0)
    H = elements.get('H', 0)
    O = elements.get('O', 0)
    N = elements.get('N', 0)
    P = elements.get('P', 0)
    S = elements.get('S', 0)
    X = sum([elements.get(x, 0) for x in ['F', 'Cl', 'Br', 'I']])  # Halogens
    
    dbe = (2*C + N - H - X + 2) / 2.0
    return dbe

def enumerate_possible_sub_formulas(elements: dict[str, int]) -> Generator[tuple[dict[str, int], float], None, None]:
    base_elements = [(elem, count) for elem, count in elements.items() if elem != "H"]
    max_h = elements.get("H", 0)
    
    # Create all combinations from 0 to the original count for each element (excluding H)
    ranges = [range(c + 1) for _, c in base_elements]

    for counts in product(*ranges):
        # Generate element count combinations
        temp_counts = {elem: count for (elem, _), count in zip(base_elements, counts)}
        
        # Try hydrogen counts from 0 to max_h
        for h in range(max_h + 1):
            dbe = calculate_dbe(temp_counts | {"H": h})
            if dbe < 0:
                continue  # Skip if degree of unsaturation is negative

            temp_counts["H"] = h
            if sum(temp_counts.values()) == 0:
                continue
            
            yield temp_counts.copy(), dbe