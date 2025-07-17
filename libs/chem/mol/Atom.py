from rdkit import Chem

class Atom:
    def __init__(self, element:str, charged:int, valence:int):
        self.element = element
        self.charged = charged
        self.valence = valence

    @staticmethod
    def is_stable(symbol:str, charged:int, valence:int) -> bool:
        """
        Check if the atom is stable based on its charge and valence.
        """
        # Check if the atom's charge and valence are consistent
        for valid_atom in valid_atoms:
            if symbol == valid_atom.element and charged == valid_atom.charged and valence == valid_atom.valence:
                return True
            
        return False


valid_atoms = [
    # Neutral atoms
    Atom("H", 0, 1),
    Atom("C", 0, 4),
    Atom("N", 0, 3),
    Atom("O", 0, 2),
    Atom("F", 0, 1),
    Atom("Cl", 0, 1),
    Atom("Br", 0, 1),
    Atom("I", 0, 1),

    # Positively charged atoms
    Atom("H", 1, 0),
    Atom("C", 1, 3),
    Atom("N", 1, 4),
    Atom("O", 1, 3),
    Atom("F", 1, 2),
    Atom("Cl", 1, 2),
    Atom("Br", 1, 2),
    Atom("I", 1, 2),

    # Negatively charged atoms
    Atom("H", -1, 0),
    Atom("C", -1, 3),
    Atom("N", -1, 2),
    Atom("O", -1, 1),
    Atom("F", -1, 1),
    Atom("Cl", -1, 1),
    Atom("Br", -1, 1),
    Atom("I", -1, 1),
]