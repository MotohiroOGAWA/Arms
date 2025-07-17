from rdkit.Chem import Bond

class BondPosition(tuple):
    """
    BondPosition is a tuple subclass to represent bond indices in a molecule,
    stored in a sorted, unique order to ensure consistent comparison and hashing.
    """

    def __new__(cls, atom_id1:int, atom_id2:int):
        return super().__new__(cls, tuple(sorted((atom_id1, atom_id2))))

    def __repr__(self):
        return f"BondPosition{tuple(self)}"

    def __eq__(self, other):
        if isinstance(other, BondPosition):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == tuple(sorted(other))
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self))
    
    def __reduce__(self):
        return (self.__class__, (self[0], self[1]))
    
    @staticmethod
    def from_bond(bond: Bond) -> 'BondPosition':
        """
        Create a BondPosition from an RDKit Bond object.
        
        Args:
            bond (Bond): The RDKit Bond object.
        
        Returns:
            BondPosition: A new BondPosition instance.
        """
        return BondPosition(
            bond.GetBeginAtom().GetAtomMapNum(), 
            bond.GetEndAtom().GetAtomMapNum())