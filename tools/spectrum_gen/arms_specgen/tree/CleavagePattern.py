from rdkit import Chem
from rdkit.Chem import rdmolops
from typing import List, Tuple

class CleavagePattern:
    def __init__(self, name: str, smarts: str, fragments: List[str]):
        """
        Represent a cleavage pattern.

        Args:
            name (str): Name of the cleavage pattern (e.g. "amide bond cleavage").
            smarts (str): SMARTS describing the bond or substructure to cleave.
            fragments (List[str]): Expected fragment templates (SMILES/SMARTS).
        """
        self.name = name
        self.smarts = smarts
        self.fragments = fragments
        self.query = Chem.MolFromSmarts(smarts)

    def match(self, mol: Chem.Mol) -> List[Tuple[int]]:
        """
        Find matches of this cleavage pattern in a molecule.

        Args:
            mol (Chem.Mol): Input molecule.

        Returns:
            List[Tuple[int]]: Atom index tuples where cleavage can occur.
        """
        return mol.GetSubstructMatches(self.query)

    def fragment(self, mol: Chem.Mol) -> List[str]:
        """
        Perform fragmentation on the molecule based on the cleavage pattern.
        Returns SMILES strings of the fragments.

        Args:
            mol (Chem.Mol): Input molecule.

        Returns:
            List[str]: List of SMILES for the fragments.
        """
        matches = self.match(mol)
        if not matches:
            return []

        all_fragments = []
        for match in matches:
            bonds_to_cut = []
            # Identify bonds between consecutive atoms in the SMARTS match
            for i, j in zip(match, match[1:]):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond:
                    bonds_to_cut.append(bond.GetIdx())

            if not bonds_to_cut:
                continue

            # Cut the bonds and add dummy atoms at cut sites
            frag_mol = Chem.FragmentOnBonds(mol, bonds_to_cut, addDummies=True)

            # Get separate fragment molecules
            parts = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)

            # Convert to SMILES
            parts_smiles = [Chem.MolToSmiles(f, canonical=True) for f in parts]

            # If templates are defined, reorder or filter to match the rule
            if self.fragments:
                # Try to align with expected fragments (very simple matching)
                ordered = []
                for templ in self.fragments:
                    for smi in parts_smiles:
                        if smi.count("*") == templ.count("*"):  # crude matching
                            ordered.append(smi)
                            break
                all_fragments.extend(ordered)
            else:
                all_fragments.extend(parts_smiles)

        return all_fragments

    def __repr__(self):
        return f"CleavagePattern(name={self.name}, smarts={self.smarts}, fragments={self.fragments})"


if __name__ == "__main__":
    # Define a cleavage rule for an amide bond
    amide_cleavage = CleavagePattern(
        name="Amide bond cleavage",
        smarts="C(=O)N",
        fragments=["C(=O)[*]", "N[*]"]  # Expected templates
    )

    mol = Chem.MolFromSmiles("CC(=O)NC")  # Acetamide
    print("Original:", Chem.MolToSmiles(mol))

    frags = amide_cleavage.fragment(mol)
    for i, frag in enumerate(frags):
        print(f"Fragment {i+1}: {frag}")

