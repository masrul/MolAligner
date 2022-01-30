from MolAligner import Aligner

ref_mol = Aligner("hsa.xyz")
mol = Aligner("hsa_rotated.xyz")

mol.kabsch_fit(ref_mol)
mol.write("hsa_kabsch.xyz")
