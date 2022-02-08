from MolAligner import Aligner

mol = Aligner("MFI.pdb")
a, b, c = 20.0900, 19.7380, 13.1420
box = [a, b, c]
mol.pbc_replicate(box, multiple=(7, -7, 3))
mol.write("MFI-replicate.gro")
