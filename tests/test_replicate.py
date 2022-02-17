from MolAligner import AlignerX

mol = AlignerX("MFI.pdb")
a, b, c = 20.0900, 19.7380, 13.1420
box = [a, b, c]
mol.set_box(box)
mol.pbc_replicate(multiple=(7, -7, 3))
mol.write("MFI-replicate.gro")
