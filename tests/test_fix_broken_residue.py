from MolAligner import Aligner

mol0 = Aligner("lastFrame.gro")
mol0.move_by(0.5 * mol0.box)
mol0.pbc_wrap()
mol0.write("broken_mol.gro")


# Now fix it
mol1 = Aligner("broken_mol.gro")
mol1.make_whole()
mol1.write("whole_mol.gro")
