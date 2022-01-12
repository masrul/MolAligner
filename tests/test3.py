from MolAligner import Aligner
from MolAligner.rotation_matrix import align_two_vectors

mol1 = Aligner("benzene.com")

plane = mol1.get_plane([1, 3, 5])
rot_mat = align_two_vectors(plane, [1, 2, 3])
mol1.rotate(rot_mat)
mol1.move_to([0, 0, 0])


mols = []

for i in range(20):
    mol = mol1.clone()
    mol.move_to([0, 3 * (i + 1), 0])
    mols.append(mol)

for mol in mols:
    mol1.merge(mol)
mol1.write("parallel.pdb")
