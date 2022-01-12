from .molecule import Molecule
import numpy as np
from copy import deepcopy


class Aligner(Molecule):
    def __init__(self, cood_file):
        Molecule.__init__(self, cood_file)

    def move_by(self, trans_vec):
        self.coords += np.array(trans_vec).reshape((3, 1))

    def move_to(self, target_position, target_atom_id=None):

        if target_atom_id:
            xref = self.x[target_atom_id - 1]
            yref = self.y[target_atom_id - 1]
            zref = self.z[target_atom_id - 1]
        else:
            xref = self.x.mean()
            yref = self.y.mean()
            zref = self.z.mean()

        trans_vec = [
            target_position[0] - xref,
            target_position[1] - yref,
            target_position[2] - zref,
        ]
        self.coords += np.array(trans_vec).reshape((3, 1))

    def rotate(self, rotation_mat):

        self.coords = np.matmul(rotation_mat, self.coords)

        # aliasing x,y,z (does not allocate new memory)
        self.x = self.coords[0]
        self.y = self.coords[1]
        self.z = self.coords[2]

    def get_position(self, atom_id):
        return deepcopy(self.coords[:, atom_id - 1])

    def get_vector_between(self, atom_ids):
        i, j = atom_ids
        return self.get_position(j) - self.get_position(i)

    def get_center(self, group_ndx=None):

        if group_ndx is None:
            xcom = self.x.mean()
            ycom = self.y.mean()
            zcom = self.z.mean()
        else:
            xcom = 0.0
            ycom = 0.0
            zcom = 0.0

            for ndx in group_ndx:
                xcom += self.x[ndx]
                ycom += self.y[ndx]
                zcom += self.z[ndx]
            xcom /= len(group_ndx)
            ycom /= len(group_ndx)
            xcom /= len(group_ndx)

        return np.array([xcom, ycom, zcom])

    def get_plane(self, atom_ids):
        i, j, k = atom_ids

        vec1 = self.get_vector_between([j, i])
        vec2 = self.get_vector_between([j, k])
        return np.cross(vec1, vec2)

    @property
    def com(self):
        return self.get_center()
