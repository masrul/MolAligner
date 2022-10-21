from .molecule import Molecule
import numpy as np
from copy import deepcopy
from .rotation_matrix import kabsch_rotate


class Aligner(Molecule):
    def __init__(self, cood_file=None, nAtoms=None):
        Molecule.__init__(self, cood_file, nAtoms)

    def move_by(self, trans_vec):
        self.coords += np.array(trans_vec)

    def move_to(self, target_position, target_atom_id=None):
        if target_atom_id:
            ref_pos = self.coords[target_atom_id]
        else:
            ref_pos = np.mean(self.coords, axis=0)

        trans_vec = target_position - ref_pos
        self.coords += trans_vec

    def rotate(self, rot_mat):
        self.coords = np.matmul(self.coords, rot_mat.T)

    def get_position(self, atom_id):
        return deepcopy(self.coords[atom_id])

    def get_vector_between(self, atom_ids):
        i, j = atom_ids
        return self.get_position(j) - self.get_position(i)

    def get_plane(self, atom_ids):
        i, j, k = atom_ids

        vec1 = self.get_vector_between([j, i])
        vec2 = self.get_vector_between([j, k])
        return np.cross(vec1, vec2)

    def kabsch_fit(self, other):
        P = self.coords
        Q = other.coords

        P_com = self.com.reshape((3, 1))
        Q_com = other.com.reshape((3, 1))

        # move com to origin
        P -= P_com
        Q -= Q_com

        # Rotate and translate to Q
        R = kabsch_rotate(P, Q)
        P = np.matmul(R, P) + Q_com.reshape((3, 1))

        # Alias
        self.coords = P

    def get_principal_axis(self):
        # alising to be less verbose
        x = self.coord[:, 0]
        y = self.coord[:, 1]
        z = self.coord[:, 2]

        center = self.com
        self.move_to([0, 0, 0])

        Ixx = np.sum(y * y + z * z)
        Iyy = np.sum(x * x + z * z)
        Izz = np.sum(x * x + y * y)
        Ixy = -np.sum(x * y)
        Ixz = -np.sum(x * z)
        Iyz = -np.sum(y * z)

        self.move_to(center)

        inertia = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

        eigval, eigvec = np.linalg.eig(inertia)

        paxis = eigvec[np.argmax(eigval)]

        return paxis

    def get_center(self, group_ndx=None):
        if group_ndx is None:
            xcom, ycom, zcom = np.mean(self.coords, axis=0)
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

    @property
    def com(self):
        return self.get_center()
