from .molecule import Molecule
import numpy as np
from copy import deepcopy
from .rotation_matrix import kabsch_rotate


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
        self.alias_xyz()

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
        self.alias_xyz()

    @property
    def com(self):
        return self.get_center()

    def get_principal_axis(self):
        center = self.com
        self.move_to([0, 0, 0])

        Ixx = 0.0
        Iyy = 0.0
        Izz = 0.0
        Ixy = 0.0
        Ixz = 0.0
        Iyz = 0.0

        for i in range(self.nAtoms):
            Ixx += self.y[i] * self.y[i] + self.z[i] * self.z[i]
            Iyy += self.x[i] * self.x[i] + self.z[i] * self.z[i]
            Izz += self.x[i] * self.x[i] + self.y[i] * self.y[i]

            Ixy += -(self.x[i] * self.y[i])
            Ixz += -(self.x[i] * self.z[i])
            Iyz += -(self.y[i] * self.z[i])
        self.move_to(center)

        inertia = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

        eigval, eigvec = np.linalg.eig(inertia)

        paxis = eigvec[np.argmax(eigval)]

        return paxis

    def pbc_wrap(self, box, lpbc=(True, True, True), origin="lower_left"):
        lx, ly, lz = box

        if origin == "center":
            xlow, ylow, zlow = (-0.5 * lx, -0.5 * ly, -0.5 * lz)
            xhigh, yhigh, zhigh = (0.5 * lx, 0.5 * ly, 0.5 * lz)
        elif origin == "lower_left":
            xlow, ylow, zlow = (0.0, 0.0, 0.0)
            xhigh, yhigh, zhigh = (lx, ly, lz)
        else:
            raise ValueError(r"Box origin should be at center/lower_left")

        for i in range(self.nAtoms):
            if lpbc[0]:
                if self.x[i] < xlow:
                    self.x[i] += lx
                elif self.x[i] > xhigh:
                    self.x[i] -= lx

            if lpbc[1]:
                if self.y[i] < ylow:
                    self.y[i] += ly
                elif self.y[i] > yhigh:
                    self.y[i] -= ly

            if lpbc[2]:
                if self.z[i] < zlow:
                    self.z[i] += lz
                elif self.z[i] > zhigh:
                    self.z[i] -= lz

    def pbc_replicate(self, box, multiple):
        lx, ly, lz = box
        na, nb, nc = multiple

        # replicate along x-axis
        collection = []
        for i in range(1, abs(na)):
            other = self.clone()
            if na > 0:
                trans_vec = [i * lx, 0, 0]
            else:
                trans_vec = [-i * lx, 0, 0]

            other.move_by(trans_vec)
            collection.append(other)

        for other in collection:
            self.merge(other)

        # replicate along x-axis
        collection = []
        for i in range(1, abs(nb)):
            other = self.clone()
            if nb > 0:
                trans_vec = [0, i * ly, 0]
            else:
                trans_vec = [0, -i * ly, 0]

            other.move_by(trans_vec)
            collection.append(other)

        for other in collection:
            self.merge(other)

        # replicate along x-axis
        collection = []
        for i in range(1, abs(nc)):
            other = self.clone()
            if nc > 0:
                trans_vec = [0, 0, i * lz]
            else:
                trans_vec = [0, 0, -i * lz]
            other.move_by(trans_vec)
            collection.append(other)

        for other in collection:
            self.merge(other)

        self.box = np.array([abs(na) * lx, abs(nb) * ly, abs(nc) * lz])
        self.alias_xyz()
