import pathlib
import numpy as np
from .units import nanometer, angstrom
from copy import deepcopy


def wrap_around(num, max_num):
    return (num - 1) % max_num + 1


class Molecule:
    def __init__(self, coord_file=None, nAtoms=None):
        if coord_file is not None:
            self.coord_file = coord_file
            self._read()
        elif coord_file is None and nAtoms is not None:
            self._malloc(nAtoms)
        else:
            raise ValueError("Can not Instantiate the object")

    def clone(self):
        return deepcopy(self)

    def __add__(self, other):
        self.merge(other)
        return self

    def merge(self, other):
        self.nAtoms += other.nAtoms
        self.symbols += other.symbols
        self.resids += [resid + self.resids[-1] for resid in other.resids]
        self.atomids += [atomid + self.atomids[-1] for atomid in other.atomids]
        self.resnames += other.resnames
        self.coords = np.vstack((self.coords, other.coords))  # coord

    def _read(self):
        file_extension = pathlib.Path(self.coord_file).suffix

        if file_extension == ".pdb":
            self._read_pdb()
        elif file_extension == ".xyz":
            self._read_xyz()
        elif file_extension == ".gro":
            self._read_gro()
        elif file_extension == ".com":
            self._read_com()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _malloc(self, nAtoms):
        self.nAtoms = nAtoms
        self.coords = np.zeros((nAtoms, 3), order="C")
        self.symbols = ["C"] * nAtoms
        self.resnames = ["MOL"] * nAtoms
        self.resids = [1] * nAtoms
        self.atomids = [i + 1 for i in range(nAtoms)]
        self.box = np.zeros(3)

    def set_box(self, box):
        self.box = np.array(box)

    def _read_pdb(self):

        with open(self.coord_file, "r") as file_handle:
            lines = file_handle.readlines()

        # find number of atoms
        nAtoms = 0
        for line in lines:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                nAtoms += 1
        assert nAtoms > 0
        self._malloc(nAtoms)

        # read coordinates
        atom_id = 0
        for line in lines:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                self.symbols[atom_id] = line[12:16].strip()
                self.resnames[atom_id] = line[17:20].strip()
                self.resids[atom_id] = int(line[22:26])
                self.coords[atom_id, 0] = float(line[30:38]) * angstrom  # coord
                self.coords[atom_id, 1] = float(line[38:46]) * angstrom
                self.coords[atom_id, 2] = float(line[46:54]) * angstrom

                atom_id += 1

    def _read_gro(self):

        with open(self.coord_file, "r") as file_handle:
            file_handle.readline()  # skip blank/comment line

            # save lines, looping as it might be trajectory
            nAtoms = int(file_handle.readline())
            lines = []
            for _ in range(nAtoms + 1):
                lines.append(file_handle.readline())

        # read num of atoms
        assert nAtoms > 0
        self._malloc(nAtoms)

        # read coordinates
        atom_id = 0
        for line in lines[:-1]:
            self.resids[atom_id] = int(line[0:5])
            self.resnames[atom_id] = line[5:10].strip()
            self.symbols[atom_id] = line[10:15].strip()
            self.coords[atom_id, 0] = float(line[20:28]) * nanometer  # coord
            self.coords[atom_id, 1] = float(line[28:36]) * nanometer
            self.coords[atom_id, 2] = float(line[36:44]) * nanometer

            atom_id += 1

        sub_strs = lines[-1].split()[0:3]
        self.box[0] = float(sub_strs[0]) * nanometer
        self.box[1] = float(sub_strs[1]) * nanometer
        self.box[2] = float(sub_strs[2]) * nanometer

    def _read_xyz(self):
        with open(self.coord_file, "r") as file_handle:
            lines = file_handle.readlines()

        # read num of atoms
        nAtoms = int(lines[0])
        assert nAtoms > 0
        self._malloc(nAtoms)

        # read coordinates
        atom_id = 0
        for line in lines[2::]:
            sub_strs = line.split()
            self.symbols[atom_id] = sub_strs[0]
            self.coords[atom_id, 0] = float(sub_strs[1]) * angstrom
            self.coords[atom_id, 1] = float(sub_strs[2]) * angstrom
            self.coords[atom_id, 2] = float(sub_strs[3]) * angstrom

            atom_id += 1

    def _read_com(self):
        with open(self.coord_file, "r") as file_handle:
            lines = file_handle.readlines()

        # read num of atoms
        lineID = -1
        skip_char = ["%", "#"]
        for line in lines:
            if line.strip() == "":
                break
            elif line.strip()[0] in skip_char:
                lineID += 1
        coord_start = lineID + 5
        nAtoms = 0
        for line in lines[coord_start:]:
            if line.strip() == "":
                break
            else:
                nAtoms += 1

        assert nAtoms > 0
        self._malloc(nAtoms)

        # read  coordinates
        coord_end = coord_start + nAtoms
        atom_id = 0
        for line in lines[coord_start:coord_end]:
            sub_strs = line.split()
            self.symbols[atom_id] = sub_strs[0]
            self.coords[atom_id, 0] = float(sub_strs[1]) * angstrom
            self.coords[atom_id, 1] = float(sub_strs[2]) * angstrom
            self.coords[atom_id, 2] = float(sub_strs[3]) * angstrom
            atom_id += 1

    def write(self, out_file, write_mode="w"):
        file_extension = pathlib.Path(out_file).suffix

        if file_extension == ".pdb":
            self._write_pdb(out_file, write_mode)
        elif file_extension == ".xyz":
            self._write_xyz(out_file, write_mode)
        elif file_extension == ".gro":
            self._write_gro(out_file, write_mode)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _write_pdb(self, out_file, write_mode):
        file_handler = open(out_file, write_mode)
        max_resid = 9999
        max_atomid = 99999

        for i in range(self.nAtoms):
            atomid = i + 1
            atomid = wrap_around(atomid, max_atomid)
            resid = wrap_around(self.resids[i], max_resid)
            file_handler.write(
                "%-6s%5d %-4s %3s  %4d    %8.3f%8.3f%8.3f\n"
                % (
                    "ATOM",
                    atomid,
                    self.symbols[i],
                    self.resnames[i],
                    resid,
                    self.coords[i, 0],
                    self.coords[i, 1],
                    self.coords[i, 2],
                )
            )

        file_handler.close()

    def _write_gro(self, out_file, write_mode):
        file_handler = open(out_file, write_mode)
        max_atomid = 99999
        max_resid = 99999

        file_handler.write("%s\n" % ("Created by MolAligner"))
        file_handler.write("%-10d\n" % (self.nAtoms))

        groFMT = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
        for i in range(self.nAtoms):
            atomid = i + 1
            atomid = wrap_around(atomid, max_atomid)
            resid = wrap_around(self.resids[i], max_resid)
            file_handler.write(
                groFMT
                % (
                    resid,
                    self.resnames[i],
                    self.symbols[i],
                    atomid,
                    self.coords[i, 0] / nanometer,
                    self.coords[i, 1] / nanometer,
                    self.coords[i, 2] / nanometer,
                )
            )
        file_handler.write(
            "%10.5f%10.5f%10.5f\n"
            % (
                self.box[0] / nanometer,
                self.box[1] / nanometer,
                self.box[2] / nanometer,
            )
        )
        file_handler.close()

    def _write_xyz(self, out_file, write_mode):
        file_handler = open(out_file, "w")

        file_handler.write("%-10d\n" % (self.nAtoms))
        file_handler.write("%s\n" % ("Created by MolAligner"))

        xyzFMT = "%-10s%15.8f%15.8f%15.8f\n"
        for i in range(self.nAtoms):
            file_handler.write(
                xyzFMT
                % (
                    self.symbols[i],
                    self.coords[i, 0],
                    self.coords[i, 1],
                    self.coords[i, 2],
                )
            )

    def keep(self, index, inplace=True):

        other = self._filter(index)

        if inplace:
            self = other
        else:
            return other

    def discard(self, index, inplace=True):

        keep_index = list(set(self.nAtoms) ^ set(index))
        keep_index.sort()

        other = self._filter(index)
        if inplace:
            self = other
        else:
            return other

    def _filter(self, index):
        other = Molecule(len(index))

        for i, idx in enumerate(index):
            other.symbols[i] = self.symbols[idx]
            other.resnames[i] = self.resnames[idx]
            other.resids[i] = self.resids[idx]
            other.atomids[i] = i + 1

            for j in range(3):
                other.coords[j][i] = self.coords[j][idx]

        for i in range(3):
            other.box[i] = self.box[i]

        return other

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    @property
    def z(self):
        return self.coords[:, 2]
