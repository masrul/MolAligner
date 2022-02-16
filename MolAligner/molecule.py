import pathlib
import numpy as np
from .units import nanometer, angstrom
from copy import deepcopy

__all__ = (
    "wrap_around",
    "Molecule",
)


def wrap_around(num, max_num):
    return (num - 1) % max_num + 1


class Molecule:
    def __init__(self, coord_file):
        self.coord_file = coord_file
        self._read()

    def clone(self):
        new_obj = deepcopy(self)
        new_obj.x = new_obj.coords[0]
        new_obj.y = new_obj.coords[1]
        new_obj.z = new_obj.coords[2]

        return new_obj

    def alias_xyz(self):
        # aliasing x,y,z (does not allocate new memory)
        self.x = self.coords[0]
        self.y = self.coords[1]
        self.z = self.coords[2]

    def __add__(self, other):
        self.merge(other)
        return self

    def merge(self, other):
        self.nAtoms += other.nAtoms
        self.symbols += other.symbols
        self.resids += [resid + self.resids[-1] for resid in other.resids]
        self.atomids += [atomid + self.atomids[-1] for atomid in other.atomids]
        self.resnames += other.resnames
        self.coords = np.hstack((self.coords, other.coords))

        # aliasing x,y,z (does not allocate new memory)
        self.x = self.coords[0]
        self.y = self.coords[1]
        self.z = self.coords[2]

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
        self.coords = np.zeros((3, nAtoms), order="C")
        self.symbols = ["C"] * nAtoms
        self.resnames = ["MOL"] * nAtoms
        self.resids = [1] * nAtoms
        self.atomids = [i + 1 for i in range(nAtoms)]
        self.box = np.zeros(3)

        # aliasing x,y,z (does not allocate new memory)
        self.x = self.coords[0]
        self.y = self.coords[1]
        self.z = self.coords[2]

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
                self.x[atom_id] = float(line[30:38]) * angstrom
                self.y[atom_id] = float(line[38:46]) * angstrom
                self.z[atom_id] = float(line[46:54]) * angstrom

                atom_id += 1

    def _read_gro(self):

        with open(self.coord_file, "r") as file_handle:
            lines = file_handle.readlines()

        # read num of atoms
        nAtoms = int(lines[1])
        assert nAtoms > 0
        self._malloc(nAtoms)

        # read coordinates
        atom_id = 0
        for line in lines[2:-1]:
            self.resids[atom_id] = int(line[0:5])
            self.resnames[atom_id] = line[5:10].strip()
            self.symbols[atom_id] = line[10:15].strip()
            self.x[atom_id] = float(line[20:28]) * nanometer
            self.y[atom_id] = float(line[28:36]) * nanometer
            self.z[atom_id] = float(line[36:44]) * nanometer

            atom_id += 1

        sub_strs = lines[-1].split()[0:3]
        self.box[0] = float(sub_strs[0])
        self.box[1] = float(sub_strs[1])
        self.box[2] = float(sub_strs[2])

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
            self.x[atom_id] = float(sub_strs[1]) * angstrom
            self.y[atom_id] = float(sub_strs[2]) * angstrom
            self.z[atom_id] = float(sub_strs[3]) * angstrom

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
            self.x[atom_id] = float(sub_strs[1]) * angstrom
            self.y[atom_id] = float(sub_strs[2]) * angstrom
            self.z[atom_id] = float(sub_strs[3]) * angstrom
            atom_id += 1

    def write(self, out_file, write_mode="w", box=None):
        file_extension = pathlib.Path(out_file).suffix

        if file_extension == ".pdb":
            self._write_pdb(out_file, write_mode)
        elif file_extension == ".xyz":
            self._write_xyz(out_file, write_mode)
        elif file_extension == ".gro":
            self._write_gro(out_file, write_mode, box)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _write_pdb(self, out_file, write_mode):
        file_handler = open(out_file, write_mode)
        max_resid = 1000

        for i in range(self.nAtoms):
            atomid = i + 1
            atomid = wrap_around(atomid,max_atomid)
            resid = wrap_around(self.resids[i], max_resid)
            file_handler.write(
                "%-6s%5d %-4s %3s  %4d    %8.3f%8.3f%8.3f\n"
                % (
                    "ATOM",
                    atomid,
                    self.symbols[i],
                    self.resnames[i],
                    resid,
                    self.x[i],
                    self.y[i],
                    self.z[i],
                )
            )

        file_handler.close()

    def _write_gro(self, out_file, write_mode, box):
        file_handler = open(out_file, write_mode)
        max_atomid = 10000
        max_resid = 10000

        if box is not None:
            lx, ly, lz = box
        else:
            lx, ly, lz = self.box

        file_handler.write("%s\n" % ("Created by MolAligner"))
        file_handler.write("%-10d\n" % (self.nAtoms))

        groFMT = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
        for i in range(self.nAtoms):
            atomid = i + 1
            atomid = wrap_around(atomid,max_atomid)
            resid = wrap_around(self.resids[i], max_resid)
            file_handler.write(
                groFMT
                % (
                    resid,
                    self.resnames[i],
                    self.symbols[i],
                    atomid,
                    self.x[i] / nanometer,
                    self.y[i] / nanometer,
                    self.z[i] / nanometer,
                )
            )
        file_handler.write(
            "%10.5f%10.5f%10.5f\n" % (lx / nanometer, ly / nanometer, lz / nanometer)
        )
        file_handler.close()

    def _write_xyz(self, out_file, write_mode):
        file_handler = open(out_file, "w")

        file_handler.write("%-10d\n" % (self.nAtoms))
        file_handler.write("%s\n" % ("Created by MolAligner"))

        xyzFMT = "%-10s%15.8f%15.8f%15.8f\n"
        for i in range(self.nAtoms):
            file_handler.write(
                xyzFMT % (self.symbols[i], self.x[i], self.y[i], self.z[i],)
            )
