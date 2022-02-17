from .aligner import Aligner
import numpy as np
from .decorators import check_box_size


class AlignerX(Aligner):
    def __init__(self, cood_file):
        Aligner.__init__(self, cood_file)
        self.residue_tracker = None

    def create_residue_tracker(self):
        residue_tracker = []

        for i in range(self.nAtoms):
            resid = self.resids[i]
            resname = self.resnames[i]

            if len(residue_tracker) == 0:
                tracker = {"id": resid, "name": resname, "nAtoms": 1, "sIDx": i}
                residue_tracker.append(tracker)
            elif (
                residue_tracker[-1]["id"] == resid
                and residue_tracker[-1]["name"] == resname
            ):
                residue_tracker[-1]["nAtoms"] += 1
            else:
                tracker = {"id": resid, "name": resname, "nAtoms": 1, "sIDx": i}
                residue_tracker.append(tracker)

        self.residue_tracker = residue_tracker
        self.nResidues = len(residue_tracker)
        self.residue_summary = {}

        for tracker in residue_tracker:
            if tracker["name"] not in self.residue_summary:
                self.residue_summary[tracker["name"]] = {
                    "n": 1,
                    "nAtoms": tracker["nAtoms"],
                }
            else:
                self.residue_summary[tracker["name"]]["n"] += 1

    def create_molecule_tracker(self, molecule_summary):
        pass

    @check_box_size
    def pbc_wrap(self, lpbc=(True, True, True), origin="lower_left", wrap_type="atom"):
        if wrap_type == "atom":
            self._pbc_wrap_atom(lpbc=lpbc, origin=origin)
        elif wrap_type == "residue":
            self._pbc_wrap_residue(lpbc=lpbc, origin=origin)
        else:
            raise ValueError(r"PBC wrap_type = {wrap_type} not supported!")

    def _pbc_wrap_atom(self, lpbc, origin):

        lx, ly, lz = self.box
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

    def _pbc_wrap_residue(self, lpbc, origin):

        lx, ly, lz = self.box
        if origin == "center":
            xlow, ylow, zlow = (-0.5 * lx, -0.5 * ly, -0.5 * lz)
            xhigh, yhigh, zhigh = (0.5 * lx, 0.5 * ly, 0.5 * lz)
        elif origin == "lower_left":
            xlow, ylow, zlow = (0.0, 0.0, 0.0)
            xhigh, yhigh, zhigh = (lx, ly, lz)
        else:
            raise ValueError(r"Box origin should be at center/lower_left")

        self.create_residue_tracker()
        for i in range(self.nResidues):
            nAtoms = self.residue_tracker[i]["nAtoms"]
            sIDx = self.residue_tracker[i]["sIDx"]
            eIDx = sIDx + nAtoms

            if lpbc[0]:
                if sum(self.x[sIDx:eIDx]) / nAtoms < xlow:
                    self.x[sIDx:eIDx] += lx
                elif sum(self.x[sIDx:eIDx]) / nAtoms > xhigh:
                    self.x[sIDx:eIDx] -= lx

            if lpbc[1]:
                if sum(self.y[sIDx:eIDx]) / nAtoms < ylow:
                    self.x[sIDx:eIDx] += ly
                elif sum(self.x[sIDx:eIDx]) / nAtoms > yhigh:
                    self.y[sIDx:eIDx] -= ly

            if lpbc[2]:
                if sum(self.z[sIDx:eIDx]) / nAtoms < zlow:
                    self.z[sIDx:eIDx] += lz
                elif sum(self.z[sIDx:eIDx]) / nAtoms > zhigh:
                    self.z[sIDx:eIDx] -= lz

    @check_box_size
    def pbc_replicate(self, multiple):
        lx, ly, lz = self.box
        na, nb, nc = multiple
        sign_a, sign_b, sign_c = np.sign(multiple)

        # replicate along x-axis
        collection = []
        for i in range(1, abs(na)):
            other = self.clone()
            trans_vec = [i * lx * sign_a, 0, 0]
            other.move_by(trans_vec)
            collection.append(other)

        for other in collection:
            self.merge(other)

        # replicate along y-axis
        collection = []
        for i in range(1, abs(nb)):
            other = self.clone()
            trans_vec = [0, i * ly * sign_b, 0]
            other.move_by(trans_vec)
            collection.append(other)

        for other in collection:
            self.merge(other)

        # replicate along z-axis
        collection = []
        for i in range(1, abs(nc)):
            other = self.clone()
            trans_vec = [0, 0, i * lz * sign_c]
            other.move_by(trans_vec)
            collection.append(other)

        for other in collection:
            self.merge(other)

        self.box = np.array([abs(na) * lx, abs(nb) * ly, abs(nc) * lz])

    @check_box_size
    def make_whole(self, lpbc=(True, True, True), whole_type="single"):
        if whole_type == "single":
            self._make_whole_single(lpbc=lpbc)
        elif whole_type == "residue":
            self._make_whole_residues(lpbc=lpbc)

    def _make_whole_single(self, lpbc):
        lx, ly, lz = self.box
        hlx, hly, hlz = 0.5 * lx, 0.5 * ly, 0.5 * lz

        xref = self.x[0]
        yref = self.y[0]
        zref = self.z[0]
        for i in range(1, self.nAtoms):
            if lpbc[0]:
                dx = xref - self.x[i]
                if dx > hlx:
                    self.x[i] += lx
                elif dx < -hlx:
                    self.x[i] -= lx

            if lpbc[1]:
                dy = yref - self.y[i]
                if dy > hly:
                    self.y[i] += ly
                elif dy < -hly:
                    self.y[i] -= ly

            if lpbc[2]:
                dz = zref - self.z[i]
                if dz > hlz:
                    self.z[i] += lz
                elif dz < -hlz:
                    self.z[i] -= lz

    def _make_whole_residues(self, lpbc):
        lx, ly, lz = self.box
        hlx, hly, hlz = 0.5 * lx, 0.5 * ly, 0.5 * lz

        self.create_residue_tracker()
        for i in range(self.nResidues):
            nAtoms = self.residue_tracker[i]["nAtoms"]
            sIDx = self.residue_tracker[i]["sIDx"]
            eIDx = sIDx + nAtoms

            xref = self.x[sIDx]
            yref = self.y[sIDx]
            zref = self.z[sIDx]

            for j in range(sIDx + 1, eIDx):

                if lpbc[0]:
                    dx = xref - self.x[j]
                    if dx > hlx:
                        self.x[j] -= lx
                    elif dx < -hlx:
                        self.x[j] += lx

                if lpbc[1]:
                    dy = yref - self.y[j]
                    if dy > hly:
                        self.y[j] -= ly
                    elif dy < -hly:
                        self.y[j] += ly

                if lpbc[2]:
                    dz = zref - self.z[j]
                    if dz > hlz:
                        self.z[j] -= lz
                    elif dz < -hlz:
                        self.z[j] += lz
