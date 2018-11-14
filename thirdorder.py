#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2018 Wu Li <wu.li.phys2011@gmail.com>
#  Copyright (C) 2012-2018 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2012-2018 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#  Copyright (C) 2014-2018 Antti J. Karttunen <antti.j.karttunen@iki.fi>
#  Copyright (C) 2016-2018 Genadi Naydenov <gan503@york.ac.uk>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range

import re
import ast

import thirdorder_core
from thirdorder_common import *

# Conversion factors (source: CODATA 2010)
BOHR_RADIUS = 5.2917721092e-2  # nm
RYDBERG = 13.60569253  # eV


def qe_cell(ibrav, celldm):
    """
    Return a set of lattice vectors according to Quantum Espresso's
    convention. ibrav=0 is not supported by this function.
    """
    nruter = np.zeros((3, 3))
    if ibrav == 1:
        nruter = np.eye(3)
    elif ibrav == 2:
        nruter[0, 0] = -0.5
        nruter[0, 1] = 0.0
        nruter[0, 2] = 0.5
        nruter[1, 0] = 0.0
        nruter[1, 1] = 0.5
        nruter[1, 2] = 0.5
        nruter[2, 0] = -0.5
        nruter[2, 1] = 0.5
        nruter[2, 2] = 0.0
    elif ibrav == 3:
        nruter[0, 0] = 0.5
        nruter[0, 1] = 0.5
        nruter[0, 2] = 0.5
        nruter[1, 0] = -0.5
        nruter[1, 1] = 0.5
        nruter[1, 2] = 0.5
        nruter[2, 0] = -0.5
        nruter[2, 1] = -0.5
        nruter[2, 2] = 0.5
    elif ibrav == 4:
        nruter[0, 0] = 1.0
        nruter[0, 1] = 0.0
        nruter[0, 2] = 0.0
        nruter[1, 0] = -0.5
        nruter[1, 1] = np.sqrt(3.) / 2.
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.
        nruter[2, 1] = 0.
        nruter[2, 2] = celldm[3]
    elif ibrav == 5:
        nruter[0, 0] = np.sqrt((1 - celldm[4]) / 2.)
        nruter[0, 1] = -np.sqrt((1 - celldm[4]) / 6.)
        nruter[0, 2] = np.sqrt((1 + 2 * celldm[4]) / 3.)
        nruter[1, 0] = 0.
        nruter[1, 1] = 2. * np.sqrt((1 - celldm[4]) / 6.)
        nruter[1, 2] = np.sqrt((1 + 2 * celldm[4]) / 3.)
        nruter[2, 0] = -np.sqrt((1 - celldm[4]) / 2.)
        nruter[2, 1] = -np.sqrt((1 - celldm[4]) / 6.)
        nruter[2, 2] = np.sqrt((1 + 2 * celldm[4]) / 3.)
    elif ibrav == 6:
        nruter[0, 0] = 1.0
        nruter[0, 1] = 0.0
        nruter[0, 2] = 0.0
        nruter[1, 0] = 0.0
        nruter[1, 1] = 1.0
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.
        nruter[2, 1] = 0.
        nruter[2, 2] = celldm[3]
    elif ibrav == 7:
        nruter[0, 0] = 0.5
        nruter[0, 1] = -0.5
        nruter[0, 2] = celldm[3]
        nruter[1, 0] = 0.5
        nruter[1, 1] = 0.5
        nruter[1, 2] = celldm[3]
        nruter[2, 0] = -0.5
        nruter[2, 1] = -0.5
        nruter[2, 2] = celldm[3]
    elif ibrav == 8:
        nruter[0, 0] = 1.0
        nruter[0, 1] = 0.0
        nruter[0, 2] = 0.0
        nruter[1, 0] = 0.0
        nruter[1, 1] = celldm[2]
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.
        nruter[2, 1] = 0.
        nruter[2, 2] = celldm[3]
    elif ibrav == 9:
        nruter[0, 0] = 0.5
        nruter[0, 1] = celldm[2] / 2.
        nruter[0, 2] = 0.0
        nruter[1, 0] = -0.5
        nruter[1, 1] = celldm[2] / 2.
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.
        nruter[2, 1] = 0.
        nruter[2, 2] = celldm[3]
    elif ibrav == 10:
        nruter[0, 0] = 0.5
        nruter[0, 1] = 0.0
        nruter[0, 2] = celldm[3] / 2.
        nruter[1, 0] = 0.5
        nruter[1, 1] = celldm[2] / 2.
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.
        nruter[2, 1] = celldm[2] / 2.
        nruter[2, 2] = celldm[3] / 2.
    elif ibrav == 11:
        nruter[0, 0] = 0.5
        nruter[0, 1] = celldm[2] / 2.
        nruter[0, 2] = celldm[3] / 2.
        nruter[1, 0] = -0.5
        nruter[1, 1] = celldm[2] / 2.
        nruter[1, 2] = celldm[3] / 2.
        nruter[2, 0] = -0.5
        nruter[2, 1] = -celldm[2] / 2.
        nruter[2, 2] = celldm[3] / 2.
    elif ibrav == 12:
        nruter[0, 0] = 1.0
        nruter[0, 1] = 0.0
        nruter[0, 2] = 0.0
        nruter[1, 0] = celldm[2] * celldm[4]
        nruter[1, 1] = celldm[2] * np.sqrt(1 - celldm[4]**2)
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.
        nruter[2, 1] = 0.
        nruter[2, 2] = celldm[3]
    elif ibrav == 13:
        nruter[0, 0] = 0.5
        nruter[0, 1] = 0.0
        nruter[0, 2] = -celldm[3] / 2.
        nruter[1, 0] = celldm[2] * celldm[4]
        nruter[1, 1] = celldm[2] * np.sqrt(1 - celldm[4]**2)
        nruter[1, 2] = 0.
        nruter[2, 0] = 0.5
        nruter[2, 1] = 0.
        nruter[2, 2] = celldm[3] / 2.
    elif ibrav == 14:
        nruter[0, 0] = 1.0
        nruter[0, 1] = 0.0
        nruter[0, 2] = 0.0
        nruter[1, 0] = celldm[2] * celldm[6]
        nruter[1, 1] = celldm[2] * np.sin(np.arccos(celldm[6]))
        nruter[1, 2] = 0.
        nruter[2, 0] = celldm[3] * celldm[5]
        nruter[2, 1] = celldm[3] * (
            celldm[4] - celldm[5] * celldm[6]) / np.sin(np.arccos(celldm[6]))
        nruter[2, 2] = celldm[3] * np.sqrt(
            1 + 2 * celldm[4] * celldm[5] * celldm[6] - celldm[4]**2 -
            celldm[5]**2 - celldm[6]**2) / np.sin(np.arccos(celldm[6]))
    else:
        raise ValueError("unknown ibrav")
    return nruter


def eval_qe_algebraic(expression):
    """
    Return the value of an algebraic expression of
    the kind allowed by Quantum Espresso for coordinates.
    """
    # Perform basic checks on the expression.
    if len(expression) == 0:
        raise ValueError("empty expression")
    validchars = "0123456789.eEdD+-*/^()"
    for i in expression:
        if i not in validchars:
            raise ValueError(
                "invalid character \"{0}\" in algebraic expression".format(i))
    if expression[0] == "+":
        raise ValueError("expression starts with +")
    # Translate the exponential notantion and the power operator into Python.
    expr = expression.lower().replace("d", "e").replace("^", "**")

    # Evaluate the result in a safe manner.
    if expr.startswith("-"):
        prefactor = -1
        expr = expr[1:]
    else:
        prefactor = 1.

    def eval_node(node):
        """
        Evaluate each node in the expression, recursing down if needed.
        """
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        elif isinstance(node, ast.Num):
            return float(node.n)
        elif isinstance(node, ast.BinOp):
            if type(node.op) == ast.Add:
                return eval_node(node.left) + eval_node(node.right)
            elif type(node.op) == ast.Sub:
                return eval_node(node.left) - eval_node(node.right)
            elif type(node.op) == ast.Mult:
                return eval_node(node.left) * eval_node(node.right)
            elif type(node.op) == ast.Div:
                return eval_node(node.left) / eval_node(node.right)
            elif type(node.op) == ast.Pow:
                return eval_node(node.left)**eval_node(node.right)
            else:
                raise ValueError("invalid binary operator")
        else:
            raise ValueError("invalid node in the parse tree")

    return prefactor * eval_node(ast.parse(expr, mode="eval").body)


def read_qe_in(filename):
    """
    Return all the relevant information about the system from a QE
    input file.
    """
    celldmre = re.compile(
        r"celldm\((?P<number>\d)\)\s*=\s*(?P<value>\S+?)(?:$|[,!\s])",
        re.MULTILINE)
    tagre = lambda keyword:re.compile(
        re.escape(keyword) +
        r"\s*=\s*(?P<value>\S+?)(?:$|[,!\s])", re.MULTILINE)
    kindre = re.compile(r"\S+\s+[\{\(\s]*(?P<kind>\w+)[\}\)\s]*")
    contents = open(filename, "r").read()
    try:
        ibrav = int(tagre("ibrav").search(contents).group("value"))
    except TypeError:
        sys.exit("Error: could not find the ibrav tag")
    try:
        natoms = int(tagre("nat").search(contents).group("value"))
    except TypeError:
        sys.exit("Error: could not find the nat tag")
    try:
        nelements = int(tagre("ntyp").search(contents).group("value"))
    except TypeError:
        sys.exit("Error: could not find the ntyp tag")
    celldm = dict()
    for m in celldmre.finditer(contents):
        res = m.groupdict()
        celldm[int(res["number"])] = float(res["value"])
    nruter = dict()
    if len(celldm) > 0:
        # celldm is not required for ibrav==0
        # (except for CELL_PARAMETERS alat, for which it's checked below)
        celldm[1] *= BOHR_RADIUS
    if ibrav == 0:
        # CELL_PARAMETERS are read in below after ATOMIC_POSITIONS
        nruter["lattvec"] = np.empty((3, 3))
    else:
        nruter["lattvec"] = qe_cell(ibrav, celldm).T * celldm[1]
    nruter["positions"] = np.empty((3, natoms))
    nruter["elements"] = []
    lines = contents.split("\n")
    # Read ATOMIC_POSITIONS
    reading = False
    read = 0
    for l in lines:
        if reading:
            fields = l.split()
            nruter["elements"].append(fields[0])
            nruter["positions"][:, read] = [
                eval_qe_algebraic(i) for i in fields[1:4]
            ]
            read = read + 1
            if read == natoms:
                break
        if l.startswith("ATOMIC_POSITIONS"):
            try:
                poskind = kindre.search(l).group("kind")
            except AttributeError:
                raise ValueError("Type of ATOMIC_POSITIONS missing")
            if poskind not in ("alat", "bohr", "angstrom", "crystal"):
                raise ValueError(
                    "cannot interpret coordinates in \"{0}\" format"
                    .format(poskind))
            reading = True
    # Sanity check
    if read < natoms:
        raise ValueError(
            "Proper ATOMIC_POSITIONS not found (expected: {0}; found: {1})"
            .format(natoms, read))
    # Read CELL_PARAMETERS if ibrav == 0
    reading = False
    read = 0
    if ibrav == 0:
        for l in lines:
            if reading:
                fields = l.split()
                nruter["lattvec"][:, read] = [float(i) for i in fields[0:3]]
                read = read + 1
                if read == 3:
                    # Convert lattvec to nm units
                    if latkind == "alat":
                        nruter["lattvec"] *= celldm[1]
                    elif latkind == "bohr":
                        nruter["lattvec"] *= BOHR_RADIUS
                    elif latkind == "angstrom":
                        nruter["lattvec"] *= .1
                    break
            if l.startswith("CELL_PARAMETERS"):
                try:
                    latkind = kindre.search(l).group("kind")
                except AttributeError:
                    raise ValueError("Type of CELL_PARAMETERS missing")
                if latkind not in ("alat", "bohr", "angstrom"):
                    raise ValueError(
                        "cannot interpret cell parameters in \"{0}\" format"
                        .format(latkind))
                if latkind == "alat" and len(celldm) == 0:
                    raise ValueError("CELL_PARAMETERS alat requires celldm(1)")
                reading = True
        if read < 3:
            raise ValueError("Proper CELL_PARAMETERS not found")
    # Lattvec has been determined, finalize positions
    if poskind == "alat":
        nruter["positions"] *= celldm[1]
    elif poskind == "bohr":
        nruter["positions"] *= BOHR_RADIUS
    elif poskind == "angstrom":
        nruter["positions"] *= .1
    if poskind != "crystal":
        nruter["positions"] = sp.linalg.solve(nruter["lattvec"],
                                              nruter["positions"])
    aux = []
    for e in nruter["elements"]:
        if e not in aux:
            aux.append(e)
    nruter["types"] = [aux.index(i) for i in nruter["elements"]]
    return nruter


def gen_supercell(poscar, na, nb, nc):
    """
    Create a dictionary similar to the first argument but describing a
    supercell.
    """
    nruter = dict()
    nruter["na"] = na
    nruter["nb"] = nb
    nruter["nc"] = nc
    nruter["lattvec"] = np.array(poscar["lattvec"])
    nruter["lattvec"][:, 0] *= na
    nruter["lattvec"][:, 1] *= nb
    nruter["lattvec"][:, 2] *= nc
    nruter["elements"] = []
    nruter["types"] = []
    nruter["positions"] = np.empty(
        (3, poscar["positions"].shape[1] * na * nb * nc))
    pos = 0
    for pos, (k, j, i, iat) in enumerate(
            itertools.product(
                xrange(nc),
                xrange(nb), xrange(na), xrange(poscar["positions"].shape[1]))):
        nruter["positions"][:, pos] = (
            poscar["positions"][:, iat] + [i, j, k]) / [na, nb, nc]
        nruter["elements"].append(poscar["elements"][iat])
        nruter["types"].append(poscar["types"][iat])
    return nruter


def write_supercell(templatefile, poscar, filename, number):
    """
    Create a Quantum Espresso input file for a supercell calculation
    from a template.
    """
    text = open(templatefile, "r").read()
    for i in ("##CELL##", "##NATOMS##", "##COORDINATES##"):
        if i not in text:
            raise ValueError(
                "the template does not contain a {0} tag".format(i))
    text = text.replace("##NATOMS##", str(len(poscar["types"])))
    celltext = "CELL_PARAMETERS angstrom\n" + "\n".join([
        " ".join(["{0:>20.15g}".format(10. * i) for i in j])
        for j in poscar["lattvec"].T.tolist()
    ])
    text = text.replace("##CELL##", celltext)
    coordtext = "ATOMIC_POSITIONS crystal\n" + "\n".join([
        e + " " + " ".join(["{0:>20.15g}".format(i) for i in j])
        for e, j in zip(poscar["elements"], poscar["positions"].T.tolist())
    ])
    text = text.replace("##COORDINATES##", coordtext)
    text = text.replace("##NUMBER##", str(number))
    open(filename, "w").write(text)


def read_forces(filename):
    """
    Read a set of forces on atoms from filename, presumably in
    Quantum Espresso's output format. Units: eV/nm
    """
    nruter = []
    with open(filename, "r") as f:
        for l in f:
            fields = l.split()
            if len(fields
                   ) == 9 and fields[0] == "atom" and fields[4] == "force":
                nruter.append([float(i) for i in fields[6:]])
            elif fields[-3:] == ["contrib.", "to", "forces"]:
                break
    nruter = np.array(nruter) * RYDBERG / BOHR_RADIUS
    return nruter


if __name__ == "__main__":

    def usage():
        """
        Print an usage message and exit.
        """
        sys.exit("""Usage:
\t{program:s} unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
\t{program:s} unitcell.in reap na nb nc cutoff[nm/-integer]"""
                 .format(program=sys.argv[0]))

    if len(sys.argv) not in (7, 8) or sys.argv[2] not in ("sow", "reap"):
        usage()
    ufilename = sys.argv[1]
    action = sys.argv[2]
    na, nb, nc = [int(i) for i in sys.argv[3:6]]
    if action == "sow":
        if len(sys.argv) != 8:
            usage()
        sfilename = sys.argv[7]
    else:
        if len(sys.argv) != 7:
            usage()
    if min(na, nb, nc) < 1:
        sys.exit("Error: na, nb and nc must be positive integers")
    if sys.argv[6][0] == "-":
        try:
            nneigh = -int(sys.argv[6])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if nneigh == 0:
            sys.exit("Error: invalid cutoff")
    else:
        nneigh = None
        try:
            frange = float(sys.argv[6])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if frange == 0.:
            sys.exit("Error: invalid cutoff")
    print("Reading {0}".format(ufilename))
    poscar = read_qe_in(ufilename)
    natoms = len(poscar["types"])
    print("Analyzing symmetries")
    symops = thirdorder_core.SymmetryOperations(
        poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC)
    print("- Symmetry group {0} detected".format(symops.symbol))
    print("- {0} symmetry operations".format(symops.translations.shape[0]))
    print("Creating the supercell")
    sposcar = gen_supercell(poscar, na, nb, nc)
    ntot = natoms * na * nb * nc
    print("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(sposcar)
    if nneigh != None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
        print("- Automatic cutoff: {0} nm".format(frange))
    else:
        print("- User-defined cutoff: {0} nm".format(frange))
    print("Looking for an irreducible set of third-order IFCs")
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts,
                                  frange)
    print("- {0} triplet equivalence classes found".format(wedge.nlist))
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print("- {0} DFT runs are needed".format(nruns))
    if action == "sow":
        print(sowblock)
        print("Writing undisplaced coordinates to BASE.{0}".format(
            os.path.basename(sfilename)))
        write_supercell(sfilename, sposcar,
                        "BASE.{0}".format(os.path.basename(sfilename)), 0)
        width = len(str(4 * (len(list4) + 1)))
        namepattern = "DISP.{0}.{{0:0{1}d}}".format(
            os.path.basename(sfilename), width)
        print("Writing displaced coordinates to DISP.{0}.*".format(
            os.path.basename(sfilename)))
        for i, e in enumerate(list4):
            for n in xrange(4):
                isign = (-1)**(n // 2)
                jsign = -(-1)**(n % 2)
                # Start numbering the files at 1 for aesthetic
                # reasons.
                number = nirred * n + i + 1
                dsposcar = move_two_atoms(sposcar, e[1], e[3], isign * H, e[0],
                                          e[2], jsign * H)
                filename = namepattern.format(number)
                write_supercell(sfilename, dsposcar, filename, number)
    else:
        print(reapblock)
        print("Waiting for a list of QE output files on stdin")
        filelist = []
        for l in sys.stdin:
            s = l.strip()
            if len(s) == 0:
                continue
            filelist.append(s)
        nfiles = len(filelist)
        print("- {0} filenames read".format(nfiles))
        if nfiles != nruns:
            sys.exit("Error: {0} filenames were expected".format(nruns))
        for i in filelist:
            if not os.path.isfile(i):
                sys.exit("Error: {0} is not a regular file".format(i))
        print("Reading the forces")
        forces = []
        for i in filelist:
            forces.append(read_forces(i))
            print("- {0} read successfully".format(i))
            res = forces[-1].mean(axis=0)
            print("- \t Average residual force:")
            print("- \t {0} eV/(nm * atom)".format(res))
        print("Computing an irreducible set of anharmonic force constants")
        phipart = np.zeros((3, nirred, ntot))
        for i, e in enumerate(list4):
            for n in xrange(4):
                isign = (-1)**(n // 2)
                jsign = -(-1)**(n % 2)
                number = nirred * n + i
                phipart[:, i, :] -= isign * jsign * forces[number].T
        phipart /= (4000. * H * H)
        print("Reconstructing the full matrix")
        phifull = thirdorder_core.reconstruct_ifcs(phipart, wedge, list4,
                                                   poscar, sposcar)
        print("Writing the constants to FORCE_CONSTANTS_3RD")
        write_ifcs(phifull, poscar, sposcar, dmin, nequi, shifts, frange,
                   "FORCE_CONSTANTS_3RD")
    print(doneblock)
