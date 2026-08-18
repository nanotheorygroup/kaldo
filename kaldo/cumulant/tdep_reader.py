"""
TDEP force-constant file readers for the cumulant kernels.

Naming and conventions follow Ethan Meitz's LatticeDynamicsToolkit.jl where
applicable (CumulantAnalysis.jl reference).
"""
from __future__ import annotations

import numpy as np


# --- TDEP IFC parsers -------------------------------------------------------

def _read_flat_block(f, n, context):
    """Read ``n`` whitespace-separated floats spanning multiple lines.

    Raises on EOF instead of spinning forever: ``''.split()`` yields no
    tokens, so a bare ``while idx < n: for t in f.readline().split()`` loop
    never advances on a truncated file.
    """
    flat = np.empty(n)
    idx = 0
    while idx < n:
        line = f.readline()
        if not line:
            raise ValueError(f"unexpected end of file while reading {context}")
        for t in line.split():
            flat[idx] = float(t)
            idx += 1
            if idx >= n:
                break
    return flat


def read_tdep_pair_fcs(fc_path, uc_positions, uc_cell, enforce_asr=True):
    """
    Parse TDEP ``infile.forceconstant``.

    ``enforce_asr`` re-imposes the acoustic sum rule per central atom i:
    ``Phi(i, i at R=0) = -sum_{(j, R) != (i, 0)} Phi(i, j at R)``.

    On Ne's sTDEP output the TDEP extractor already writes an ASR-consistent
    FC file (sum residual ~1e-16 eV/Å²) so this is a near-no-op. We keep it
    enabled by default as defensive hygiene: for other potentials or PIMD-TDEP
    runs the residual can be larger.

    Even with ASR exact, Γ acoustic modes show ~1e-11 relative leak from
    floating-point roundoff during the q=0 Fourier sum. Harmless for scalar
    observables.
    """
    neighbors = []
    with open(fc_path) as f:
        n = int(f.readline().split()[0])
        _c = float(f.readline().split()[0])
        for i in range(n):
            nn = int(f.readline().split()[0])
            il = []
            for _ in range(nn):
                j = int(f.readline().split()[0]) - 1
                lp = np.array(f.readline().split(), dtype=float)
                phi = np.array([f.readline().split() for _ in range(3)], dtype=float)
                rj = uc_positions[j] + lp @ uc_cell
                il.append((j, rj, lp, phi))
            neighbors.append(il)

    if enforce_asr:
        for i, il in enumerate(neighbors):
            phi_sum = np.zeros((3, 3))
            self_idx = -1
            for k, (j, rj, lp, phi) in enumerate(il):
                if j == i and np.allclose(lp, 0.0, atol=1e-8):
                    self_idx = k
                else:
                    phi_sum += phi
            phi_self = -phi_sum
            r_self = uc_positions[i]
            if self_idx >= 0:
                il[self_idx] = (i, r_self, np.zeros(3), phi_self)
            else:
                il.append((i, r_self, np.zeros(3), phi_self))

    return neighbors


def read_tdep_ifc3(fc3_path, na_uc=None):
    """Parse TDEP ``infile.forceconstant_thirdorder`` into per-central-atom lists.

    ``na_uc`` is optional: when given, it is asserted to match the file header
    (defensive validation). The atom count actually used comes from the file.
    """
    per_atom = []
    with open(fc3_path) as f:
        na = int(f.readline().split()[0])
        _c = float(f.readline().split()[0])
        if na_uc is not None and na != na_uc:
            raise ValueError(
                f"IFC3 file declares n_atoms={na}, caller passed na_uc={na_uc}"
            )
        for _a1 in range(na):
            nt = int(f.readline().split()[0])
            ts = []
            for _ in range(nt):
                i1 = int(f.readline().split()[0])
                i2 = int(f.readline().split()[0])
                i3 = int(f.readline().split()[0])
                _lv1 = np.array(f.readline().split(), dtype=float)
                lv2 = np.array(f.readline().split(), dtype=float)
                lv3 = np.array(f.readline().split(), dtype=float)
                flat = _read_flat_block(f, 27, "IFC3 tensor block")
                ts.append((i2 - 1, i3 - 1, lv2, lv3, flat.reshape(3, 3, 3)))
            per_atom.append(ts)
    return per_atom


def load_tdep_folder(folder, *, include_fourth=False, enforce_asr=True):
    """One-shot reader for a TDEP sTDEP output folder.

    Returns everything the cumulant F1/F2 kernels need, without going through
    ``kaldo.ForceConstants`` (which currently requires a diagonal
    primitive->ssposcar tiling; SNF non-diagonal support is a follow-up).

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing ``infile.ucposcar``, ``infile.ssposcar``,
        ``infile.forceconstant``, ``infile.forceconstant_thirdorder``,
        (+ ``infile.forceconstant_fourthorder`` if ``include_fourth=True``).
    include_fourth : bool
        If True, also read the IFC4 file and return in the ``quartets`` key.
    enforce_asr : bool
        Pass-through to ``read_tdep_pair_fcs``. Defaults True.

    Returns
    -------
    dict with keys:
        * ``atoms`` (ase.Atoms) — primitive cell
        * ``ss_atoms`` (ase.Atoms) — ssposcar supercell
        * ``uc_positions`` (n_uc, 3) Cartesian Å
        * ``uc_cell`` (3, 3) Cartesian Å
        * ``neighbors`` — IFC2 list per central atom (for F2/harmonic/dynmat)
        * ``triplets`` — IFC3 list per central atom (for F2)
        * ``quartets`` — IFC4 list per central atom (for F1), None if
          ``include_fourth=False``.
        * ``M`` (3, 3) — primitive-to-supercell integer matrix
        * ``is_diagonal`` (bool) — whether M is diagonal.
    """
    import ase.io
    from pathlib import Path
    folder = Path(folder)
    atoms = ase.io.read(str(folder / "infile.ucposcar"), format="vasp")
    ss_atoms = ase.io.read(str(folder / "infile.ssposcar"), format="vasp")
    uc_positions = np.asarray(atoms.positions)
    uc_cell = np.asarray(atoms.cell)
    n_uc = len(atoms)

    M = np.linalg.solve(uc_cell, np.asarray(ss_atoms.cell))
    is_diagonal = np.allclose(M - np.diag(np.diag(M)), 0.0, atol=1e-6)

    neighbors = read_tdep_pair_fcs(
        str(folder / "infile.forceconstant"),
        uc_positions, uc_cell,
        enforce_asr=enforce_asr,
    )
    triplets = read_tdep_ifc3(
        str(folder / "infile.forceconstant_thirdorder"), n_uc,
    )
    quartets = None
    if include_fourth:
        quartets = read_tdep_ifc4(
            str(folder / "infile.forceconstant_fourthorder"), n_uc,
        )

    return dict(
        atoms=atoms,
        ss_atoms=ss_atoms,
        uc_positions=uc_positions,
        uc_cell=uc_cell,
        neighbors=neighbors,
        triplets=triplets,
        quartets=quartets,
        M=M,
        is_diagonal=bool(is_diagonal),
    )


def read_tdep_ifc4(fc4_path, na_uc=None):
    """Parse TDEP ``infile.forceconstant_fourthorder`` into per-central-atom lists.

    ``na_uc`` is optional: when given, asserted to match the file header
    (defensive validation). The atom count actually used comes from the file.
    """
    per_atom = []
    with open(fc4_path) as f:
        na = int(f.readline().split()[0])
        _c = float(f.readline().split()[0])
        if na_uc is not None and na != na_uc:
            raise ValueError(
                f"IFC4 file declares n_atoms={na}, caller passed na_uc={na_uc}"
            )
        for _a1 in range(na):
            nq = int(f.readline().split()[0])
            qs = []
            for _ in range(nq):
                i1 = int(f.readline().split()[0])
                i2 = int(f.readline().split()[0])
                i3 = int(f.readline().split()[0])
                i4 = int(f.readline().split()[0])
                _lv1 = np.array(f.readline().split(), dtype=float)
                lv2 = np.array(f.readline().split(), dtype=float)
                lv3 = np.array(f.readline().split(), dtype=float)
                lv4 = np.array(f.readline().split(), dtype=float)
                flat = _read_flat_block(f, 81, "IFC4 tensor block")
                qs.append((i2 - 1, i3 - 1, i4 - 1, lv2, lv3, lv4, flat.reshape(3, 3, 3, 3)))
            per_atom.append(qs)
    return per_atom


# --- Dynamical matrix ------------------------------------------------------
