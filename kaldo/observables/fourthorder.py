"""
Fourth-order force-constant observable.

Stores the rank-4 interatomic force constants (IFC4) as a sparse-COO tensor
with shape::

    (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)

mirroring :class:`kaldo.observables.thirdorder.ThirdOrder` one rank higher.

TDEP (``format="tdep"``) and pheasy (``format="pheasy"``) are wired, reading
``infile.forceconstant_fourthorder`` via
:func:`kaldo.interfaces.tdep_io.parse_tdep_fourth_forceconstant`. Other
formats will be added as needed.
"""
from __future__ import annotations

import os

import numpy as np

from kaldo.interfaces.tdep_io import parse_tdep_fourth_forceconstant
from kaldo.observables.forceconstant import ForceConstant


class FourthOrder(ForceConstant):
    """Rank-4 interatomic force constants on a primitive × supercell tiling."""

    @classmethod
    def load(cls,
             folder: str,
             supercell: tuple[int, int, int] = (1, 1, 1),
             format: str = "tdep",
             supercell_matrix: np.ndarray | None = None):
        """Load IFC4 from a folder in the given format.

        Parameters
        ----------
        folder : str
            Directory containing the IFC4 + structure files for ``format``.
        supercell : (int, int, int)
            Primitive → supercell tiling. Used for diagonal supercells; a
            non-diagonal ``infile.ssposcar`` is detected automatically and
            routed through the SNF (NonDiagonalGrid) path.
        format : {"tdep", "pheasy"}
            File format. TDEP reads ``infile.forceconstant_fourthorder``; pheasy reads
            ``FORCE_CONSTANTS_4TH``.
        supercell_matrix : np.ndarray, optional
            3x3 integer supercell expansion matrix. Accepted for API symmetry
            with ``ForceConstants.from_folder``; for TDEP the supercell is
            inferred from ``infile.ucposcar`` / ``infile.ssposcar`` instead.
        """
        match format:
            case "tdep":
                from kaldo.interfaces.tdep_io import (
                    build_nondiag_observable_kwargs,
                    attach_snf_metadata,
                    resolve_tdep_supercell,
                )

                uc, sc, diagonal_supercell = resolve_tdep_supercell(folder, supercell, supercell_matrix)
                fc_filename = os.path.join(folder, "infile.forceconstant_fourthorder")

                if diagonal_supercell is None:
                    kw = build_nondiag_observable_kwargs(uc, sc)
                    mapping = kw.pop("_mapping")
                    fourth_ifcs = parse_tdep_fourth_forceconstant(fc_filename=fc_filename, primitive=uc,
                                                                  grid=kw["grid"])
                    fourth_order = cls(value=fourth_ifcs, folder=folder, **kw)
                    return attach_snf_metadata(fourth_order, mapping)

                supercell = diagonal_supercell
                fourth_ifcs = parse_tdep_fourth_forceconstant(fc_filename=fc_filename, primitive=uc,
                                                              supercell=supercell)

                return cls(
                    atoms=uc,
                    replicated_positions=sc.positions,
                    supercell=supercell,
                    value=fourth_ifcs,
                    folder=folder,
                )

            case "pheasy":
                from kaldo.interfaces import pheasy_io

                atoms = pheasy_io.read_pheasy_structure(folder)
                fourth_ifcs = pheasy_io.read_pheasy_fourth(folder, atoms, supercell)
                return cls.from_supercell(
                    atoms=atoms,
                    supercell=tuple(int(x) for x in supercell),
                    grid_type="C",
                    value=fourth_ifcs,
                    folder=folder,
                )

            case _:
                raise ValueError(f"FourthOrder: format={format!r} is not supported")
