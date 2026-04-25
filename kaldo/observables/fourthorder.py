"""
Fourth-order force-constant observable.

Stores the rank-4 interatomic force constants (IFC4) as a sparse-COO tensor
with shape::

    (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)

mirroring :class:`kaldo.observables.thirdorder.ThirdOrder` one rank higher.

Currently only the TDEP format is wired (``format="tdep"``), reading
``infile.forceconstant_fourthorder`` via
:func:`kaldo.interfaces.tdep_io.parse_tdep_fourth_forceconstant`. Other
formats will be added as needed.
"""
from __future__ import annotations

import os

import ase.io
import numpy as np

from kaldo.helpers.logger import get_logger
from kaldo.interfaces.tdep_io import parse_tdep_fourth_forceconstant
from kaldo.observables.forceconstant import ForceConstant

logging = get_logger()


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
            Primitive → supercell tiling (must be diagonal for TDEP today;
            SNF support is a planned follow-up — see DESIGN_TDEP_READER.md).
        format : {"tdep"}
            File format. Only ``"tdep"`` is supported at this time.
        """
        match format:
            case "tdep":
                uc = ase.io.read(os.path.join(folder, "infile.ucposcar"), format="vasp")
                sc = ase.io.read(os.path.join(folder, "infile.ssposcar"), format="vasp")
                M = np.linalg.solve(np.asarray(uc.cell), np.asarray(sc.cell))

                from kaldo.interfaces.tdep_io import (
                    validate_tdep_supercell_matrix,
                    build_nondiag_observable_kwargs,
                    attach_snf_metadata,
                    parse_tdep_fourth_forceconstant_nondiag,
                )
                M_int = validate_tdep_supercell_matrix(supercell_matrix, M, supercell)
                if M_int is not None:
                    kw = build_nondiag_observable_kwargs(uc, sc)
                    mapping = kw.pop("_mapping")
                    fourth_ifcs = parse_tdep_fourth_forceconstant_nondiag(
                        fc_filename=os.path.join(folder, "infile.forceconstant_fourthorder"),
                        primitive=uc,
                        replica_table=mapping["replica_table"],
                        M=mapping["M"],
                    )
                    fourth_order = cls(value=fourth_ifcs, folder=folder, **kw)
                    return attach_snf_metadata(fourth_order, mapping)

                fourth_ifcs = parse_tdep_fourth_forceconstant(
                    fc_filename=os.path.join(folder, "infile.forceconstant_fourthorder"),
                    primitive=os.path.join(folder, "infile.ucposcar"),
                    supercell=supercell,
                )

                return cls(
                    atoms=uc,
                    replicated_positions=sc.positions,
                    supercell=supercell,
                    value=fourth_ifcs,
                    folder=folder,
                )

            case _:
                raise ValueError(f"FourthOrder: format={format!r} is not supported")
