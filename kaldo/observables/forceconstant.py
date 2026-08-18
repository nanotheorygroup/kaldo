"""Shared geometry and translation topology for real-space IFC containers.

An IFC file has two related but different lattices. ``SupercellGrid`` records
the finite supercell's physical periodic classes, while ``TranslationSupport``
records the translations that actually index the IFC tensor. File formats may
store several translations from one periodic class, so those axes must never
be inferred from the number of replicated atoms.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.helpers.logger import get_logger
from kaldo.observables.observable import Observable
from ase import Atoms
logging = get_logger()


def chi(qvec, list_of_replicas, cell_inv):
    chi_k = np.exp(1j * 2 * np.pi * list_of_replicas.dot(cell_inv.dot(qvec.T)))
    return chi_k


class ForceConstant(Observable):
    """Base container for an IFC tensor and its two translation topologies.

    ``n_replicas`` and ``replica_translations`` describe the physical
    Born--von Karman supercell: exactly one representative of every quotient
    class. ``n_translations`` and ``translation_support`` instead describe the
    tensor axes used in Fourier sums. They coincide for compact periodic
    inputs, but literal TDEP or ShengBTE data and compiled Wigner--Seitz IFC3
    can have ``n_translations != n_replicas``.

    The optional ``ifc_interpolation_hint`` records a format's native Fourier
    convention. It is provenance supplied by an interface, not a user-facing
    interpolation choice; :class:`HarmonicWithQ` and :class:`Phonons` resolve
    the public ``ifc_interpolation`` request against it.
    """

    def __init__(
        self,
        atoms: Atoms,
        replicated_positions: NDArray,
        supercell: tuple[int, int, int] | NDArray,
        folder: str,
        value: ArrayLike | None = None,
        supercell_grid: SupercellGrid | None = None,
        translation_support: TranslationSupport | None = None,
        replica_translations: ArrayLike | None = None,
        ifc_interpolation_hint: str | None = None,
        **kwargs,
    ):
        """Validate and attach physical replicas and IFC translation axes."""
        super().__init__(folder=folder, **kwargs)
        self.atoms = atoms
        self.supercell = supercell
        self.value = value

        self._replicated_atoms = None
        # Loaders may provide either a flat atom list or an explicit
        # (replica, basis atom, Cartesian) array. Normalize both forms so the
        # first axis can be checked against the physical quotient size.
        self.replicated_positions = replicated_positions.reshape(
            (-1, self.atoms.positions.shape[0], self.atoms.positions.shape[1])
        )
        if supercell_grid is None:
            matrix = np.asarray(supercell)
            if matrix.shape == (3,):
                matrix = np.diag(matrix)
            order = self._detect_replica_order(matrix)
            supercell_grid = SupercellGrid(matrix, order=order)
        if translation_support is None:
            translation_support = TranslationSupport.periodic(supercell_grid)
        if not np.array_equal(
            translation_support.supercell.matrix, supercell_grid.matrix
        ):
            raise ValueError("translation_support must belong to supercell_grid")
        self.supercell_grid = supercell_grid
        self.translation_support = translation_support
        if ifc_interpolation_hint not in (None, "periodic", "wigner-seitz"):
            raise ValueError(
                "ifc_interpolation_hint must be None, 'periodic', or 'wigner-seitz'"
            )
        self.ifc_interpolation_hint = ifc_interpolation_hint
        self.n_replicas = supercell_grid.size
        self.n_translations = translation_support.size
        if replica_translations is None:
            replica_translations = supercell_grid.representatives
        replica_translations = np.asarray(replica_translations)
        if replica_translations.shape != (self.n_replicas, 3) or not np.allclose(
            replica_translations,
            np.rint(replica_translations),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                "replica_translations must have shape (n_replicas, 3) and be integer"
            )
        self.replica_translations = np.array(
            np.rint(replica_translations), dtype=np.int64, copy=True
        )
        class_ids = [
            supercell_grid.class_id(vector) for vector in self.replica_translations
        ]
        if sorted(class_ids) != list(range(self.n_replicas)):
            raise ValueError(
                "replica_translations must contain each periodic class exactly once"
            )
        if len(self.replicated_positions) != self.n_replicas:
            raise ValueError(
                "replicated_positions must contain one physical structure for "
                f"each of the {self.n_replicas} supercell classes"
            )
        if value is not None and getattr(value, "ndim", None) == 6:
            if value.shape[3] != self.n_translations:
                raise ValueError(
                    "IFC2 translation axis does not match translation_support"
                )
        if value is not None and getattr(value, "ndim", None) == 8:
            if (
                value.shape[2] != self.n_translations
                or value.shape[5] != self.n_translations
            ):
                raise ValueError(
                    "IFC3 translation axes do not match translation_support"
                )
        if value is not None and getattr(value, "ndim", None) == 11:
            if any(value.shape[axis] != self.n_translations for axis in (2, 5, 8)):
                raise ValueError(
                    "IFC4 translation axes do not match translation_support"
                )
        self._cell_inv = None
        self._replicated_cell_inv = None
        self._list_of_replicas = None

    def _detect_replica_order(self, matrix):
        """Detect the historical C/F ordering from physical replica positions."""
        matrix = np.asarray(matrix, dtype=np.int64)
        if np.count_nonzero(matrix - np.diag(np.diag(matrix))) or np.any(
            np.diag(matrix) <= 0
        ):
            return "C"
        detected = np.rint(
            (self.replicated_positions - self.atoms.positions[None, :, :])
            @ np.linalg.inv(self.atoms.cell)
        ).astype(np.int64)
        if not np.all(detected == detected[:, :1, :]):
            raise ValueError(
                "replicated_positions do not describe rigid lattice translations"
            )
        detected = detected[:, 0, :]
        for order in ("C", "F"):
            if np.array_equal(
                SupercellGrid(matrix, order=order).representatives, detected
            ):
                logging.debug("Using %s-style replica position order", order)
                return order
        raise ValueError("Unable to detect C/F replica order from replicated_positions")

    @classmethod
    def from_supercell(
        cls,
        atoms: Atoms,
        supercell: tuple[int, int, int] | NDArray,
        grid_type: str,
        value: ArrayLike | None = None,
        folder: str = "kALDo",
        **kwargs,
    ):
        """Construct compact physical replicas for a diagonal or matrix cell.

        The physical positions use one deterministic representative of every
        supercell quotient class. A caller may still pass a distinct
        ``translation_support`` in ``kwargs`` when the IFC tensor carries
        literal or Wigner--Seitz-expanded translation axes.
        """
        matrix = np.asarray(supercell)
        if matrix.shape == (3,):
            matrix = np.diag(matrix)
        elif matrix.shape != (3, 3):
            raise ValueError("supercell must be a length-3 vector or 3x3 matrix")
        supercell_grid = SupercellGrid(matrix, order=grid_type)
        translation_support = kwargs.pop(
            "translation_support", TranslationSupport.periodic(supercell_grid)
        )
        _grid_arr = supercell_grid.representatives
        # R @ cell + r_i constructs one physical copy of every basis atom for
        # each quotient representative R. These positions do not define the
        # IFC Fourier axes; ``translation_support`` does.
        replicated_positions = (
            _grid_arr.dot(atoms.cell)[:, np.newaxis, :]
            + atoms.positions[np.newaxis, :, :]
        )
        return cls(
            atoms=atoms,
            replicated_positions=replicated_positions,
            supercell=supercell,
            value=value,
            folder=folder,
            supercell_grid=supercell_grid,
            translation_support=translation_support,
            replica_translations=_grid_arr,
            **kwargs,
        )

    @property
    def positions(self):
        return self.atoms.positions


    @property
    def cell_inv(self):
        if self._cell_inv is None:
            self._cell_inv = np.linalg.inv(self.atoms.cell)
        return self._cell_inv


    @property
    def replicated_atoms(self):
        """Physical Born--von Karman supercell as an ASE ``Atoms`` object."""
        if self._replicated_atoms is None:
            atoms = self.atoms
            n_replicas = len(self.replicated_positions)
            replicated_atoms = Atoms(
                symbols=atoms.get_chemical_symbols() * n_replicas,
                positions=self.replicated_positions.reshape(-1, 3),
                cell=self._replicated_cell,
                pbc=atoms.pbc,
            )
            self._replicated_atoms = replicated_atoms
        return self._replicated_atoms

    @property
    def _replicated_cell(self):
        """Cartesian cell of the replicated (supercell) structure.

        The exact integer tiling matrix also covers non-diagonal supercells.
        """
        return np.asarray(self.supercell_grid.matrix) @ np.asarray(self.atoms.cell)

    @property
    def replicated_cell_inv(self):
        if self._replicated_cell_inv is None:
            self._replicated_cell_inv = np.linalg.inv(self.replicated_atoms.cell)
        return self._replicated_cell_inv


    @property
    def list_of_replicas(self):
        """Cartesian vectors for the IFC tensor's translation support.

        The legacy property name is retained for callers, but the returned
        length is ``n_translations`` and need not equal ``n_replicas``.
        """
        if self._list_of_replicas is None:
            list_of_index = self.translation_support.translations
            self._list_of_replicas = list_of_index.dot(self.atoms.cell)
        return self._list_of_replicas

    def _chi_k(self, k_points):
        """Return ``exp(2*pi*i*q.R)`` for every q and stored IFC translation."""
        n_k_points = np.shape(k_points)[0]
        ch = np.zeros((n_k_points, self.n_translations), dtype=complex)
        for index_q in range(n_k_points):
            k_point = k_points[index_q]

            list_of_replicas = self.list_of_replicas
            cell_inv = self.cell_inv
            ch[index_q] = chi(k_point, list_of_replicas, cell_inv)
        return ch

