"""Third-order IFC storage, loading, and source-aware interpolation.

IFC3 has two independently phased real-space legs.  This module preserves the
translations supplied by each interface, or compiles compact periodic blocks
onto pair-specific Wigner--Seitz images, before the anharmonic controller
projects the tensor into phonon modes.
"""

from dataclasses import dataclass

from kaldo.grid import TranslationSupport, WignerSeitzImages
from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import ase.io
import hashlib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sparse import COO
from kaldo.interfaces.eskm_io import import_from_files
from kaldo.interfaces.tdep_io import parse_tdep_third_forceconstant
import kaldo.interfaces.shengbte_io as shengbte_io
import kaldo.interfaces.qe_io as qe_io
import ase.units as units
from kaldo.controllers.displacement import calculate_third, try_symmetrize_ifc
from kaldo.parallel import is_parallel, validate_parallel_calculator, maybe_warn_ml_delta_shift
from kaldo.helpers.logger import get_logger

logging = get_logger()

REPLICATED_ATOMS_THIRD_FILE = 'replicated_atoms_third.xyz'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'

_IFC_INTERPOLATION_MODES = ("auto", "wigner-seitz", "periodic")


def _ifc3_source_digest(source_value):
    """Content digest of the raw IFC3 tensor, for interpolation caching.

    ``id()`` is not a safe cache key: in-place mutation keeps the id while
    changing the content, and a freed id can be reused by a new object.
    Hash the actual buffers instead (sparse coordinates and data, or the
    dense array).
    """
    h = hashlib.sha256()
    if hasattr(source_value, "coords") and hasattr(source_value, "data"):
        h.update(np.ascontiguousarray(source_value.coords).tobytes())
        h.update(np.ascontiguousarray(source_value.data).tobytes())
    else:
        h.update(np.ascontiguousarray(source_value).tobytes())
    h.update(str(source_value.shape).encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class _ThirdOrderIFCInterpolation:
    """Immutable result of compiling IFC3 onto a translation support.

    ``value`` follows ``(i,a,Rj,j,b,Rk,k,c)``.  Its two translation axes
    refer to the same ``support``.  Wigner--Seitz compilation changes only
    those integer translations and partitions tied images by their geometric
    weights; it does not add basis offsets to kALDo's Fourier phase gauge.
    """

    value: object
    support: TranslationSupport
    resolved_mode: str


def _as_coo(value):
    """Return a pydata/sparse COO without densifying an existing sparse IFC."""
    if isinstance(value, COO):
        return value
    return COO.from_numpy(np.asarray(value))


def _rank8_ifc3(value, n_atoms, n_translations):
    """Normalize legacy flattened IFC3 storage without densifying COO data."""
    expected = (
        n_atoms,
        3,
        n_translations,
        n_atoms,
        3,
        n_translations,
        n_atoms,
        3,
    )
    if getattr(value, "ndim", None) == 8:
        if value.shape != expected:
            raise ValueError(
                f"third-order IFC shape {value.shape} does not match {expected}"
            )
        return value
    flattened = (
        n_atoms * 3,
        n_translations * n_atoms * 3,
        n_translations * n_atoms * 3,
    )
    if getattr(value, "ndim", None) == 3 and value.shape == flattened:
        return value.reshape(expected)
    raise ValueError(
        "third-order IFC value must have rank 8, or the legacy flattened "
        f"shape {flattened}"
    )


def _coalesced_coo(coords, data, shape, dtype):
    """Build a deterministic COO and sum entries mapped to the same slot."""
    data = np.asarray(data, dtype=dtype)
    if data.size:
        coords = np.asarray(coords, dtype=np.int64)
        if coords.shape[0] != len(shape):
            coords = coords.T
        return COO(
            coords,
            data,
            shape=shape,
            has_duplicates=True,
            sorted=False,
        )
    return COO(
        np.empty((len(shape), 0), dtype=np.int64), np.empty(0, dtype=dtype), shape=shape
    )


def detect_path(files: list[str], folder: str = ""):
    """return the path and the filename of the first existed file in the ``files`` list in ``folder``.
    Raise an error if none of the files in the list is found in ``folder``.
    """
    # file_list = list(map(lambda f: os.path.join(folder, f), files))
    results = list(filter(lambda f: os.path.isfile(os.path.join(folder, f)), files))
    if results:
        return os.path.join(folder, results[0]), results[0]
    else:
        raise ValueError(f"{' or '.join(files)} are not found.")


class ThirdOrder(ForceConstant):
    """Third-order IFCs with explicit translation axes for both outer legs.

    The canonical tensor layout is ``(i,a,Rj,j,b,Rk,k,c)``. ``Rj`` and
    ``Rk`` index :attr:`translation_support`, not the physical replica count.
    Legacy rank-three sparse storage is accepted at load time and reshaped
    without densification when interpolation begins.

    :meth:`get_interpolation` returns an immutable plan rather than mutating
    the source tensor. This preserves literal file provenance and allows
    explicit periodic or Wigner--Seitz diagnostics to coexist safely.
    """

    def get_interpolation(self, mode="auto"):
        """Compile IFC3 for the requested real-space interpolation rule.

        ``auto`` preserves literal file-provided translations, honors an
        interface's explicit native-gauge hint (for example d3q's unrecentered
        cell indices), and otherwise selects pair-dependent Wigner--Seitz
        interpolation.  ``periodic`` and ``wigner-seitz`` are explicit user
        overrides: both first fold every source translation into its exact
        supercell quotient class.  The latter then distributes each ``(i,j)``
        and ``(i,k)`` pair independently over all tied shortest images, with
        the Cartesian-product weight.

        Sparse input remains sparse throughout compilation.  The returned
        object is cached per requested mode and source identity.
        """
        if mode not in _IFC_INTERPOLATION_MODES:
            raise ValueError(
                f"ifc_interpolation={mode!r} is invalid; choose one of "
                f"{_IFC_INTERPOLATION_MODES}"
            )
        support = self.translation_support
        source_value = self.value
        if source_value is None:
            raise ValueError("third-order IFC value is not available")
        value = _rank8_ifc3(source_value, len(self.atoms), support.size)

        cache_key = (mode, _ifc3_source_digest(source_value), support.digest)
        cache = getattr(self, "_interpolation_cache", None)
        if cache is None:
            cache = {}
            self._interpolation_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        hint = getattr(self, "ifc_interpolation_hint", None)
        if mode == "auto" and support.provenance == "file":
            result = _ThirdOrderIFCInterpolation(value, support, "file")
        else:
            resolved = (hint or "wigner-seitz") if mode == "auto" else mode
            if support.provenance == "file" and mode != "auto":
                logging.warning(
                    "ifc_interpolation=%r folds %d file-provided IFC3 "
                    "translations into %d periodic classes by explicit "
                    "user request.",
                    mode,
                    support.size,
                    support.supercell.size,
                )
            folded, periodic_support = self._fold_periodic_classes(value, support)
            if resolved == "periodic":
                result = _ThirdOrderIFCInterpolation(folded, periodic_support, resolved)
            else:
                compiled, compiled_support = self._compile_wigner_seitz(
                    folded, periodic_support
                )
                result = _ThirdOrderIFCInterpolation(
                    compiled, compiled_support, resolved
                )
        cache[cache_key] = result
        return result

    def gamma_contracted_value(self):
        """Return IFC3 summed over both real-space translation axes.

        At Gamma the two IFC3 Fourier phases are identically one. The exact
        tensor consumed by a Gamma-only amorphous calculation is therefore
        ``sum_Rj,Rk Phi(Rj,Rk)``. Performing that reduction on the canonical
        rank-eight representation supports compact, literal, and expanded
        translation supports without confusing their size with the physical
        replica count.

        Returns
        -------
        sparse.COO or numpy.ndarray
            Tensor with shape ``(3*n_atoms, 3*n_atoms, 3*n_atoms)`` in the
            historical ``(central, partner-j, partner-k)`` mode ordering.
        """
        rank8 = _rank8_ifc3(self.value, len(self.atoms), self.n_translations)
        n_modes = 3 * len(self.atoms)
        return rank8.sum(axis=(2, 5)).reshape((n_modes, n_modes, n_modes))

    @staticmethod
    def _fold_periodic_classes(value, support):
        """Coalesce both IFC3 translation axes into exact quotient classes."""
        source = _as_coo(value)
        coords = np.array(source.coords, dtype=np.int64, copy=True)
        coords[2] = support.class_ids[coords[2]]
        coords[5] = support.class_ids[coords[5]]
        shape = list(source.shape)
        shape[2] = shape[5] = support.supercell.size
        folded = COO(
            coords,
            np.asarray(source.data),
            shape=tuple(shape),
            has_duplicates=True,
            sorted=False,
        )
        periodic_support = TranslationSupport.periodic(
            support.supercell, order=support.supercell.order
        )
        return folded, periodic_support

    def _compile_wigner_seitz(self, folded, support):
        """Partition nonzero IFC3 entries over pair-shortest images.

        Pair geometry is evaluated once per distinct ``(R,i,j)`` leg, not
        once per Cartesian tensor component.  Coordinate expansion is then
        vectorized by tied-image index.  This keeps the cost proportional to
        the sparse IFC support and avoids millions of Python list appends for
        production IFC3 tensors.
        """
        images = WignerSeitzImages.build(
            support,
            np.asarray(self.atoms.positions),
            np.asarray(self.atoms.cell),
            pbc=self.atoms.pbc,
        )
        coords = np.asarray(folded.coords, dtype=np.int64)
        data = np.asarray(folded.data)

        # The first three columns identify a reusable IFC leg: stored
        # translation, central atom i, and partner atom j/k.  np.unique also
        # gives a vectorized map from every scalar COO entry to that geometry.
        pair_j = np.column_stack((coords[2], coords[0], coords[3]))
        pair_k = np.column_stack((coords[5], coords[0], coords[6]))
        pair_rows = np.concatenate((pair_j, pair_k), axis=0)
        unique_pairs, pair_inverse = np.unique(pair_rows, axis=0, return_inverse=True)
        pair_inverse = np.reshape(pair_inverse, -1)
        pair_j_ids = pair_inverse[: data.size]
        pair_k_ids = pair_inverse[data.size :]

        pair_images = [
            images.image(source_id, atom_i, atom_j)
            for source_id, atom_i, atom_j in unique_pairs
        ]

        # A shared, sorted axis is required because Rj and Rk occupy axes of
        # one tensor. Build it from precisely the images used by nonzero IFCs.
        translations = {
            tuple(int(x) for x in translation)
            for image_translations, _, _ in pair_images
            for translation in image_translations
        }
        if not translations:
            translations.update(tuple(int(x) for x in r) for r in support.translations)
        ordered = np.asarray(sorted(translations), dtype=np.int64)
        compiled_support = TranslationSupport(
            ordered, support.supercell, provenance="wigner-seitz"
        )
        translation_ids = {tuple(r): index for index, r in enumerate(ordered)}

        max_multiplicity = max((len(item[0]) for item in pair_images), default=1)
        target_ids = np.full((len(pair_images), max_multiplicity), -1, dtype=np.int64)
        pair_weights = np.zeros((len(pair_images), max_multiplicity), dtype=float)
        for pair_id, (pair_translations, _, weights) in enumerate(pair_images):
            if not np.isclose(np.sum(weights), 1.0, rtol=0, atol=1e-14):
                raise ValueError(
                    "Wigner-Seitz weights for an IFC3 pair do not sum to one"
                )
            multiplicity = len(weights)
            target_ids[pair_id, :multiplicity] = [
                translation_ids[tuple(r)] for r in pair_translations
            ]
            pair_weights[pair_id, :multiplicity] = weights

        # At most ``max_multiplicity**2`` vector operations are needed.  Each
        # selects every COO scalar with the same tied-image indices at once.
        output_coords = []
        output_data = []
        for image_j in range(max_multiplicity):
            targets_j = target_ids[pair_j_ids, image_j]
            weights_j = pair_weights[pair_j_ids, image_j]
            for image_k in range(max_multiplicity):
                targets_k = target_ids[pair_k_ids, image_k]
                selected = (targets_j >= 0) & (targets_k >= 0)
                if not np.any(selected):
                    continue
                target = np.array(coords[:, selected], copy=True)
                target[2] = targets_j[selected]
                target[5] = targets_k[selected]
                output_coords.append(target)
                output_data.append(
                    data[selected]
                    * weights_j[selected]
                    * pair_weights[pair_k_ids[selected], image_k]
                )

        shape = list(folded.shape)
        shape[2] = shape[5] = compiled_support.size
        if output_coords:
            output_coords = np.concatenate(output_coords, axis=1)
            output_data = np.concatenate(output_data)
        else:
            output_coords = np.empty((len(shape), 0), dtype=np.int64)
            output_data = np.empty(0, dtype=folded.dtype)
        compiled = _coalesced_coo(
            output_coords, output_data, tuple(shape), folded.dtype
        )
        # Redistribution must preserve the Gamma zeroth moment separately for
        # every Cartesian (i,a,j,b,k,c) block.  A comparison of one global
        # signed sum is ill-conditioned for physical IFC3 tensors because the
        # acoustic sum rule makes that scalar nearly zero through cancellation.
        # Reducing only the two translation axes tests the actual invariant
        # without densifying the six remaining atom/Cartesian axes.
        source_gamma = folded.sum(axis=(2, 5))
        compiled_gamma = compiled.sum(axis=(2, 5))
        gamma_error = compiled_gamma - source_gamma
        error = float(np.max(np.abs(gamma_error.data))) if gamma_error.nnz else 0.0
        scale = max(
            float(np.max(np.abs(source_gamma.data))) if source_gamma.nnz else 0.0,
            1.0,
        )
        if error > 5e-13 * scale:
            raise ValueError(
                "Wigner-Seitz IFC3 compilation did not conserve the per-block "
                "Gamma translation sum"
            )
        return compiled, compiled_support

    @classmethod
    def load(
        cls,
        folder: str,
        supercell: tuple[int, int, int] | np.ndarray = (1, 1, 1),
        format: str = "sparse",
        third_energy_threshold: float = 0.0,
        chunk_size: int = 100000,
        supercell_matrix: np.ndarray | None = None,
        atoms_override: Atoms | None = None,
    ):
        """
        Load third order force constants from a folder in the given format, used for library internally.

        To load force constants data, ``ForceConstants.from_folder`` is recommended.

        Parameters
        ----------
        folder : str
            Specifies where to load the data files.
        supercell : tuple[int, int, int] or ndarray
            Diagonal repetitions or an integer 3 by 3 supercell matrix for
            the third-order force constants. TDEP and ShengBTE-style literal
            IFC3 readers support matrices; compact legacy formats require a
            diagonal three-vector.
            Default: (1, 1, 1)
        format : str
            Format of the third order force constant information being loaded into ForceConstant object.
            Default: 'sparse'
        third_energy_threshold : float, optional
            When importing sparse third order force constant matrices, energies below
            the threshold value in magnitude are ignored. Units: eV/A^3
            Default: `None`
        chunk_size : int, optional
            Number of entries to process per chunk when reading sparse third order files.
            Larger values use more memory but may be faster for very large files.
            Default: 100000
        supercell_matrix : np.ndarray, optional
            Expected 3x3 integer supercell expansion matrix. For TDEP the
            structure files remain authoritative, and a supplied matrix must
            match their inferred (possibly non-diagonal) tiling.
            Default: None
        atoms_override : ase.Atoms, optional
            Authoritative primitive structure supplied by an already loaded
            IFC2 object. Used for QE q2r-backed formats so IFC3 cannot be
            interpreted in a stale CONTROL/POSCAR atom order.
            Default: None

        Returns
        -------
        third_order : ThirdOrder object
            A new instance of the ThirdOrder class
        """

        matrix_capable_formats = {
            "tdep",
            "vasp-sheng",
            "shengbte",
            "qe-sheng",
            "shengbte-qe",
        }
        supplied_matrix = np.asarray(supercell)
        if supplied_matrix.shape == (3, 3) and format not in matrix_capable_formats:
            diagonal = np.diag(supplied_matrix)
            if np.array_equal(supplied_matrix, np.diag(diagonal)):
                rounded = np.rint(diagonal)
                if not np.allclose(diagonal, rounded, rtol=0, atol=1e-12) or np.any(
                    rounded <= 0
                ):
                    raise ValueError(
                        "diagonal supercell matrix entries must be positive integers"
                    )
                supercell = tuple(int(value) for value in rounded)
            else:
                raise ValueError(
                    f"format={format!r} does not encode a non-diagonal IFC3 "
                    "topology; use TDEP/ShengBTE input or a diagonal supercell"
                )

        match format:
            case 'sparse' | 'numpy':
                config_path, _ = detect_path([REPLICATED_ATOMS_THIRD_FILE, REPLICATED_ATOMS_FILE], folder)
                replicated_atoms = ase.io.read(config_path, format='extxyz')

                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = n_total_atoms // n_replicas
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols,
                              positions=unit_positions,
                              cell=unit_cell,
                              pbc=[1, 1, 1])

                _third_order = (
                    COO.from_scipy_sparse(
                        load_npz(os.path.join(folder, THIRD_ORDER_FILE_SPARSE))
                    )
                    .reshape(
                        (
                            n_unit_atoms * 3,
                            n_replicas * n_unit_atoms * 3,
                            n_replicas * n_unit_atoms * 3,
                        )
                    )
                    .astype(np.float64)
                )
                third_order = ThirdOrder(
                    atoms=atoms,
                    replicated_positions=replicated_atoms.positions,
                    supercell=supercell,
                    value=_third_order,
                    folder=folder,
                )

            case 'eskm' | 'lammps':
                if format == 'eskm':
                    config_file = os.path.join(folder, "CONFIG")
                    replicated_atoms = ase.io.read(config_file, format='dlp4')
                elif format == 'lammps':
                    # Mixed-supercell support: a replicated_atoms_third.xyz
                    # describes a (smaller) third-order supercell alongside the
                    # second-order replicated_atoms.xyz, matching the
                    # convention of the sparse/numpy loaders.
                    config_file, config_name = detect_path(
                        [REPLICATED_ATOMS_THIRD_FILE, REPLICATED_ATOMS_FILE], folder)
                    if config_name == REPLICATED_ATOMS_THIRD_FILE:
                        logging.info(f'Loading third-order supercell from {config_name} '
                                     f'(mixed-supercell run; second order uses {REPLICATED_ATOMS_FILE})')
                    replicated_atoms = ase.io.read(config_file, format='extxyz')

                third_file = os.path.join(folder, "THIRD")
                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = n_total_atoms // n_replicas
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols,
                              positions=unit_positions,
                              cell=unit_cell,
                              pbc=[1, 1, 1])

                out = import_from_files(replicated_atoms=replicated_atoms,
                                        third_file=third_file,
                                        supercell=supercell,
                                        third_energy_threshold=third_energy_threshold,
                                        chunk_size=chunk_size)
                third_order = ThirdOrder(atoms=atoms,
                                         replicated_positions=replicated_atoms.positions,
                                         supercell=supercell,
                                         value=out[1],
                                         folder=folder)

            case ("vasp-sheng" | "shengbte") | ("qe-sheng" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q") | "vasp-d3q":
                # all these readers produce C-ordered replica data; declared
                # together with SecondOrder.load for the same formats (#272)
                grid_type = 'C'
                config_path, config_file = detect_path(['CONTROL', 'POSCAR'], folder)
                match config_file:
                    case 'CONTROL':
                        atoms, _supercell, charges = shengbte_io.import_control_file(config_path)
                    case 'POSCAR':
                        logging.info('Trying to open POSCAR')
                        atoms = ase.io.read(config_path)

                if atoms_override is not None:
                    if format not in (
                        "qe-sheng",
                        "shengbte-qe",
                        "qe-d3q",
                        "shengbte-d3q",
                    ):
                        raise ValueError(
                            "atoms_override is only valid for QE q2r-backed IFC3 formats"
                        )
                    header = qe_io.read_q2r_header(
                        os.path.join(folder, "espresso.ifc2")
                    )
                    # IFC2-only loading may diagnose and ignore an unrelated
                    # auxiliary structure. IFC2+IFC3 cannot: the third-order
                    # tensor has the same atom labels, so a mismatch would
                    # silently combine two different crystals.
                    qe_io.validate_q2r_auxiliary_structure(header, atoms, strict=True)
                    qe_io.validate_q2r_auxiliary_structure(
                        header, atoms_override, strict=True
                    )
                    # IFC2 supplies the authoritative q2r geometry and atom
                    # order, but IFC2 and IFC3 must not share one mutable
                    # Atoms instance. Otherwise wrapping or translating one
                    # observable silently changes the other's Fourier gauge.
                    atoms = atoms_override.copy()

                match format:
                    case ("vasp-sheng" | "shengbte") | ("qe-sheng" | "shengbte-qe"):
                        # load VASP third order force constant
                        third_file = os.path.join(folder, 'FORCE_CONSTANTS_3RD')
                        third_order, translation_support = (
                            shengbte_io.read_third_order_matrix(
                                third_file,
                                atoms,
                                supercell,
                                order="C",
                                return_support=True,
                            )
                        )
                    case _:
                        # load d3q third order force constant
                        third_file = os.path.join(folder, 'FORCE_CONSTANTS_3RD_D3Q')
                        third_order = qe_io.read_third_d3q(
                            third_file, atoms, supercell, order="C"
                        )
                        translation_support = None
                third_order = ThirdOrder.from_supercell(
                    atoms=atoms,
                    grid_type=grid_type,
                    supercell=supercell,
                    value=third_order,
                    # d3q writes explicit unrecentered cell indices; retain
                    # that native direct Fourier gauge.
                    ifc_interpolation_hint=(
                        "periodic"
                        if format
                        in (
                            "qe-d3q",
                            "shengbte-d3q",
                            "vasp-d3q",
                        )
                        else None
                    ),
                    folder=folder,
                    **(
                        {"translation_support": translation_support}
                        if translation_support is not None
                        else {}
                    ),
                )

            case 'hiphive':
                filename = 'atom_prim.xyz'
                # TODO: add replicated filename in example
                replicated_filename = 'replicated_atoms.xyz'
                try:
                    import kaldo.interfaces.hiphive_io as hiphive_io
                except ImportError:
                    logging.error(
                        "In order to use hiphive along with kaldo, hiphive is required. \
                        Please consider installing hihphive. More info can be found at: \
                        https://hiphive.materialsmodeling.org/"
                    )

                atom_prime_file = os.path.join(folder, filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                # TODO: Make this independent of replicated file
                atoms = ase.io.read(atom_prime_file)
                if os.path.isfile(replicated_atom_prime_file):
                    replicated_atoms = ase.io.read(replicated_atom_prime_file)
                else:
                    logging.warning('Replicated atoms file not found. Please check if the file exists. Use the unit cell atoms instead.')
                    replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])

                if 'model3.fcs' in os.listdir(str(folder)):
                    # Derive constants used for third-order reshape
                    supercell = np.array(supercell)
                    n_prim = atoms.copy().get_masses().shape[0]
                    n_sc = np.prod(supercell)
                    pbc_conditions = replicated_atoms.get_pbc()
                    dim = len(pbc_conditions[pbc_conditions == True])
                    _third_order = hiphive_io.import_third_from_hiphive(atoms, supercell, folder)
                    _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                           n_sc * n_prim * dim)
                    third_order = cls(atoms=atoms,
                                      replicated_positions=replicated_atoms.positions,
                                      supercell=supercell,
                                      value=_third_order,
                                      folder=folder)

            case 'tdep':
                from kaldo.interfaces.tdep_io import (
                    build_nondiag_observable_kwargs,
                    attach_snf_metadata,
                    resolve_tdep_supercell,
                )
                from kaldo.grid import SupercellGrid

                uc, sc, diagonal_supercell = resolve_tdep_supercell(folder, supercell, supercell_matrix)
                fc_filename = os.path.join(folder, 'infile.forceconstant_thirdorder')

                matrix = np.rint(
                    np.asarray(sc.cell) @ np.linalg.inv(np.asarray(uc.cell))
                ).astype(int)
                physical_grid = SupercellGrid(matrix, order="C")
                third_ifcs, support = parse_tdep_third_forceconstant(
                    fc_filename=fc_filename,
                    primitive=uc,
                    supercell_grid=physical_grid,
                    return_support=True,
                )
                if diagonal_supercell is None:
                    # Deliberately the det(M) class table, not the per-pair
                    # table SecondOrder uses since issue #297: nothing pins
                    # third-order interpolation and the two paths are
                    # independent.
                    kw = build_nondiag_observable_kwargs(uc, sc)
                    mapping = kw.pop("_mapping")
                    third_order = cls(
                        value=third_ifcs,
                        folder=folder,
                        translation_support=support,
                        **kw,
                    )
                    return attach_snf_metadata(third_order, mapping)

                supercell = diagonal_supercell
                third_order = cls.from_supercell(
                    atoms=uc,
                    supercell=supercell,
                    grid_type="C",
                    value=third_ifcs,
                    folder=folder,
                    translation_support=support,
                )

            case 'gpumd':
                from kaldo.interfaces import gpumd_io
                meta = gpumd_io.read_gpumd_fc(folder)
                fc3 = meta['fc3']
                if third_energy_threshold > 0.:
                    mask = np.abs(fc3.data) > third_energy_threshold
                    fc3 = COO(fc3.coords[:, mask], fc3.data[mask], shape=fc3.shape)
                third_order = cls.from_supercell(
                    atoms=meta['atoms'],
                    grid_type=meta['grid_order'],
                    supercell=meta['third_supercell'],
                    value=fc3.astype(np.float64),
                    folder=folder,
                )

            case _:
                logging.error('Third order format not recognized: ' + str(format))
                raise ValueError

        return third_order

    def save(self, filename='THIRD', format='sparse', min_force=1e-6):
        """Export a compact periodic IFC3 tensor in a legacy file format.

        The existing ESKM and sparse/numpy formats do not serialize an
        arbitrary ``TranslationSupport``. Export is therefore permitted only
        when the tensor axes exactly match the physical compact support;
        literal file translations or Wigner--Seitz-expanded tensors would be
        irreversibly mislabelled and are rejected.

        Parameters
        ----------
        filename : str
            ESKM output name. Sparse/numpy output retains the historical
            fixed filenames in ``self.folder``.
        format : {"eskm", "sparse", "numpy"}
            Legacy target representation.
        min_force : float
            Norm threshold used when writing ESKM text blocks.
        """
        if format in ("eskm", "sparse", "numpy"):
            compact_support = TranslationSupport.periodic(self.supercell_grid)
            if self.translation_support.provenance != "periodic" or not np.array_equal(
                self.translation_support.translations,
                compact_support.translations,
            ):
                raise ValueError(
                    f"format={format!r} cannot preserve this IFC3 translation "
                    "support; export requires compact periodic axes"
                )
        folder = self.folder
        filename = folder + '/' + filename
        n_atoms = self.atoms.positions.shape[0]
        match format:
            case 'eskm':
                logging.info('Exporting third in eskm format')
                n_replicated_atoms = n_atoms * self.n_translations
                tenjovermoltoev = 10 * units.J / units.mol
                third = self.value.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
                with open(filename, 'w') as out_file:
                    for i in range(n_atoms):
                        for alpha in range(3):
                            for j in range(n_replicated_atoms):
                                for beta in range(3):
                                    value = third[i, alpha, j, beta].todense()
                                    mask = np.argwhere(np.linalg.norm(value, axis=1) > min_force)
                                    if mask.any():
                                        for k in mask:
                                            k = k[0]
                                            out_file.write("{:5d} ".format(i + 1))
                                            out_file.write("{:5d} ".format(alpha + 1))
                                            out_file.write("{:5d} ".format(j + 1))
                                            out_file.write("{:5d} ".format(beta + 1))
                                            out_file.write("{:5d} ".format(k + 1))
                                            for gamma in range(3):
                                                out_file.write(' {:16.6f}'.format(third[i, alpha, j, beta, k, gamma]))
                                            out_file.write('\n')
                logging.info('Done exporting third.')
            case 'sparse' | 'numpy':
                config_file = os.path.join(folder, REPLICATED_ATOMS_THIRD_FILE)
                ase.io.write(config_file, self.replicated_atoms, format='extxyz')

                save_npz(
                    folder + "/" + THIRD_ORDER_FILE_SPARSE,
                    self.value.reshape(
                        (
                            n_atoms * 3 * self.n_translations * n_atoms * 3,
                            self.n_translations * n_atoms * 3,
                        )
                    ).to_scipy_sparse(),
                )
            case _:
                super(ThirdOrder, self).save(filename, format)



    def calculate(self, calculator=None, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False,
                  n_workers=1, scratch_dir=None, keep_scratch=False, jat_flush_every=50, use_symmetry=False,
                  symprec=1e-5, symmetrize=True):
        """Calculate the third order force constants.

        This is the method typically reached through ``fc.third.calculate(...)``.
        It can load an existing stored result from ``self.folder`` when
        ``is_storing`` is enabled, or compute the anharmonic force constants
        directly from finite-difference force evaluations.

        See the *Parallel runs with ML calculators* section of the
        ForceConstants documentation for the recommended pattern when
        running torch-based calculators (Orb, MACE, MatterSim, CPUNEP) in
        parallel: define a no-arg factory function at module top level
        and pass it (without parentheses) as ``calculator``.

        Parameters
        ----------
        calculator : callable or ASE Calculator instance
            For serial runs, pass an ASE Calculator instance (the existing
            kaldo idiom). For parallel runs, pass a callable that returns
            a fresh ASE Calculator: a class with a no-arg constructor, a
            top-level factory function, ``functools.partial``, etc. Each
            worker invokes the callable once to build its own isolated
            calculator::

                from ase.calculators.emt import EMT
                calculator=EMT

            If None, replicated_atoms must already have a calculator attached.
        delta_shift : float
            Finite-difference displacement in Angstrom. The default ``1e-4``
            is tuned for analytical calculators (EMT, LAMMPS). ML potentials
            in float32 (Orb, MACE, MatterSim, ...) need ``1e-2`` or larger
            because float32 force noise (~1e-7 eV/Å) divided by a tiny
            delta produces FC noise that swamps the physics. A warning
            fires when ``delta_shift < 1e-2`` and the calculator looks
            ML-based.
            Default: 1e-4
        n_workers : int or None
            Number of parallel worker processes. ``1`` runs serially.
            ``None`` uses all available CPUs. Each worker is capped to one
            OpenMP / MKL / OpenBLAS thread so calculators with internal
            multithreading (PyNEP, torch CPU, numpy+MKL) don't oversubscribe.
            Override by setting ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` in
            the environment before invoking.
            Default: 1 (serial)
        scratch_dir : str or None
            Directory for scratch chunk files written during calculation to keep
            peak memory low. Pass an explicit path to override. Pass an
            empty string ``''`` to disable scratch files and fall back to
            in-memory accumulation.
            Default: ``{folder}/third_order`` when ``self.folder`` is set,
            ``n_workers > 1``, and ``use_symmetry=False``. With
            ``use_symmetry=True`` the auto-default is suppressed (the two
            modes are mutually incompatible — see the ``use_symmetry``
            docstring below).
        keep_scratch : bool
            If True, scratch files are kept after assembly.
            Default: False
        jat_flush_every : int
            Number of jat iterations each worker buffers before flushing to disk.
            Smaller values use less memory at the cost of more I/O. Default 50.
        use_symmetry : bool, optional
            If True, use the crystal spacegroup to reduce the number of
            atom pairs (i, jat) computed by the FD method. Only spacegroup
            operations compatible with the supercell shape are used (e.g.
            an in-plane subgroup for slab supercells). Requires a
            diagonal integer supercell expansion. Not compatible with
            ``scratch_dir`` — pass ``scratch_dir=None`` (the default)
            when enabling.
            Default: False
        symprec : float, optional
            precision for symmetry using spglib.
            Default: 1e-5
        symmetrize : bool, optional
            If True, project freshly computed force constants onto the
            space-group-invariant subspace (full-group average). Exact
            force constants are a fixed point, so this only removes
            finite-difference noise and symmetry violations from
            potentials that do not exactly respect the crystal symmetry
            (e.g. rotationally unconstrained ML potentials). Skipped
            with a warning when the symmetry analysis is unavailable.
            Force constants loaded from ``self.folder`` are never
            re-projected. Default: True
        """
        if is_parallel(n_workers):
            validate_parallel_calculator(calculator, method='ThirdOrder.calculate')
        maybe_warn_ml_delta_shift(calculator, delta_shift, method='ThirdOrder.calculate')
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        # Attach the calculator instance to replicated_atoms once and skip the
        # per-atom rebind in _compute_iat_third. Some calculator libraries
        # require a calculator to stay bound to a single atoms object.
        if n_workers == 1 and calculator is not None and not callable(calculator):
            replicated_atoms.calc = calculator
            worker_calculator = None
        else:
            worker_calculator = calculator
        # Auto-resolve the default scratch directory only for parallel runs;
        # serial stays in memory to avoid creating unexpected directories.
        # use_symmetry is incompatible with scratch_dir (calculate_third
        # raises ValueError on the combo), so don't auto-assign in that case.
        if (scratch_dir is None and self.folder and is_parallel(n_workers)
                and not use_symmetry):
            scratch_dir = os.path.join(self.folder, 'third_order')
        elif scratch_dir == '':
            scratch_dir = None
        if is_storing:
            try:
                self.value = ThirdOrder.load(folder=self.folder, supercell=self.supercell).value

            except FileNotFoundError:
                logging.info('Third order not found. Calculating.')
                self.value = calculate_third(atoms,
                                             replicated_atoms,
                                             delta_shift,
                                             distance_threshold=distance_threshold,
                                             is_verbose=is_verbose,
                                             n_workers=n_workers,
                                             calculator=worker_calculator,
                                             scratch_dir=scratch_dir,
                                             keep_scratch=keep_scratch,
                                             jat_flush_every=jat_flush_every,
                                             use_symmetry=use_symmetry,
                                             symprec=symprec)
                if symmetrize:
                    self.value = try_symmetrize_ifc(3, self.value, atoms, self.supercell, symprec)
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')
            else:
                logging.info('Reading stored third')
        else:
            self.value = calculate_third(atoms,
                                         replicated_atoms,
                                         delta_shift,
                                         distance_threshold=distance_threshold,
                                         is_verbose=is_verbose,
                                         n_workers=n_workers,
                                         calculator=worker_calculator,
                                         scratch_dir=scratch_dir,
                                         keep_scratch=keep_scratch,
                                         jat_flush_every=jat_flush_every,
                                         use_symmetry=use_symmetry,
                                         symprec=symprec)
            if symmetrize:
                self.value = try_symmetrize_ifc(3, self.value, atoms, self.supercell, symprec)
            if is_storing:
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')

    def symmetrize(self, symprec=1e-5):
        """Project the stored force constants onto the space-group-invariant subspace.

        See kaldo.controllers.displacement.symmetrize_ifc_third. Diagonal
        supercells only.
        """
        from kaldo.controllers.displacement import symmetrize_ifc_third
        if getattr(self, '_snf_mapping', None) is not None:
            # See SecondOrder.symmetrize: the SNF-linearized supercell would
            # silently symmetrize against the wrong replica lattice.
            raise NotImplementedError(
                'symmetrize() supports diagonal supercells only; this observable '
                'was loaded on a non-diagonal (SNF) replica mapping.'
            )
        self.value = symmetrize_ifc_third(self.value, self.atoms, self.supercell, symprec)

    def __str__(self):
        return 'third'
