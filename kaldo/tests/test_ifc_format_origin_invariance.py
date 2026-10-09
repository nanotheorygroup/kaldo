"""Rigid-origin invariance contracts for every public IFC input route.

Moving the unit-cell origin and wrapping the basis is a gauge change, not a
change of crystal.  Each loader must therefore feed IFC interpolation enough
translation information to preserve the commensurate spectrum and the
basis-invariant RTA conductivity.  The direct finite-displacement/calculator
route is covered separately by ``test_ifc_origin_invariance_lj.py``.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil

import ase.io
import numpy as np
import pytest
from scipy.sparse import save_npz
from sparse import COO

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.grid import TranslationSupport
from kaldo.observables.thirdorder import _coalesced_coo, _rank8_ifc3
from kaldo.phonons import Phonons

TESTS = Path(__file__).parent
ORIGIN_SHIFT = np.array([0.8, 0.1, 0.1])


@dataclass(frozen=True)
class _FormatCase:
    name: str
    folder: str
    supercell: tuple[int, int, int]
    third_supercell: tuple[int, int, int]
    expected_harmonic_mode: str
    expected_third_mode: str


FORMAT_CASES = (
    _FormatCase(
        "numpy", "generated/numpy", (3, 3, 3), (3, 3, 3), "wigner-seitz", "wigner-seitz"
    ),
    _FormatCase(
        "eskm", "si-crystal", (3, 3, 3), (3, 3, 3), "wigner-seitz", "wigner-seitz"
    ),
    _FormatCase(
        "lammps",
        "generated/lammps",
        (3, 3, 3),
        (3, 3, 3),
        "wigner-seitz",
        "wigner-seitz",
    ),
    _FormatCase(
        "vasp-sheng",
        "si-crystal/vasp",
        (5, 5, 5),
        (3, 3, 3),
        "wigner-seitz",
        "file",
    ),
    _FormatCase(
        "qe-sheng", "si-crystal/qe", (3, 3, 3), (3, 3, 3), "wigner-seitz", "file"
    ),
    _FormatCase(
        "vasp-d3q",
        "ge-crystal/vasp-d3q",
        (5, 5, 5),
        (3, 3, 3),
        "wigner-seitz",
        "periodic",
    ),
    _FormatCase(
        "qe-d3q", "ge-crystal/d3q", (5, 5, 5), (3, 3, 3), "wigner-seitz", "periodic"
    ),
    _FormatCase(
        "hiphive",
        "si-crystal/hiphive",
        (3, 3, 3),
        (3, 3, 3),
        "wigner-seitz",
        "wigner-seitz",
    ),
    _FormatCase("tdep", "si-tdep", (5, 5, 5), (5, 5, 5), "file", "file"),
    _FormatCase(
        "gpumd",
        "si-crystal/gpumd",
        (3, 3, 3),
        (3, 3, 3),
        "wigner-seitz",
        "wigner-seitz",
    ),
)


ALIASES = (
    ("shengbte", "vasp-sheng", "si-crystal/vasp", (5, 5, 5), (3, 3, 3)),
    ("shengbte-qe", "qe-sheng", "si-crystal/qe", (3, 3, 3), (3, 3, 3)),
    ("shengbte-d3q", "qe-d3q", "ge-crystal/d3q", (5, 5, 5), (3, 3, 3)),
)


def _write_generated_formats(root):
    """Derive numpy and lammps fixtures from the committed ESKM Si data."""
    source_folder = TESTS / "si-crystal"
    source = ForceConstants.from_folder(
        folder=str(source_folder),
        supercell=(3, 3, 3),
        third_supercell=(3, 3, 3),
        format="eskm",
    )

    numpy_folder = root / "numpy"
    numpy_folder.mkdir(parents=True)
    ase.io.write(
        numpy_folder / "replicated_atoms.xyz",
        source.second.replicated_atoms,
        format="extxyz",
    )
    ase.io.write(
        numpy_folder / "replicated_atoms_third.xyz",
        source.third.replicated_atoms,
        format="extxyz",
    )
    np.save(numpy_folder / "second.npy", source.second.value)
    third = source.third.value
    n_atoms = len(source.third.atoms)
    n_replicas = source.third.n_replicas
    flattened = third.reshape(
        (
            n_atoms * 3 * n_replicas * n_atoms * 3,
            n_replicas * n_atoms * 3,
        )
    )
    save_npz(numpy_folder / "third.npz", flattened.to_scipy_sparse())

    lammps_folder = root / "lammps"
    lammps_folder.mkdir(parents=True)
    ase.io.write(
        lammps_folder / "replicated_atoms.xyz",
        source.second.replicated_atoms,
        format="extxyz",
    )
    shutil.copyfile(source_folder / "Dyn.form", lammps_folder / "Dyn.form")
    shutil.copyfile(source_folder / "THIRD", lammps_folder / "THIRD")


@pytest.fixture(scope="session")
def generated_format_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("origin-invariant-formats")
    _write_generated_formats(root)
    return root


def _case_folder(case, generated_format_root):
    if case.folder.startswith("generated/"):
        return generated_format_root / case.folder.split("/", 1)[1]
    return TESTS / case.folder


def _load(case, generated_format_root):
    return ForceConstants.from_folder(
        folder=str(_case_folder(case, generated_format_root)),
        supercell=case.supercell,
        third_supercell=case.third_supercell,
        format=case.name,
    )


def _translated_support(observable, face_offsets):
    """Build the exact translation support after changing basis gauge."""
    source = observable.translation_support
    if source.provenance == "periodic":
        return source
    translations = {
        tuple(translation + face_offsets[atom_j] - face_offsets[atom_i])
        for translation in source.translations
        for atom_i in range(len(observable.atoms))
        for atom_j in range(len(observable.atoms))
    }
    ordered = sorted(translations, key=lambda vector: (vector != (0, 0, 0), vector))
    return TranslationSupport(
        np.asarray(ordered, dtype=np.int64),
        source.supercell,
        provenance=source.provenance,
    )


def _translation_slot(support, translation):
    """Resolve an exact literal translation or a compact periodic class."""
    if support.provenance == "periodic":
        class_id = support.supercell.class_id(translation)
        matches = np.flatnonzero(support.class_ids == class_id)
    else:
        matches = np.flatnonzero(np.all(support.translations == translation, axis=1))
    assert len(matches) == 1
    return int(matches[0])


def _pair_translation_slots(
    observable,
    face_offsets,
    target_support,
):
    """Map each ``(source R, i, j)`` tuple to its transformed slot."""
    source_support = observable.translation_support
    n_atoms = len(observable.atoms)
    slots = np.empty((source_support.size, n_atoms, n_atoms), dtype=np.int64)
    for source_id, translation in enumerate(source_support.translations):
        for atom_i in range(n_atoms):
            for atom_j in range(n_atoms):
                slots[source_id, atom_i, atom_j] = _translation_slot(
                    target_support,
                    translation + face_offsets[atom_j] - face_offsets[atom_i],
                )
    return slots


def _regauge_second(second, face_offsets):
    """Apply R' = R + n_j - n_i to every IFC2 block."""
    source_support = second.translation_support
    target_support = _translated_support(second, face_offsets)
    target_slots = _pair_translation_slots(
        second,
        face_offsets,
        target_support,
    )
    source = np.asarray(second.value)
    target = np.zeros(
        source.shape[:3] + (target_support.size,) + source.shape[4:],
        dtype=source.dtype,
    )
    for atom_i in range(len(second.atoms)):
        for source_id, _ in enumerate(source_support.translations):
            for atom_j in range(len(second.atoms)):
                target_id = target_slots[source_id, atom_i, atom_j]
                target[:, atom_i, :, target_id, atom_j, :] += source[
                    :, atom_i, :, source_id, atom_j, :
                ]
    second.value = target
    second.translation_support = target_support
    second.n_translations = target_support.size
    # is_acoustic_sum=True (used by the QE/Sheng loaders) can materialize the
    # mass-scaled tensor during construction.  It is derived from ``value`` and
    # must not survive this deliberate test-only representation change.
    if hasattr(second, "_dynmat"):
        del second._dynmat


def _regauge_third(third, face_offsets):
    """Relabel both IFC3 partner translations under the wrapped basis."""
    source_support = third.translation_support
    target_support = _translated_support(third, face_offsets)
    source = _rank8_ifc3(
        third.value,
        len(third.atoms),
        source_support.size,
    )
    source = source if isinstance(source, COO) else COO.from_numpy(source)
    coords = np.asarray(source.coords, dtype=np.int64).copy()
    target_slots = _pair_translation_slots(third, face_offsets, target_support)
    old_j = coords[2].copy()
    old_k = coords[5].copy()
    coords[2] = target_slots[coords[2], coords[0], coords[3]]
    coords[5] = target_slots[coords[5], coords[0], coords[6]]
    moved_weight = np.sum(
        np.abs(source.data[(coords[2] != old_j) | (coords[5] != old_k)])
    )
    shape = list(source.shape)
    shape[2] = target_support.size
    shape[5] = target_support.size
    third.value = _coalesced_coo(coords, source.data, tuple(shape), source.dtype)
    third.translation_support = target_support
    third.n_translations = target_support.size
    return moved_weight


def _move_origin(forceconstants):
    """Wrap a loaded basis and consistently transform its IFC gauge."""
    crossed_faces = None
    for observable in (forceconstants.second, forceconstants.third):
        scaled = observable.atoms.get_scaled_positions(wrap=False)
        shifted = scaled + ORIGIN_SHIFT
        face_offsets = np.floor(shifted).astype(np.int64)
        if crossed_faces is None:
            crossed_faces = face_offsets
        if observable is forceconstants.second:
            _regauge_second(observable, face_offsets)
        else:
            moved_third_weight = _regauge_third(observable, face_offsets)
        observable.atoms.set_scaled_positions(np.mod(shifted, 1.0))
        observable.replicated_positions = (
            observable.replica_translations[:, np.newaxis, :]
            @ np.asarray(observable.atoms.cell)
            + observable.atoms.positions[np.newaxis, :, :]
        )
        observable._replicated_atoms = None

    # A common shift where every atom crosses the same faces is trivial.  Each
    # fixture must exercise an atom-pair-dependent change of cell representative.
    assert np.unique(crossed_faces, axis=0).shape[0] > 1
    assert moved_third_weight > 0.0


def _phonons(forceconstants, folder):
    return Phonons(
        forceconstants=forceconstants,
        kpts=(3, 3, 3),
        is_classic=False,
        temperature=300.0,
        third_bandwidth=0.5,
        include_isotopes=False,
        folder=str(folder),
        storage="memory",
    )


def _rta_tensor(phonons):
    tensor = Conductivity(
        phonons=phonons,
        method="rta",
        storage="memory",
    ).conductivity.sum(axis=0)
    assert np.isfinite(tensor).all()
    assert np.linalg.norm(tensor) > 0.0
    return tensor


@pytest.mark.parametrize("case", FORMAT_CASES, ids=lambda case: case.name)
def test_file_format_is_invariant_to_wrapped_crystal_origin(
    case,
    generated_format_root,
    tmp_path,
):
    """Every canonical IFC loader preserves spectrum and RTA conductivity."""
    reference_fc = _load(case, generated_format_root)
    translated_fc = _load(case, generated_format_root)
    _move_origin(translated_fc)

    reference = _phonons(reference_fc, tmp_path / "reference")
    translated = _phonons(translated_fc, tmp_path / "translated")
    assert reference.ifc_interpolation_resolved == case.expected_harmonic_mode
    assert translated.ifc_interpolation_resolved == case.expected_harmonic_mode
    assert (
        reference.forceconstants.third.get_interpolation("auto").resolved_mode
        == case.expected_third_mode
    )
    assert (
        translated.forceconstants.third.get_interpolation("auto").resolved_mode
        == case.expected_third_mode
    )

    np.testing.assert_allclose(
        translated.frequency[1:],
        reference.frequency[1:],
        rtol=0.0,
        atol=2e-13,
    )
    # Diagonalize the near-zero Gamma translations separately: regenerated or
    # regauged matrices can rotate this numerically singular subspace without
    # changing any physical mode.
    np.testing.assert_allclose(
        translated.frequency[0],
        reference.frequency[0],
        rtol=0.0,
        atol=2e-7,
    )
    translated_kappa = _rta_tensor(translated)
    reference_kappa = _rta_tensor(reference)
    # These committed loader fixtures are cubic and contain exactly degenerate
    # branches.  Their unsymmetrized Cartesian off-diagonal components depend
    # weakly on the numerical eigenvector basis; the rotationally invariant
    # trace is the rigorous scalar conductivity contract for these datasets.
    # rtol calibrated cross-platform: on macOS arm64 (Accelerate BLAS) the
    # degenerate-subspace eigenbases drift the trace by up to 0.34% (tdep) /
    # 0.07% (hiphive, gpumd) between the two gauge-equivalent runs; linux
    # x86 stays below 5e-4. The historical origin-dependence bug moved this
    # trace by 20-60%, so 1e-2 keeps the assertion discriminating.
    np.testing.assert_allclose(
        np.trace(translated_kappa) / 3.0,
        np.trace(reference_kappa) / 3.0,
        rtol=1e-2,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "alias,canonical,folder,supercell,third_supercell",
    ALIASES,
    ids=[item[0] for item in ALIASES],
)
def test_legacy_format_alias_has_identical_translation_contract(
    alias,
    canonical,
    folder,
    supercell,
    third_supercell,
):
    """Accepted legacy names must load the canonical IFC representation."""
    kwargs = dict(
        folder=str(TESTS / folder),
        supercell=supercell,
        third_supercell=third_supercell,
    )
    alias_fc = ForceConstants.from_folder(format=alias, **kwargs)
    canonical_fc = ForceConstants.from_folder(format=canonical, **kwargs)

    np.testing.assert_allclose(alias_fc.second.value, canonical_fc.second.value)
    np.testing.assert_array_equal(
        alias_fc.second.translation_support.translations,
        canonical_fc.second.translation_support.translations,
    )
    np.testing.assert_array_equal(
        alias_fc.third.translation_support.translations,
        canonical_fc.third.translation_support.translations,
    )
    assert alias_fc.third.value.shape == canonical_fc.third.value.shape
    alias_nnz = (
        alias_fc.third.value.nnz
        if hasattr(alias_fc.third.value, "nnz")
        else np.count_nonzero(alias_fc.third.value)
    )
    canonical_nnz = (
        canonical_fc.third.value.nnz
        if hasattr(canonical_fc.third.value, "nnz")
        else np.count_nonzero(canonical_fc.third.value)
    )
    assert alias_nnz == canonical_nnz
