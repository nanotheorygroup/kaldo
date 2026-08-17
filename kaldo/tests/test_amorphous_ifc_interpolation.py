"""Real-potential amorphous regression for pair-dependent IFC geometry.

The 32-atom Si cell is a deterministic Tersoff melt--quench. At Gamma the
dynamical matrix contains the zeroth IFC2 moment, while the Allen--Feldman
heat-flux operator contains its first Cartesian moment. This makes the same
fixture prove both why frequencies miss the image bug and why diffusivity and
QHGK conductivity require the corrected pair displacement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import spglib
from sparse import COO

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.grid import TranslationSupport
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.thirdorder import _rank8_ifc3
from kaldo.phonons import Phonons

FIXTURE = Path(__file__).parent / "data" / "input" / "amorphous-si-tersoff-32"
METADATA = json.loads((FIXTURE / "expected.json").read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_forceconstants():
    return ForceConstants.from_folder(
        folder=str(FIXTURE),
        format="numpy",
        supercell=(1, 1, 1),
        third_supercell=(1, 1, 1),
        is_acoustic_sum=True,
    )


def _move_compact_origin(forceconstants):
    """Wrap the large-cell basis while retaining its compact periodic classes."""
    shift = np.asarray(METADATA["origin_shift_fractional"], dtype=float)
    basis_shifts = None
    for observable in (forceconstants.second, forceconstants.third):
        scaled = observable.atoms.get_scaled_positions(wrap=False)
        shifted = scaled + shift
        face_offsets = np.floor(shifted).astype(np.int64)
        if basis_shifts is None:
            # tau' = tau + common_shift + basis_shift.  The common part drops
            # out of every pair, while basis_shift=-face_offset determines the
            # exact Fourier-gauge transformation.
            basis_shifts = -face_offsets
        observable.atoms.set_scaled_positions(np.mod(shifted, 1.0))
        observable.replicated_positions = observable.atoms.positions[np.newaxis, :, :]
        observable._replicated_atoms = None
    assert np.unique(basis_shifts, axis=0).shape[0] > 1
    return basis_shifts


def _harmonic(forceconstants, mode):
    return HarmonicWithQ(
        q_point=np.zeros(3),
        second=forceconstants.second,
        storage="memory",
        is_amorphous=True,
        ifc_interpolation=mode,
    )


def _flux(harmonic):
    return np.stack((harmonic._sij_x, harmonic._sij_y, harmonic._sij_z))


def _phonons(forceconstants, mode):
    return Phonons(
        forceconstants=forceconstants,
        kpts=(1, 1, 1),
        temperature=300.0,
        is_classic=False,
        third_bandwidth=0.05 / 4.135,
        broadening_shape="triangle",
        storage="memory",
        ifc_interpolation=mode,
    )


def _qhgk(phonons):
    conductivity = Conductivity(
        phonons=phonons,
        method="qhgk",
        storage="memory",
    )
    tensor = conductivity.conductivity.sum(axis=0)
    assert np.isfinite(tensor).all()
    assert np.linalg.norm(tensor) > 0.0
    return tensor, np.asarray(conductivity.diffusivity)


def _boundary_weight(forceconstants):
    """Return independent cubic-cell IFC2/IFC3 boundary L1 statistics."""
    atoms = forceconstants.atoms
    fractional = atoms.get_scaled_positions(wrap=False)
    pair_shift = -np.rint(
        fractional[np.newaxis, :, :] - fractional[:, np.newaxis, :]
    ).astype(np.int64)
    boundary_pair = np.any(pair_shift != 0, axis=-1)

    second = np.asarray(forceconstants.second.dynmat)[0, :, :, 0, :, :]
    second_block_weight = np.sum(np.abs(second), axis=(1, 3))
    second_total = float(np.sum(second_block_weight))
    second_boundary = float(np.sum(second_block_weight[boundary_pair]))

    third = forceconstants.third
    rank8 = _rank8_ifc3(third.value, len(atoms), third.n_translations)
    sparse = rank8 if isinstance(rank8, COO) else COO.from_numpy(np.asarray(rank8))
    coords = np.asarray(sparse.coords)
    boundary_entry = (
        boundary_pair[coords[0], coords[3]] | boundary_pair[coords[0], coords[6]]
    )
    third_abs = np.abs(np.asarray(sparse.data))
    third_total = float(np.sum(third_abs))
    third_boundary = float(np.sum(third_abs[boundary_entry]))
    return {
        "ifc2_nonzero_boundary_pairs": int(
            np.count_nonzero((second_block_weight > 0.0) & boundary_pair)
        ),
        "ifc2_boundary_l1_fraction": second_boundary / second_total,
        "ifc3_nonzero_boundary_entries": int(np.count_nonzero(boundary_entry)),
        "ifc3_boundary_l1_fraction": third_boundary / third_total,
    }


def _fourier_tensor(interpolation, qj, qk, n_atoms):
    """Evaluate the two translation sums without production projection code."""
    value = interpolation.value
    sparse = value if isinstance(value, COO) else COO.from_numpy(np.asarray(value))
    coords = np.asarray(sparse.coords)
    phase_j = interpolation.support.phases(np.asarray([qj]))[0, coords[2]]
    phase_k = interpolation.support.phases(np.asarray([qk]))[0, coords[5]]
    tensor_coords = coords[[0, 1, 3, 4, 6, 7]]
    return COO(
        tensor_coords,
        np.asarray(sparse.data) * phase_j * phase_k,
        shape=(n_atoms, 3, n_atoms, 3, n_atoms, 3),
        has_duplicates=True,
        sorted=False,
    )


def _expand_literal_gamma_support(forceconstants):
    """Split real IFC3 entries over two same-class literal translations.

    The two tensors have exactly the same Gamma contraction, but the expanded
    representation has ``n_translations=2`` while the physical BvK cell has
    ``n_replicas=1``.  This catches the historical assumption that the two
    dimensions were interchangeable in the amorphous projection.
    """
    third = forceconstants.third
    source = _rank8_ifc3(third.value, len(third.atoms), third.n_translations)
    source = source if isinstance(source, COO) else COO.from_numpy(np.asarray(source))
    coords = np.asarray(source.coords)
    expanded_coords = np.concatenate((coords, coords), axis=1)
    expanded_coords[2, coords.shape[1] :] = 1
    expanded_coords[5, coords.shape[1] :] = 1
    shape = list(source.shape)
    shape[2] = shape[5] = 2
    third.value = COO(
        expanded_coords,
        np.concatenate((0.5 * source.data, 0.5 * source.data)),
        shape=tuple(shape),
        has_duplicates=False,
        sorted=False,
    )
    third.translation_support = TranslationSupport(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        third.supercell_grid,
        provenance="file",
    )
    third.n_translations = 2


def test_fixture_provenance_and_boundary_weight():
    """The retained real-potential data must exercise both boundary legs."""
    for name, expected in METADATA["sha256"].items():
        assert _sha256(FIXTURE / name) == expected
    assert METADATA["n_atoms"] == 32
    assert METADATA["density_kg_m3"] == pytest.approx(2330.0, rel=2e-4)
    assert METADATA["max_force_eV_per_A"] < 2.0e-5
    assert METADATA["cell_length_angstrom"] > 2 * 3.2

    forceconstants = _load_forceconstants()
    scaled = forceconstants.atoms.get_scaled_positions(wrap=False)
    assert np.all(scaled >= 0.0)
    assert np.all(scaled < 1.0)
    dataset = spglib.get_symmetry_dataset(
        (
            np.asarray(forceconstants.atoms.cell),
            forceconstants.atoms.get_scaled_positions(),
            forceconstants.atoms.numbers,
        ),
        symprec=1e-5,
    )
    assert dataset.number == 1

    frequencies = np.sort(_harmonic(forceconstants, "auto").frequency.ravel())
    # Finite-difference antisymmetry leaves the three translations within
    # 0.008 THz of zero; the first physical mode is separated by 1.89 THz.
    assert np.max(np.abs(frequencies[:3])) < 0.01
    assert frequencies[3] > 1.0

    actual = _boundary_weight(forceconstants)
    expected = METADATA["boundary_weight"]
    assert actual["ifc2_nonzero_boundary_pairs"] > 0
    assert actual["ifc3_nonzero_boundary_entries"] > 0
    for key in actual:
        if key.endswith("fraction"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-14)
        else:
            assert actual[key] == expected[key]


def test_gamma_frequencies_hide_but_heat_flux_exposes_legacy_image_bug():
    """Zeroth moments agree while the wrong first moment is origin dependent."""
    reference_fc = _load_forceconstants()
    shifted_fc = _load_forceconstants()
    _move_compact_origin(shifted_fc)

    reference = _harmonic(reference_fc, "auto")
    shifted = _harmonic(shifted_fc, "auto")
    legacy_reference = _harmonic(reference_fc, "periodic")
    legacy_shifted = _harmonic(shifted_fc, "periodic")

    for other in (shifted, legacy_reference, legacy_shifted):
        np.testing.assert_allclose(
            other.frequency, reference.frequency, rtol=0, atol=2e-11
        )
    np.testing.assert_allclose(_flux(shifted), _flux(reference), rtol=2e-11, atol=2e-11)

    legacy_change = np.linalg.norm(
        _flux(legacy_shifted) - _flux(legacy_reference)
    ) / np.linalg.norm(_flux(legacy_reference))
    assert legacy_change > 0.05
    np.testing.assert_allclose(
        legacy_change,
        METADATA["references"]["legacy_origin_flux_relative_change"],
        rtol=2e-10,
        atol=0.0,
    )


def test_ifc3_fourier_magnitude_is_origin_invariant_off_gamma():
    """Both IFC3 legs obey the exact atom-dependent Fourier-gauge phase."""
    reference = _load_forceconstants()
    shifted = _load_forceconstants()
    basis_shifts = _move_compact_origin(shifted)
    qj, qk = np.asarray(METADATA["ifc3_probe_q_points"], dtype=float)

    reference_tensor = _fourier_tensor(
        reference.third.get_interpolation("auto"), qj, qk, reference.n_atoms
    )
    shifted_tensor = _fourier_tensor(
        shifted.third.get_interpolation("auto"), qj, qk, shifted.n_atoms
    )
    np.testing.assert_array_equal(shifted_tensor.coords, reference_tensor.coords)
    tensor_coords = reference_tensor.coords
    atom_i, atom_j, atom_k = tensor_coords[0], tensor_coords[2], tensor_coords[4]
    expected_gauge = np.exp(
        2j
        * np.pi
        * (
            (basis_shifts[atom_i] - basis_shifts[atom_j]) @ qj
            + (basis_shifts[atom_i] - basis_shifts[atom_k]) @ qk
        )
    )
    assert np.any(np.abs(expected_gauge - 1.0) > 1e-3)
    np.testing.assert_allclose(
        shifted_tensor.data,
        expected_gauge * reference_tensor.data,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        shifted.third.gamma_contracted_value().todense(),
        reference.third.gamma_contracted_value().todense(),
        rtol=0,
        atol=0,
    )


def test_amorphous_projection_uses_literal_ifc3_support_not_replica_count():
    """A same-class S=2 IFC3 representation gives the same Gamma transport."""
    compact_fc = _load_forceconstants()
    literal_fc = _load_forceconstants()
    _expand_literal_gamma_support(literal_fc)
    assert literal_fc.third.n_replicas == 1
    assert literal_fc.third.n_translations == 2

    compact = _phonons(compact_fc, "auto")
    literal = _phonons(literal_fc, "auto")
    compact_tensor, compact_diffusivity = _qhgk(compact)
    literal_tensor, literal_diffusivity = _qhgk(literal)

    np.testing.assert_allclose(literal.bandwidth, compact.bandwidth, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        literal_diffusivity, compact_diffusivity, rtol=2e-11, atol=2e-13
    )
    np.testing.assert_allclose(literal_tensor, compact_tensor, rtol=2e-11, atol=2e-13)


def test_qhgk_transport_is_origin_invariant_and_legacy_route_is_not():
    """The complete user-visible amorphous transport result obeys the gauge."""
    reference_fc = _load_forceconstants()
    shifted_fc = _load_forceconstants()
    _move_compact_origin(shifted_fc)

    reference = _phonons(reference_fc, "auto")
    shifted = _phonons(shifted_fc, "auto")
    legacy_reference = _phonons(reference_fc, "periodic")
    legacy_shifted = _phonons(shifted_fc, "periodic")

    reference_tensor, reference_diffusivity = _qhgk(reference)
    shifted_tensor, shifted_diffusivity = _qhgk(shifted)
    legacy_reference_tensor, _ = _qhgk(legacy_reference)
    legacy_shifted_tensor, _ = _qhgk(legacy_shifted)

    np.testing.assert_allclose(
        shifted.bandwidth, reference.bandwidth, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        shifted_diffusivity, reference_diffusivity, rtol=2e-10, atol=2e-12
    )
    np.testing.assert_allclose(shifted_tensor, reference_tensor, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(
        reference_tensor,
        np.asarray(METADATA["references"]["qhgk_tensor_W_mK"]),
        rtol=2e-9,
        atol=2e-12,
    )

    reference_trace = float(np.trace(legacy_reference_tensor))
    shifted_trace = float(np.trace(legacy_shifted_tensor))
    legacy_change = abs(shifted_trace - reference_trace) / abs(reference_trace)
    assert legacy_change > 0.01
    np.testing.assert_allclose(
        legacy_change,
        METADATA["references"]["legacy_origin_qhgk_trace_relative_change"],
        rtol=0.0,
        # Parallel TensorFlow reductions move this diagnostic ratio by about
        # 2e-9 between otherwise identical runs; the invariant comparisons
        # above retain their much tighter tolerances.
        atol=3e-9,
    )
