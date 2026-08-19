"""Real-material regressions for non-diagonal TDEP translation support.

The literal Fourier oracle below deliberately does not call kALDo's TDEP
parser. This separates two questions: whether the loader preserves every
file-provided translation, and whether those force constants reproduce the
external ALAMODE spectrum.
"""

from __future__ import annotations

import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import units

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ


REFERENCE_ROOT = (
    Path(__file__).parent / "data" / "input" / "tdep-nondiagonal-reference"
)
THZ_TO_CM = 33.3564095198152

# These bounds sit above the independently measured interpolation/reference
# floors but far below the historical failures (41--237 cm^-1).
REFERENCE_TOLERANCE_CM = {
    "mp-7": 1.0,
    "mp-1000": 0.01,
    "mp-9947": 0.05,
    "mp-1221485": 0.10,
}


def _read_literal_tdep_ifc2(path, n_atoms):
    """Read raw ``(i, j, R, Phi)`` records without kALDo loader helpers."""
    records = []
    with path.open() as stream:
        file_n_atoms = int(stream.readline().split()[0])
        assert file_n_atoms == n_atoms
        stream.readline()  # cutoff
        for atom_i in range(n_atoms):
            n_neighbors = int(stream.readline().split()[0])
            for _ in range(n_neighbors):
                atom_j = int(stream.readline().split()[0]) - 1
                translation_float = np.asarray(
                    stream.readline().split(), dtype=float
                )
                translation = np.rint(translation_float).astype(int)
                np.testing.assert_allclose(
                    translation_float, translation, rtol=0.0, atol=1.0e-8
                )
                tensor = np.asarray(
                    [stream.readline().split() for _ in range(3)], dtype=float
                )
                records.append((atom_i, atom_j, translation, tensor))
    return records


def _literal_dynamical_matrix(primitive, records, q_point):
    """Return the mass-weighted direct file-record Fourier sum."""
    masses = primitive.get_masses()
    n_atoms = len(primitive)
    dynamical = np.zeros((n_atoms, 3, n_atoms, 3), dtype=np.complex128)
    for atom_i, atom_j, translation, tensor in records:
        phase = np.exp(2j * np.pi * np.dot(q_point, translation))
        dynamical[atom_i, :, atom_j, :] += (
            phase * tensor / np.sqrt(masses[atom_i] * masses[atom_j])
        )
    return (
        dynamical.reshape(3 * n_atoms, 3 * n_atoms)
        * units.mol
        / (10 * units.J)
    )
@pytest.mark.parametrize("case", tuple(REFERENCE_TOLERANCE_CM))
def test_real_nondiagonal_tdep_frequencies(case):
    """Imported IFC2 must match both literal and external frequency oracles."""
    folder = REFERENCE_ROOT / case
    metadata = json.loads((folder / "expected.json").read_text())
    matrix = np.asarray(metadata["supercell_matrix"], dtype=int)
    primitive = ase.io.read(folder / "infile.ucposcar", format="vasp")
    records = _read_literal_tdep_ifc2(
        folder / "infile.forceconstant", len(primitive)
    )
    forceconstants = ForceConstants.from_folder(
        str(folder),
        format="tdep",
        supercell_matrix=matrix,
        only_second=True,
    )

    second = forceconstants.second
    np.testing.assert_array_equal(second.supercell_grid.matrix, matrix)
    assert second.translation_support.provenance == "file"

    reference_errors = []
    for q_point, expected_cm in zip(
        metadata["q_points_frac"], metadata["expected_freqs_cminv"]
    ):
        q_point = np.asarray(q_point, dtype=float)
        harmonic = HarmonicWithQ(
            q_point=q_point,
            second=second,
            ifc_interpolation="auto",
            storage="memory",
        )
        actual = np.sort(np.asarray(harmonic.frequency).ravel())
        actual_dynamical, _ = harmonic._get_ifc_interpolation_plan().matrices(
            q_point
        )
        literal_dynamical = _literal_dynamical_matrix(
            primitive, records, q_point
        )
        np.testing.assert_allclose(
            actual_dynamical,
            literal_dynamical,
            rtol=1.0e-13,
            atol=1.0e-11,
        )
        reference_errors.append(
            np.max(np.abs(actual * THZ_TO_CM - np.sort(expected_cm)))
        )

    assert max(reference_errors) < REFERENCE_TOLERANCE_CM[case], (
        f"{case} differs from its ALAMODE reference by "
        f"{max(reference_errors):.6f} cm^-1"
    )


def test_underresolved_case_retains_repeated_periodic_classes():
    """mp-1221485 must not collapse its long-range records while loading."""
    folder = REFERENCE_ROOT / "mp-1221485"
    metadata = json.loads((folder / "expected.json").read_text())
    forceconstants = ForceConstants.from_folder(
        str(folder),
        format="tdep",
        supercell_matrix=np.asarray(metadata["supercell_matrix"], dtype=int),
        only_second=True,
    )
    support = forceconstants.second.translation_support
    assert support.size > support.supercell.size
    assert len(np.unique(support.class_ids)) < support.size
