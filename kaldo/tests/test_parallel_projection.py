"""Tests for parallel per-k-point projection in _project_crystal.

Validates that parallel projection (n_workers > 1) produces numerically
identical sparse_phase and sparse_potential tensors as the serial path.
"""

import numpy as np
import pytest
import tensorflow as tf
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


@pytest.fixture(scope="module")
def si_forceconstants():
    """Load Si crystal force constants (shared across all tests in module)."""
    return ForceConstants.from_folder("kaldo/tests/si-crystal", supercell=(3, 3, 3), format="eskm")


@pytest.fixture(scope="module")
def serial_phonons(si_forceconstants):
    """Compute projection serially (baseline)."""
    return Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_serial_projection",
        n_workers=1,
    )


def test_parallel_projection_matches_serial(si_forceconstants, serial_phonons):
    """Parallel projection (2 workers) must match serial at rtol=1e-7."""
    parallel = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_parallel_projection",
        n_workers=2,
    )

    phase_s = serial_phonons.sparse_phase
    phase_p = parallel.sparse_phase
    pot_s = serial_phonons.sparse_potential
    pot_p = parallel.sparse_potential

    assert len(phase_s) == len(phase_p), "sparse_phase length mismatch"
    assert len(pot_s) == len(pot_p), "sparse_potential length mismatch"

    for nu in range(len(phase_s)):
        for ip in range(2):
            ps = phase_s[nu][ip]
            pp = phase_p[nu][ip]
            if ps is None:
                assert pp is None, f"Phase mismatch at nu={nu}, is_plus={ip}: serial is None, parallel is not"
                continue
            assert pp is not None, f"Phase mismatch at nu={nu}, is_plus={ip}: serial has data, parallel is None"
            np.testing.assert_allclose(
                ps.values.numpy(), pp.values.numpy(), rtol=1e-7,
                err_msg=f"Phase values differ at nu={nu}, is_plus={ip}",
            )
            np.testing.assert_array_equal(
                ps.indices.numpy(), pp.indices.numpy(),
                err_msg=f"Phase indices differ at nu={nu}, is_plus={ip}",
            )

            # Also check potential
            ss = pot_s[nu][ip]
            sp = pot_p[nu][ip]
            if ss is None:
                assert sp is None
                continue
            np.testing.assert_allclose(
                ss.values.numpy(), sp.values.numpy(), rtol=1e-7,
                err_msg=f"Potential values differ at nu={nu}, is_plus={ip}",
            )


def test_projection_output_dir_with_resume(si_forceconstants, serial_phonons, tmp_path):
    """Disk-based projection with resume produces correct results."""
    output_dir = str(tmp_path / "projection_output")

    # Full run with output to disk
    p1 = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_output_dir_proj",
        n_workers=2,
        projection_output_dir=output_dir,
    )
    phase_disk = p1.sparse_phase

    # Verify matches serial
    phase_s = serial_phonons.sparse_phase
    for nu in range(len(phase_s)):
        for ip in range(2):
            ps = phase_s[nu][ip]
            pd = phase_disk[nu][ip]
            if ps is None:
                assert pd is None
                continue
            np.testing.assert_allclose(
                ps.values.numpy(), pd.values.numpy(), rtol=1e-7,
                err_msg=f"Disk phase values differ at nu={nu}, is_plus={ip}",
            )


def test_n_workers_zero_raises():
    """n_workers=0 should raise ValueError."""
    fc = ForceConstants.from_folder("kaldo/tests/si-crystal", supercell=(3, 3, 3), format="eskm")
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        Phonons(forceconstants=fc, kpts=(3, 3, 3), temperature=300, n_workers=0)


def test_sparse_worker_and_storage_payload_round_trip():
    """Workers and NumPy persistence share one lossless sparse boundary."""
    phase = tf.SparseTensor(
        indices=[[0, 1], [2, 0]],
        values=[1.25, -0.5],
        dense_shape=[3, 2],
    )
    potential = tf.SparseTensor(
        indices=phase.indices,
        values=[0.75, 2.5],
        dense_shape=phase.dense_shape,
    )
    payload = Phonons._sparse_tensor_to_numpy(phase)
    restored = Phonons._numpy_to_sparse_tensor(payload)

    np.testing.assert_array_equal(restored.indices.numpy(), phase.indices.numpy())
    np.testing.assert_allclose(restored.values.numpy(), phase.values.numpy())
    np.testing.assert_array_equal(
        restored.dense_shape.numpy(), phase.dense_shape.numpy()
    )

    phonons = object.__new__(Phonons)
    stored = phonons._convert_sparse_tensors_to_per_mu_arrays(
        [[phase, None]], [[potential, None]]
    )
    restored_phase, restored_potential = phonons._convert_per_mu_arrays_to_sparse_tensors(
        stored
    )
    np.testing.assert_array_equal(
        restored_phase[0][0].indices.numpy(), phase.indices.numpy()
    )
    np.testing.assert_allclose(
        restored_phase[0][0].values.numpy(), phase.values.numpy()
    )
    np.testing.assert_allclose(
        restored_potential[0][0].values.numpy(), potential.values.numpy()
    )
    assert restored_phase[0][1] is None
    assert restored_potential[0][1] is None
    assert Phonons._sparse_tensor_to_numpy(None) is None
    assert Phonons._numpy_to_sparse_tensor(None) is None
