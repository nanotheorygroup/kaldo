"""Tests for parallel per-k-point projection in _project_crystal.

Validates that parallel projection (n_workers > 1) produces numerically
identical sparse_phase and sparse_potential tensors as the serial path.
"""

import numpy as np
import pytest
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


def test_output_dir_writes_files_for_every_kpoint(si_forceconstants, tmp_path):
    """When projection_output_dir is set, every k-point's .npz must exist
    on disk after the run (not just the sentinel). Regression for the
    PR #231 memory bug: the old code accumulated results in memory and
    only wrote files at the end; this test pins that the new code writes
    inside the dispatch loop (and drops the in-memory copy).
    """
    import os
    import glob

    output_dir = str(tmp_path / "kpt_out")
    phonons = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder=str(tmp_path / "phonons_folder"),
        projection_output_dir=output_dir,
        n_workers=1,
    )
    _ = phonons.sparse_phase

    npz_files = sorted(glob.glob(os.path.join(output_dir, "kpt_*.npz")))
    done_files = sorted(glob.glob(os.path.join(output_dir, "kpt_*.done")))

    # 3x3x3 mesh = 27 k-points
    assert len(npz_files) == 27, (
        f"Expected 27 .npz files, got {len(npz_files)}: {npz_files}"
    )
    assert len(done_files) == 27, (
        f"Expected 27 .done sentinels, got {len(done_files)}"
    )
