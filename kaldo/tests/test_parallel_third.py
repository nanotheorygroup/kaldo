"""
Tests verifying that parallel and scratch/resume execution of calculate_third
return numerically identical results to the serial baseline.

System: Al FCC conventional cell (4 atoms) with ASE's built-in EMT calculator.
"""

import glob
import os
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from kaldo.controllers.displacement import calculate_third


@pytest.fixture(scope="module")
def al_atoms():
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)  # 4 atoms
    replicated_atoms = atoms.repeat((1, 1, 2))      # asymmetric supercell
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms


@pytest.fixture(scope="module")
def serial_result(al_atoms):
    atoms, replicated_atoms = al_atoms
    return calculate_third(
        atoms, replicated_atoms, 1e-5, n_workers=1, calculator=EMT(),
    )


def test_parallel_matches_serial(al_atoms, serial_result):
    """Parallel execution must return the same third-order tensor as serial."""
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))
    result_parallel = calculate_third(
        atoms, replicated_atoms_no_calc, 1e-5, n_workers=2, calculator=EMT,
    )
    np.testing.assert_allclose(
        serial_result.todense(), result_parallel.todense(), rtol=1e-7,
        err_msg="Parallel third order calculation results differ from serial.",
    )


def test_scratch_resume_matches_serial(al_atoms, serial_result, tmp_path):
    """
    Procedure:
    1. Full parallel run with scratch_dir → matches serial.
    2. Delete atom-0 sentinel + chunks to simulate an interrupted run.
    3. Re-run with the same scratch_dir → atom 0 is recomputed; rest resumed.
    4. Assert result still matches serial.

    Step 1 subsumes a plain ``test_scratch_matches_serial`` test.
    """
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))
    scratch_dir = str(tmp_path / "scratch")

    # --- Step 1: full scratch-backed run, keeping files ---
    result_full = calculate_third(
        atoms, replicated_atoms_no_calc, 1e-5, n_workers=2, calculator=EMT,
        scratch_dir=scratch_dir, keep_scratch=True,
    )
    np.testing.assert_allclose(
        serial_result.todense(), result_full.todense(), rtol=1e-7,
        err_msg="Initial scratch-backed third order calculation differs from serial.",
    )

    # --- Step 2: simulate interruption by removing atom-0 output ---
    sentinel = os.path.join(scratch_dir, "iat_00000.done")
    assert os.path.exists(sentinel), "Expected sentinel file for atom 0"
    os.remove(sentinel)
    for chunk_file in glob.glob(os.path.join(scratch_dir, "iat_00000_chunk_*.npz")):
        os.remove(chunk_file)

    # --- Step 3: resume run ---
    result_resumed = calculate_third(
        atoms, replicated_atoms_no_calc, 1e-5, n_workers=2, calculator=EMT,
        scratch_dir=scratch_dir, keep_scratch=False,
    )

    # --- Step 4: resumed result must match serial ---
    np.testing.assert_allclose(
        serial_result.todense(), result_resumed.todense(), rtol=1e-7,
        err_msg="Resumed scratch result differs from serial baseline",
    )
