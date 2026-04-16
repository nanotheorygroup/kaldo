"""
Tests verifying that parallel second-order finite differences return
numerically identical results to the serial baseline.

System: Al FCC conventional cell (4 atoms) with ASE's built-in EMT calculator.
"""

import glob
import os
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from kaldo.controllers.displacement import calculate_second
from kaldo.parallel import CalculatorFactory


@pytest.fixture(scope="module")
def al_atoms():
    print("Preparing Al Object")
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated_atoms = atoms.repeat((1, 1, 2))
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms


@pytest.fixture(scope="module")
def serial_result(al_atoms):
    print("Preparing second order force constants in serial")
    atoms, replicated_atoms = al_atoms
    return calculate_second(
        atoms,
        replicated_atoms,
        1e-5,
        n_workers=1,
        calculator=EMT(),
    )


def test_parallel_matches_serial(al_atoms, serial_result):
    """Parallel execution must return the same second-order tensor as serial."""
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))
    result_parallel = calculate_second(
        atoms,
        replicated_atoms_no_calc,
        1e-5,
        n_workers=2,
        calculator=EMT,
    )
    np.testing.assert_allclose(
        serial_result,
        result_parallel,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Parallel second order calculation results differ from serial.",
    )


def test_factory_matches_serial(al_atoms, serial_result):
    """CalculatorFactory must produce the same tensor as a pre-built EMT."""
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))
    result_factory = calculate_second(
        atoms,
        replicated_atoms_no_calc,
        1e-5,
        n_workers=2,
        calculator=CalculatorFactory(EMT),
    )
    np.testing.assert_allclose(
        serial_result,
        result_factory,
        rtol=1e-7,
        atol=1e-9,
        err_msg="CalculatorFactory-based second order differs from serial.",
    )


def test_parallel_rejects_unpicklable_calculator(al_atoms):
    """Unpicklable callables must be rejected up-front with a clear TypeError."""
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))

    def local_factory():  # nested, not picklable by name
        return EMT()

    with pytest.raises(TypeError, match="not picklable"):
        calculate_second(
            atoms,
            replicated_atoms_no_calc,
            1e-5,
            n_workers=2,
            calculator=local_factory,
        )


def test_serial_accepts_unpicklable_calculator(al_atoms):
    """Serial runs don't pickle, so unpicklable callables must still work."""
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))

    def local_factory():  # nested, not picklable by name
        return EMT()

    calculate_second(
        atoms,
        replicated_atoms_no_calc,
        1e-5,
        n_workers=1,
        calculator=local_factory,
    )


def test_scratch_matches_serial(al_atoms, serial_result, tmp_path):
    """Scratch-backed second-order assembly must match the serial baseline."""
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))
    scratch_dir = str(tmp_path / "scratch")

    result_scratch = calculate_second(
        atoms,
        replicated_atoms_no_calc,
        1e-5,
        n_workers=2,
        calculator=EMT,
        scratch_dir=scratch_dir,
        keep_scratch=True,
    )
    np.testing.assert_allclose(
        serial_result,
        result_scratch,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Scratch-backed second order calculation differs from serial.",
    )


def test_scratch_resume_matches_serial(al_atoms, serial_result, tmp_path):
    """
    Procedure:
    1. Full scratch-backed run while keeping scratch files.
    2. Delete atom-0 sentinel + dense scratch file to simulate interruption.
    3. Re-run with the same scratch_dir.
    4. Assert result matches serial and scratch files are cleaned up.
    """
    atoms, _ = al_atoms
    replicated_atoms_no_calc = atoms.repeat((1, 1, 2))
    scratch_dir = str(tmp_path / "scratch")

    result_full = calculate_second(
        atoms,
        replicated_atoms_no_calc,
        1e-5,
        n_workers=2,
        calculator=EMT,
        scratch_dir=scratch_dir,
        keep_scratch=True,
    )
    np.testing.assert_allclose(
        serial_result,
        result_full,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Initial scratch-backed second order calculation differs from serial.",
    )

    sentinel = os.path.join(scratch_dir, "iat_00000.done")
    scratch_file = os.path.join(scratch_dir, "iat_00000.npy")
    assert os.path.exists(sentinel), "Expected sentinel file for atom 0"
    assert os.path.exists(scratch_file), "Expected scratch file for atom 0"
    os.remove(sentinel)
    os.remove(scratch_file)

    result_resumed = calculate_second(
        atoms,
        replicated_atoms_no_calc,
        1e-5,
        n_workers=2,
        calculator=EMT,
        scratch_dir=scratch_dir,
        keep_scratch=False,
    )
    np.testing.assert_allclose(
        serial_result,
        result_resumed,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Resumed scratch-backed second order calculation differs from serial.",
    )
    assert not glob.glob(os.path.join(scratch_dir, "iat_*.npy"))
    assert not glob.glob(os.path.join(scratch_dir, "iat_*.done"))
    assert not os.path.exists(scratch_dir)
