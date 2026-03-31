"""
Tests verifying that parallel and scratch/resume execution of calculate_second
return numerically identical results to the serial baseline.
"""

import glob
import os

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.controllers.displacement import calculate_second
from kaldo.observables.secondorder import SecondOrder


@pytest.fixture(scope="module")
def al_atoms():
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated_atoms = atoms.repeat((1, 1, 2))
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms


@pytest.fixture(scope="module")
def serial_result(al_atoms):
    atoms, replicated_atoms = al_atoms
    return calculate_second(
        atoms,
        replicated_atoms,
        1e-5,
        n_workers=1,
        calculator=EMT(),
    )


def test_parallel_matches_serial(al_atoms, serial_result):
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


def test_scratch_matches_serial(al_atoms, serial_result, tmp_path):
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
        err_msg="Scratch second order calculation results differ from serial.",
    )


def test_scratch_resume_matches_serial(al_atoms, serial_result, tmp_path):
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
    np.testing.assert_allclose(serial_result, result_full, rtol=1e-7, atol=1e-9)

    sentinel = os.path.join(scratch_dir, "iat_00000.done")
    assert os.path.exists(sentinel), "Expected sentinel file for atom 0"
    os.remove(sentinel)
    os.remove(os.path.join(scratch_dir, "iat_00000.npz"))

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
        err_msg="Resumed scratch result differs from serial baseline.",
    )
    assert not glob.glob(os.path.join(scratch_dir, "iat_*.done"))
