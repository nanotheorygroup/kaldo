"""Regression tests for NAC group velocities.

The removed legacy NAC implementation evaluated frequencies and velocity
derivatives with separate Ewald formulas.  On an FCC primitive cell their
Gaussian widths differed, so the reported velocity was not the gradient of
the dispersion.  The current controller finite-differences one shared polar
dynamical matrix; these tests preserve the physical invariant through the
public API rather than replaying private intermediate tensors. Both the generic
Gonze and QE q2r conventions are exercised on non-symmetric cells so an
accidental direct-cell transpose cannot hide behind cubic symmetry.
"""

from pathlib import Path
import shutil

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from ase.units import Bohr

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces.qe_io import read_q2r_header
from kaldo.observables.harmonic_with_q import HarmonicWithQ


NAC_BVK_MATRIX = np.diag([8, 8, 8])
Q_POINT = np.array([0.073, 0.041, 0.029])
WAVEVECTOR_STEP = 1.0e-3  # 1/angstrom


@pytest.fixture(scope="module")
def generic_fcc_second(tmp_path_factory):
    """Load the NaCl FCC fixture as total IFCs for the generic Gonze path."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    second = forceconstants.second
    # This fixture is used as total force constants, as in the generic
    # VASP/Phonopy convention.  Remove its q2r provenance deliberately so the
    # test exercises Gonze preparation rather than QE rigid-ion restoration.
    second.atoms.info.pop("dipole_subtracted_fc", None)
    second._qe_q2r_header = None
    second.folder = str(tmp_path_factory.mktemp("nac_velocity_fcc"))
    assert "dielectric" in second.atoms.info
    assert np.max(np.abs(second.atoms.get_array("charges"))) > 0
    return second


@pytest.fixture(scope="module")
def qe_skew_second(tmp_path_factory):
    """Load the strained oF QE reference through the public q2r interface."""
    case_id = "oF-synthetic-strained-NaCl"
    source = (
        Path(__file__).parent
        / "data"
        / "input"
        / "qe76-bravais-reference"
        / case_id
        / f"{case_id}.fc"
    )
    header = read_q2r_header(source)
    folder = tmp_path_factory.mktemp("qe_nac_velocity_skew")
    symbols = [header.symbols[index - 1] for index in header.atom_types]
    atoms = Atoms(
        symbols=symbols,
        positions=header.positions_rows_bohr * Bohr,
        cell=header.cell_rows_bohr * Bohr,
        pbc=True,
    )
    write(folder / "POSCAR", atoms, format="vasp", direct=True)
    shutil.copyfile(source, folder / "espresso.ifc2")
    forceconstants = ForceConstants.from_folder(
        folder=str(folder),
        supercell=header.q_grid,
        only_second=True,
        format="qe-d3q",
    )
    second = forceconstants.second
    assert second.atoms.info.get("dipole_subtracted_fc") is True
    assert not np.allclose(second.atoms.cell.array, second.atoms.cell.array.T)
    return second


def _sorted_frequencies(second, q_point, nac_bvk_supercell_matrix=NAC_BVK_MATRIX):
    nac_options = {}
    if nac_bvk_supercell_matrix is not None:
        nac_options["nac_bvk_supercell_matrix"] = nac_bvk_supercell_matrix
    harmonic = HarmonicWithQ(
        q_point=np.asarray(q_point, dtype=float),
        second=second,
        storage="memory",
        **nac_options,
    )
    return np.sort(np.asarray(harmonic.frequency).reshape(-1))


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_generic_fcc_nac_velocity_is_dispersion_gradient(generic_fcc_second, axis):
    """The FCC NAC velocity equals the independently sampled dispersion slope."""
    harmonic = HarmonicWithQ(
        q_point=Q_POINT,
        second=generic_fcc_second,
        storage="memory",
        nac_bvk_supercell_matrix=NAC_BVK_MATRIX,
    )
    order = np.argsort(np.asarray(harmonic.frequency).reshape(-1))
    velocity = np.asarray(harmonic.velocity)[0][order, axis]

    direction = np.eye(3)[axis]
    cell = np.asarray(generic_fcc_second.atoms.cell)
    # q_red uses cycles per reciprocal lattice vector, while the independent
    # slope below is taken with respect to the physical wavevector k in 1/A.
    dq_red = cell @ direction * WAVEVECTOR_STEP / (2 * np.pi)
    plus = _sorted_frequencies(generic_fcc_second, Q_POINT + dq_red)
    minus = _sorted_frequencies(generic_fcc_second, Q_POINT - dq_red)
    slope = (plus - minus) / (2 * WAVEVECTOR_STEP)

    usable = np.abs(slope) > 0.05
    assert np.any(usable), f"no dispersive modes found along Cartesian axis {axis}"
    # kALDo stores frequency in cycles/ps but velocity is d(2*pi*nu)/dk.
    np.testing.assert_allclose(
        velocity[usable] / slope[usable],
        2 * np.pi,
        rtol=1.0e-2,
        atol=0.0,
    )


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_qe_skew_nac_velocity_is_dispersion_gradient(qe_skew_second, axis):
    """QE q2r velocities use the requested Cartesian direction on a skew cell."""
    harmonic = HarmonicWithQ(
        q_point=Q_POINT,
        second=qe_skew_second,
        storage="memory",
    )
    order = np.argsort(np.asarray(harmonic.frequency).reshape(-1))
    velocity = np.asarray(harmonic.velocity)[0][order, axis]

    direction = np.eye(3, dtype=np.float64)[axis]
    cell = np.asarray(qe_skew_second.atoms.cell, dtype=np.float64)
    dq_red = cell @ direction * WAVEVECTOR_STEP / (2.0 * np.pi)
    plus = _sorted_frequencies(qe_skew_second, Q_POINT + dq_red, None)
    minus = _sorted_frequencies(qe_skew_second, Q_POINT - dq_red, None)
    slope = (plus - minus) / (2.0 * WAVEVECTOR_STEP)

    usable = np.abs(slope) > 0.05
    assert np.any(usable), f"no dispersive modes found along Cartesian axis {axis}"
    np.testing.assert_allclose(
        velocity[usable] / slope[usable],
        2.0 * np.pi,
        rtol=1.0e-2,
        atol=0.0,
    )
