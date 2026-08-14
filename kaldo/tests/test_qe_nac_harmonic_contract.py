"""Behavioral contracts for QE q2r in the common harmonic NAC entry point."""

from pathlib import Path

import numpy as np
import pytest

from kaldo.controllers import nac
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

ROOT = Path(__file__).parents[1]


def _load_mgo_second(tmp_path):
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    forceconstants.second.folder = str(tmp_path)
    return forceconstants.second


def test_qe_precomputation_owns_a_kernel_and_ws_mapping(tmp_path) -> None:
    """q2r provenance selects QE electrostatics and shared WS interpolation."""
    second = _load_mgo_second(tmp_path)
    bundle = second.get_nac_precomputed()

    assert bundle["mapping"] is not None
    assert "phase_weights" in bundle["mapping"]
    assert isinstance(bundle["static_data"]["qe_kernel"], nac._QERigidIonKernel)


def test_qe_harmonic_path_does_not_run_gonze_dipole_subtraction(
    tmp_path, monkeypatch
) -> None:
    """The loaded q2r body is WS interpolated and restored exactly once."""
    second = _load_mgo_second(tmp_path)

    def fail_if_gonze_dipole_is_requested(*args, **kwargs):
        raise AssertionError("QE q2r must not run the Gonze IFC subtraction")

    monkeypatch.setattr(nac, "_dipole_dipole_dynamical_matrix", fail_if_gonze_dipole_is_requested)
    phonon = HarmonicWithQ(
        q_point=np.array([0.3, 0.0, 0.3]),
        second=second,
        storage="memory",
        is_unfolding=True,
    )

    assert np.isfinite(phonon.frequency).all()


def test_qe_q2r_rejects_remeshing_onto_a_different_bvk_cell(tmp_path) -> None:
    """A different BvK grid would require an unimplemented q2r resampling."""
    second = _load_mgo_second(tmp_path)
    with pytest.raises(NotImplementedError, match="cannot remesh QE q2r"):
        second.get_nac_precomputed(np.diag([4, 5, 5]))


def test_format_specific_code_stays_in_existing_interface_and_controller_modules() -> None:
    """Keep the interface split without introducing QE-specific library files."""
    interfaces = ROOT / "interfaces"
    assert (interfaces / "qe_io.py").is_file()
    assert (interfaces / "vasp_io.py").is_file()
    assert not (interfaces / "qe_lattice.py").exists()
    assert not (interfaces / "qe_nac.py").exists()
