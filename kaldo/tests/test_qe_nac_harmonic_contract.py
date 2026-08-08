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


def test_qe_precomputation_owns_a_kernel_and_skips_gonze_mapping(tmp_path, monkeypatch) -> None:
    """q2r provenance must select its native kernel before Gonze preparation."""
    second = _load_mgo_second(tmp_path)

    def fail_if_gonze_mapping_is_built(*args, **kwargs):
        raise AssertionError("QE q2r must not construct a Gonze BvK mapping")

    monkeypatch.setattr(nac, "build_mapping", fail_if_gonze_mapping_is_built)
    bundle = second.get_nac_precomputed()

    assert bundle["mapping"] is None
    assert isinstance(bundle["static_data"]["qe_kernel"], nac._QERigidIonKernel)


def test_qe_harmonic_path_does_not_request_gonze_short_range_ifcs(tmp_path, monkeypatch) -> None:
    """The loaded q2r body is Fourier transformed directly and restored once."""
    second = _load_mgo_second(tmp_path)

    def fail_if_gonze_subtraction_is_requested(*args, **kwargs):
        raise AssertionError("QE q2r must not run the Gonze IFC subtraction")

    monkeypatch.setattr(
        second,
        "get_nac_short_range_force_constants",
        fail_if_gonze_subtraction_is_requested,
    )
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
