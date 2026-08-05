"""Contracts for integrating QE q2r into the common harmonic NAC path."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_qe_uses_common_harmonic_entry_point_with_its_ifc_convention() -> None:
    source = (ROOT / "controllers" / "nac.py").read_text(encoding="utf-8")
    assert 'static_data.get("convention") == "qe_q2r"' in source
    assert "_qe_correction(" in source
    assert "_dynamical_matrix_from_second_order(" in source


def test_q2r_short_range_ifcs_do_not_enter_gonze_subtraction() -> None:
    source = (ROOT / "observables" / "secondorder.py").read_text(encoding="utf-8")
    assert 'getattr(self, "_qe_q2r_header", None) is not None' in source
    assert "already contains QE's" in source
    assert 'static_data.get("convention") != "qe_q2r"' in source


def test_format_specific_code_stays_in_existing_interface_and_controller_modules() -> (
    None
):
    interfaces = ROOT / "interfaces"
    assert (interfaces / "qe_io.py").is_file()
    assert (interfaces / "vasp_io.py").is_file()
    assert not (interfaces / "qe_lattice.py").exists()
    assert not (interfaces / "qe_nac.py").exists()
