"""Ensure one NAC grid convention reaches all harmonic consumers."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

import kaldo.conductivity as conductivity_module
import kaldo.phonons as phonons_module
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


@pytest.fixture
def nac_phonons(tmp_path):
    """Create a small polar Phonons object without evaluating observables."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.atoms.info.pop("dipole_subtracted_fc", None)
    forceconstants.second._qe_q2r_header = None
    matrix = np.diag([8, 8, 8])
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=(1, 1, 1),
        temperature=300,
        storage="memory",
        folder=str(tmp_path),
        is_nac=False,
        nac_bvk_supercell_matrix=matrix,
    )
    return phonons, matrix


def test_q2r_blank_species_label_uses_auxiliary_identity_without_geometry_override():
    """QE permits blank type labels in q2r; only that identity may be borrowed."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        format="shengbte-qe",
    )
    second = forceconstants.second
    header = second._qe_q2r_header

    assert "" in header.symbols
    assert tuple(second.atoms.get_chemical_symbols()) == ("Na", "Cl")
    np.testing.assert_allclose(
        second.atoms.cell,
        header.cell_rows_bohr * 0.529177210903,
    )
    np.testing.assert_allclose(
        second.atoms.positions,
        header.positions_rows_bohr * 0.529177210903,
    )


@pytest.mark.parametrize(
    "property_name", ["heat_capacity", "heat_capacity_2d", "population"]
)
def test_temperature_observables_propagate_nac_bvk_matrix(
    nac_phonons, monkeypatch, property_name
) -> None:
    """Thermodynamics must use the same harmonic matrix as frequencies."""
    phonons, matrix = nac_phonons
    observed = []

    class FakeHarmonicWithQTemp:
        def __init__(self, **kwargs):
            observed.append((kwargs["nac_bvk_supercell_matrix"], kwargs["is_nac"]))
            n_modes = len(kwargs["second"].atoms) * 3
            self.heat_capacity = np.ones((1, n_modes))
            self.heat_capacity_2d = np.ones((n_modes, n_modes))
            self.population = np.ones((1, n_modes))

    monkeypatch.setattr(
        phonons_module,
        "HarmonicWithQTemp",
        FakeHarmonicWithQTemp,
    )

    getattr(phonons, property_name)
    assert observed
    for propagated_matrix, propagated_is_nac in observed:
        np.testing.assert_array_equal(propagated_matrix, matrix)
        assert propagated_is_nac is False


def test_qhgk_propagates_nac_bvk_matrix(monkeypatch) -> None:
    """QHGK heat capacities and flux matrices must share the NAC grid."""
    matrix = np.diag([2, 2, 2])
    fake_phonons = SimpleNamespace(
        omega=np.ones((1, 1)),
        n_k_points=1,
        n_modes=1,
        atoms=Atoms("H", cell=np.eye(3)),
        _reciprocal_grid=SimpleNamespace(fractional_points=np.zeros((1, 3))),
        physical_mode=np.ones((1, 1), dtype=bool),
        forceconstants=SimpleNamespace(second=object(), distance_threshold=None),
        is_nw=False,
        ifc_interpolation="auto",
        _is_amorphous=False,
        is_nac=False,
        nac_bvk_supercell_matrix=matrix,
    )
    conductivity = object.__new__(Conductivity)
    conductivity.phonons = fake_phonons
    conductivity.diffusivity_shape = "lorentz"
    conductivity.is_diffusivity_including_antiresonant = False
    conductivity.diffusivity_bandwidth = 0.1
    conductivity.diffusivity_threshold = None
    conductivity.folder = "unused"
    conductivity.storage = "memory"
    conductivity.temperature = 300
    conductivity.is_classic = False

    class ExpectedConstruction(RuntimeError):
        pass

    def inspect_harmonic_kwargs(**kwargs):
        np.testing.assert_array_equal(kwargs["nac_bvk_supercell_matrix"], matrix)
        assert kwargs["is_nac"] is False
        raise ExpectedConstruction

    monkeypatch.setattr(
        conductivity_module.hwqwt,
        "HarmonicWithQTemp",
        inspect_harmonic_kwargs,
    )

    with pytest.raises(ExpectedConstruction):
        conductivity.calculate_conductivity_and_diffusivity_qhgk()
