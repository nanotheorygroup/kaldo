"""
Unit and regression test for the kaldo package.
Tests the unfolding routine for second (and soon, third) order
force constants.
"""

# Import package, test suite, and other packages as needed
import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        is_unfolding=True,
        storage="memory",
    )
    return phonons


def test_nac_dispersion(phonons):
    q_point = np.array([0.3, 0, 0.3])  # chosen to check we get a degenerate pair for both acoustic and optical
    frequency_expected = np.array([7.18794357,  7.18794363, 11.02311516, 12.67918914, 12.67918918, 17.46740768])
    frequency_actual = HarmonicWithQ(q_point=q_point, second=phonons.forceconstants.second, is_unfolding=True).frequency
    frequency_actual = frequency_actual.flatten()  # HWQ outputs a 2d array
    np.testing.assert_array_almost_equal(frequency_expected, frequency_actual, decimal=2)

def test_nac_velocity(phonons):
    q_point = np.array([0.3, 0, 0.3])
    # Rebaselined for the NAC controller's finite-difference kernel.
    velocity_expected = np.array([29.51233120, 29.51232955, 46.84582095, 9.99299064, 9.99298982, 32.47622821])
    velocity = HarmonicWithQ(q_point=q_point, second=phonons.forceconstants.second, is_unfolding=True).velocity
    velocity_actual = np.linalg.norm(velocity, axis=-1).flatten()
    np.testing.assert_array_almost_equal(velocity_expected, velocity_actual, decimal=2)


def test_scalar_charges_are_not_polar_metadata():
    """A LAMMPS-style per-atom charge column is not a Born-charge block."""
    from ase import Atoms
    from kaldo.observables.harmonic_with_q import _resolve_nac_activation

    atoms = Atoms("NaCl", positions=[[0, 0, 0], [2.8, 0, 0]], cell=np.eye(3) * 5.6)
    atoms.set_array("charges", np.array([1.0, -1.0]))
    assert _resolve_nac_activation(atoms, None) is False

    atoms.info["dielectric"] = np.eye(3) * 2.5
    with pytest.raises(ValueError, match=r"has shape \(2,\)"):
        _resolve_nac_activation(atoms, None)
    with pytest.raises(ValueError, match=r"has shape \(2,\)"):
        _resolve_nac_activation(atoms, True)
    assert _resolve_nac_activation(atoms, False) is False


def test_nac_input_guards(tmp_path):
    """Deferred or invalid NAC inputs fail loudly instead of being ignored."""
    from types import SimpleNamespace
    from kaldo.controllers.nac import build_mapping, normalize_bvk_supercell_matrix
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    with pytest.raises(ValueError, match="integer-valued"):
        normalize_bvk_supercell_matrix(np.diag([1.9, 2.0, 2.0]))
    with pytest.raises(ValueError, match="integer-valued"):
        normalize_bvk_supercell_matrix(np.diag([100000.5, 2.0, 2.0]))
    with pytest.raises(ValueError, match="integer-valued"):
        normalize_bvk_supercell_matrix(np.full((3, 3), np.inf))

    fake_second = SimpleNamespace(supercell=np.array([[3, 1, 0], [0, 3, 0], [0, 0, 3]]))
    with pytest.raises(NotImplementedError, match="diagonal supercells only"):
        build_mapping(fake_second)

    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo", supercell=[5, 5, 5], format="shengbte-qe", only_second=True
    )
    second = forceconstants.second
    with pytest.raises(ValueError, match="nonzero direction"):
        HarmonicWithQ(np.zeros(3), second, storage="memory", nac_q_direction=(0, 0, 0))
    with pytest.raises(ValueError, match="distance_threshold"):
        HarmonicWithQ(np.zeros(3), second, storage="memory", distance_threshold=15.0)
    with pytest.raises(ValueError, match="finite 3-vector"):
        HarmonicWithQ(np.zeros(3), second, storage="memory", nac_q_direction=(1, 0))
    with pytest.raises(NotImplementedError, match="heat-flux operator"):
        HarmonicWithQ(np.array([0.2, 0.0, 0.0]), second, storage="memory").calculate_sij(0)


def test_phonons_forwards_nac_q_direction(monkeypatch):
    """The kwarg reaches every HarmonicWithQ construction, cubic symmetry aside."""
    import kaldo.phonons as phonons_module

    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo", supercell=[5, 5, 5], format="shengbte-qe", only_second=True
    )
    captured = []
    original = phonons_module.HarmonicWithQ

    def capturing(*args, **kwargs):
        captured.append(kwargs.get("nac_q_direction"))
        return original(*args, **kwargs)

    monkeypatch.setattr(phonons_module, "HarmonicWithQ", capturing)
    phonons = Phonons(
        forceconstants=forceconstants, kpts=[1, 1, 1], temperature=300,
        storage="memory", nac_q_direction=(0, 1, 0),
    )
    phonons.frequency
    assert captured and all(tuple(d) == (0, 1, 0) for d in captured)
