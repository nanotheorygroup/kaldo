"""Public API contract for ordinary IFC interpolation."""

import warnings

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons


@pytest.fixture(scope="module")
def cu_second(tmp_path_factory):
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=(2, 2, 2),
        folder=str(tmp_path_factory.mktemp("fc_cu_interpolation")),
    )
    forceconstants.second.calculate(
        calculator=EMT(), delta_shift=1.0e-3, is_storing=False
    )
    return forceconstants.second


def test_auto_resolves_compact_periodic_input_to_wigner_seitz(cu_second):
    harmonic = HarmonicWithQ(
        np.array([0.1, 0.2, 0.3]), cu_second, storage="memory"
    )

    assert harmonic.ifc_interpolation == "auto"
    assert harmonic.ifc_interpolation_resolved == "wigner-seitz"


def test_explicit_periodic_override_is_honored(cu_second):
    q_point = np.array([0.1, 0.2, 0.3])
    periodic = HarmonicWithQ(
        q_point,
        cu_second,
        storage="memory",
        ifc_interpolation="periodic",
    )
    wigner_seitz = HarmonicWithQ(
        q_point,
        cu_second,
        storage="memory",
        ifc_interpolation="wigner-seitz",
    )

    assert periodic.ifc_interpolation_resolved == "periodic"
    assert np.max(np.abs(periodic.frequency - wigner_seitz.frequency)) > 1.0e-4


def test_invalid_or_removed_interpolation_arguments_fail_loudly(cu_second):
    with pytest.raises(ValueError, match="ifc_interpolation"):
        HarmonicWithQ(
            np.zeros(3), cu_second, ifc_interpolation="unfolded"
        )
    unfolding = HarmonicWithQ(
        np.zeros(3), cu_second, storage="memory", is_unfolding=True
    )
    assert unfolding.ifc_interpolation_resolved == "wigner-seitz"


def test_phonons_warns_only_about_unknown_keywords(tmp_path):
    """An unknown keyword emits a FutureWarning that names it; known ones are silent."""
    forceconstants = ForceConstants(atoms=bulk("Cu", "fcc", a=3.61, cubic=True),
                                    supercell=(2, 2, 2), folder=str(tmp_path))
    with pytest.warns(FutureWarning, match="ifc_interpolation"):
        Phonons(forceconstants=forceconstants, storage="memory", ifc_interpolation="periodic")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        Phonons(forceconstants=forceconstants, storage="memory")


def test_default_third_order_falls_back_to_periodic_for_a_wire(tmp_path):
    """A wire periodic only along z has no Wigner-Seitz images; like second
    order, the default third order keeps the periodic representation."""
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.center(vacuum=6.0, axis=(0, 1))
    atoms.pbc = (False, False, True)
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 3), third_supercell=(1, 1, 3),
                        folder=str(tmp_path))
    fc.second.calculate(EMT(), delta_shift=1e-3, is_storing=False)
    fc.third.calculate(EMT(), delta_shift=1e-3, is_storing=False)
    settings = dict(forceconstants=fc, kpts=(1, 1, 3), temperature=300, is_nw=True,
                    storage="memory")
    default = Phonons(**settings).bandwidth
    periodic = Phonons(_ifc_interpolation="periodic", **settings).bandwidth
    np.testing.assert_allclose(default, periodic)
