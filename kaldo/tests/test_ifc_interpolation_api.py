"""Public API contract for ordinary IFC interpolation."""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ


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
