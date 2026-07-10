"""
Tests for the incommensurate-q warning in HarmonicWithQ.

At q-points incommensurate with the supercell the default (is_unfolding=False)
dynamical-matrix construction uses the periodic-replica convention, which can
break symmetry-protected degeneracies (e.g. split transverse-acoustic branches
in diamond-structure crystals). A single warning per process points users to
is_unfolding=True.
"""
import logging

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.forceconstants import ForceConstants
from kaldo.observables import harmonic_with_q as hwq_module
from kaldo.observables.harmonic_with_q import HarmonicWithQ


@pytest.fixture(scope='module')
def cu_second(tmp_path_factory):
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    fc = ForceConstants(atoms=atoms, supercell=(2, 2, 2),
                        folder=str(tmp_path_factory.mktemp('fc_cu')))
    fc.second.calculate(calculator=EMT(), delta_shift=1e-3, is_storing=False)
    return fc.second


def test_incommensurate_q_without_unfolding_warns_once(cu_second, caplog):
    hwq_module._warned_incommensurate = False
    with caplog.at_level(logging.WARNING, logger='kaldo'):
        HarmonicWithQ(np.array([0.1, 0.0, 0.0]), cu_second, storage='memory')
        HarmonicWithQ(np.array([0.2, 0.0, 0.0]), cu_second, storage='memory')
    hits = [r for r in caplog.records if 'is_unfolding' in r.message]
    assert len(hits) == 1


def test_commensurate_or_unfolded_q_does_not_warn(cu_second, caplog):
    hwq_module._warned_incommensurate = False
    with caplog.at_level(logging.WARNING, logger='kaldo'):
        HarmonicWithQ(np.array([0.5, 0.0, 0.0]), cu_second, storage='memory')
        HarmonicWithQ(np.array([0.1, 0.0, 0.0]), cu_second, storage='memory',
                      is_unfolding=True)
    assert not [r for r in caplog.records if 'is_unfolding' in r.message]
