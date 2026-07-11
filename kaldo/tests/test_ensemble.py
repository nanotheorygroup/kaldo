import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.ensemble import PhononsEnsemble


def _cu_phonons(tmp_path, n=1):
    """Build n identical EMT Phonons for Cu (deterministic; std must be ~0)."""
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    members = []
    for i in range(n):
        fc = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=str(tmp_path / f'm{i}'))
        fc.second.calculate(calculator=EMT(), delta_shift=1e-2, is_storing=False)
        members.append(Phonons(forceconstants=fc, kpts=(3, 3, 3), temperature=300, storage='memory'))
    return members


def test_members_and_count(tmp_path):
    members = _cu_phonons(tmp_path, n=2)
    ens = PhononsEnsemble(members)
    assert ens.n_members == 2
    assert ens.members is members or list(ens.members) == members


def test_empty_ensemble_raises():
    with pytest.raises(ValueError, match="at least one member"):
        PhononsEnsemble([])
