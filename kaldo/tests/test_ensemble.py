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


def test_identical_members_zero_std(tmp_path):
    members = _cu_phonons(tmp_path, n=3)
    ens = PhononsEnsemble(members)
    mean, std = ens.mean_std('frequency')
    ref = np.asarray(members[0].frequency)
    assert mean.shape == ref.shape
    np.testing.assert_allclose(mean, ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(std, 0.0, atol=1e-10)


def test_mean_and_std_helpers_match_mean_std(tmp_path):
    members = _cu_phonons(tmp_path, n=2)
    ens = PhononsEnsemble(members)
    mean, std = ens.mean_std('frequency')
    np.testing.assert_array_equal(ens.mean('frequency'), mean)
    np.testing.assert_array_equal(ens.std('frequency'), std)


def test_unknown_observable_raises_helpful(tmp_path):
    members = _cu_phonons(tmp_path, n=1)
    ens = PhononsEnsemble(members)
    with pytest.raises(AttributeError, match="not_a_prop"):
        ens.mean_std('not_a_prop')


def _cu_second(tmp_path, folder):
    """Compute one raw (unsymmetrized) SecondOrder for Cu via EMT."""
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    fc = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=str(tmp_path / folder))
    fc.second.calculate(calculator=EMT(), delta_shift=1e-2, is_storing=False, symmetrize=False)
    return atoms, fc.second


def test_member_from_second_builds_phonons(tmp_path):
    atoms, second = _cu_second(tmp_path, 'a')
    member = PhononsEnsemble._member_from_second(
        atoms, (2, 2, 2), second, symmetrize=True,
        phonons_kwargs=dict(kpts=(3, 3, 3), temperature=300, storage='memory')
    )
    assert isinstance(member, Phonons)
    assert np.asarray(member.frequency).shape[1] == len(atoms) * 3


def test_shape_mismatch_across_members_raises(tmp_path):
    atoms_a, second_a = _cu_second(tmp_path, 'a')
    atoms_b, second_b = _cu_second(tmp_path, 'b')
    m_a = PhononsEnsemble._member_from_second(
        atoms_a, (2, 2, 2), second_a, symmetrize=False,
        phonons_kwargs=dict(kpts=(3, 3, 3), temperature=300, storage='memory'))
    m_b = PhononsEnsemble._member_from_second(
        atoms_b, (2, 2, 2), second_b, symmetrize=False,
        phonons_kwargs=dict(kpts=(2, 2, 2), temperature=300, storage='memory'))
    ens = PhononsEnsemble([m_a, m_b])
    with pytest.raises(ValueError, match="disagree on the shape"):
        ens.mean_std('frequency')
