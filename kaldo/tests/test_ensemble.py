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


def test_from_calculators_deterministic_zero_std(tmp_path):
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    ens = PhononsEnsemble.from_calculators(
        atoms, (2, 2, 2), [EMT(), EMT(), EMT()],
        delta_shift=1e-2, symmetrize=True,
        kpts=(3, 3, 3), temperature=300, storage='memory',
        folder=str(tmp_path / 'ald'),
    )
    assert ens.n_members == 3
    mean, std = ens.mean_std('frequency')
    np.testing.assert_allclose(std, 0.0, atol=1e-8)
    assert np.all(np.isfinite(mean))


def test_from_calculators_empty_raises():
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    with pytest.raises(ValueError, match="at least one"):
        PhononsEnsemble.from_calculators(
            atoms, (2, 2, 2), [], kpts=(3, 3, 3), temperature=300,
            storage='memory'
        )


def _perturbed_second(atoms, base_second, scale, seed):
    """Return a copy of base_second with symmetry-breaking noise on its value."""
    from kaldo.observables.secondorder import SecondOrder
    rng = np.random.default_rng(seed)
    value = np.asarray(base_second.value).copy()
    value = value + rng.normal(scale=scale, size=value.shape)
    return SecondOrder.from_supercell(
        atoms=atoms, supercell=(2, 2, 2), grid_type='C', value=value, is_acoustic_sum=False)


def test_symmetrization_reduces_gamma_frequency_spread(tmp_path):
    atoms, base_second = _cu_second(tmp_path, 'base')

    def build(symmetrize):
        members = [PhononsEnsemble._member_from_second(
            atoms, (2, 2, 2), s, symmetrize=symmetrize,
            phonons_kwargs=dict(kpts=(3, 3, 3), temperature=300, storage='memory'))
            for s in [_perturbed_second(atoms, base_second, scale=1e-2, seed=s) for s in range(4)]]
        return PhononsEnsemble(members).std('frequency')

    std_raw = build(symmetrize=False)
    std_sym = build(symmetrize=True)
    # Projection removes per-member symmetry violations, tightening the spread.
    assert std_sym.sum() < std_raw.sum()


def test_single_member_zero_std(tmp_path):
    members = _cu_phonons(tmp_path, n=1)
    ens = PhononsEnsemble(members)
    assert ens.n_members == 1
    _, std = ens.mean_std('frequency')
    np.testing.assert_allclose(std, 0.0, atol=1e-12)


def test_from_calculators_gives_members_distinct_folders(tmp_path):
    # Regression: on-disk storage must not have members clobber each other.
    # Each member's Phonons folder must be distinct (member_i under base).
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    base = str(tmp_path / 'ald')
    ens = PhononsEnsemble.from_calculators(
        atoms, (2, 2, 2), [EMT(), EMT()],
        delta_shift=1e-2, symmetrize=True,
        kpts=(3, 3, 3), temperature=300, storage='numpy', folder=base,
    )
    folders = [m.folder for m in ens.members]
    assert folders == [f"{base}/member_0", f"{base}/member_1"]
    assert len(set(folders)) == ens.n_members
