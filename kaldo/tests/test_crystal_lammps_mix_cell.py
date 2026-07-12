"""
Regression tests for mixed-supercell LAMMPS loading.

A universal ML potential needs a large second-order supercell (slow decay of
the dynamical matrix) but converges the third order on a much smaller one.
The lammps format supports this by placing a replicated_atoms_third.xyz
describing the third-order supercell next to the second order's
replicated_atoms.xyz; the loader prefers it when present (same convention as
the sparse/numpy formats).

Conductivity pins follow the gauge-invariant regression configuration of
issue #290: fixed third_bandwidth (and diffusivity_bandwidth for QHGK), so
the goldens are machine-independent and pinned at rtol=5e-3.
"""

from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest


@pytest.fixture(scope="session")
def forceconstants():
    return ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal-mix-cell",
        supercell=[6, 6, 6],
        third_supercell=[3, 3, 3],
        format="lammps",
    )


@pytest.fixture(scope="session")
def phonons(forceconstants):
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[6, 6, 6],
        is_classic=False,
        temperature=300,
        third_bandwidth=0.5,  # fixed: gauge-invariant regression config (#290)
        storage="memory",
    )
    phonons.frequency
    return phonons


def test_mixed_supercells_are_loaded(forceconstants):
    """The defining property of the feature: the two orders live on different
    supercells read from the same folder."""
    assert forceconstants.second.n_replicas == 216  # 6 x 6 x 6
    assert forceconstants.third.n_replicas == 27    # 3 x 3 x 3


def test_sc_conductivity(phonons):
    cond = np.abs(
        np.mean(
            Conductivity(phonons=phonons, method="sc", max_n_iterations=71, storage="memory")
            .conductivity.sum(axis=0)
            .diagonal()
        )
    )
    np.testing.assert_allclose(cond, 253.544245, rtol=5e-3, atol=0.0)


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 3.118208, rtol=5e-3, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 219.174159, rtol=5e-3, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 253.544246, rtol=5e-3, atol=0.0)


def test_third_falls_back_to_replicated_atoms(tmp_path):
    """Without a replicated_atoms_third.xyz the loader must fall back to
    replicated_atoms.xyz unchanged (single-supercell runs keep working)."""
    import shutil
    from pathlib import Path
    from kaldo.observables.thirdorder import ThirdOrder

    src = Path("kaldo/tests/si-crystal-mix-cell")
    shutil.copy(src / "THIRD", tmp_path / "THIRD")
    # same 3x3x3 geometry, but under the fallback filename
    shutil.copy(src / "replicated_atoms_third.xyz", tmp_path / "replicated_atoms.xyz")

    third_mixed = ThirdOrder.load(folder=str(src), supercell=(3, 3, 3), format='lammps')
    third_fallback = ThirdOrder.load(folder=str(tmp_path), supercell=(3, 3, 3), format='lammps')

    a = np.asarray(third_mixed.value.todense())
    b = np.asarray(third_fallback.value.todense())
    np.testing.assert_allclose(b, a, rtol=0.0, atol=0.0)


def test_third_missing_both_files_raises(tmp_path):
    """With neither supercell file present the loader raises a ValueError
    naming both candidates instead of a raw file error deep in ase."""
    import shutil
    from pathlib import Path
    from kaldo.observables.thirdorder import ThirdOrder

    shutil.copy(Path("kaldo/tests/si-crystal-mix-cell") / "THIRD", tmp_path / "THIRD")
    with pytest.raises(ValueError, match="replicated_atoms_third.xyz or replicated_atoms.xyz"):
        ThirdOrder.load(folder=str(tmp_path), supercell=(3, 3, 3), format='lammps')
