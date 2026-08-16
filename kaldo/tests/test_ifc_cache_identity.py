"""Cache identity contracts for provenance-aware IFC interpolation."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.phonons import Phonons


def _forceconstants(*, third_supercell=(2, 1, 1), third_support=None, pbc=True):
    atoms = Atoms("H", cell=np.eye(3), pbc=pbc)
    second_grid = SupercellGrid(np.eye(3, dtype=int))
    second = SimpleNamespace(
        atoms=atoms,
        translation_support=TranslationSupport.periodic(second_grid),
    )
    third = (
        None
        if third_support is None
        else SimpleNamespace(translation_support=third_support)
    )
    return SimpleNamespace(
        atoms=atoms,
        second=second,
        _third=third,
        third_supercell=third_supercell,
        supercell=(1, 1, 1),
        n_atoms=1,
        n_modes=3,
    )


def _phonons(forceconstants, tmp_path, **kwargs):
    return Phonons(
        forceconstants=forceconstants,
        kpts=(1, 1, 1),
        temperature=300,
        storage="memory",
        folder=str(tmp_path),
        is_nac=False,
        **kwargs,
    )


def test_lazy_ifc3_supercells_have_distinct_cache_namespaces(tmp_path):
    """A lazy IFC3 object must not hide its declared translation topology."""
    first = _phonons(
        _forceconstants(third_supercell=(2, 1, 1)), tmp_path / "first"
    )
    second = _phonons(
        _forceconstants(third_supercell=(1, 2, 1)), tmp_path / "second"
    )

    assert first.forceconstants._third is None
    assert second.forceconstants._third is None
    assert first.ifc_cache_key != second.ifc_cache_key
    assert first._add_grid_components("") != second._add_grid_components("")


def test_loaded_ifc3_supports_have_distinct_cache_namespaces(tmp_path):
    """Literal IFC3 supports remain part of the persistent cache identity."""
    grid = SupercellGrid(np.diag([2, 1, 1]))
    compact = TranslationSupport.periodic(grid)
    literal = TranslationSupport([[0, 0, 0], [2, 0, 0]], grid, provenance="file")
    first = _phonons(
        _forceconstants(third_supercell=(2, 1, 1), third_support=compact),
        tmp_path / "compact",
    )
    second = _phonons(
        _forceconstants(third_supercell=(2, 1, 1), third_support=literal),
        tmp_path / "literal",
    )

    assert first.ifc_cache_key != second.ifc_cache_key
    assert first._add_grid_components("") != second._add_grid_components("")


def test_source_aware_auto_does_not_share_explicit_periodic_ifc3_cache(tmp_path):
    """Equal harmonic routes can still request different IFC3 interpolation."""
    forceconstants = _forceconstants(third_supercell=(2, 1, 1))
    forceconstants.second.ifc_interpolation_hint = "periodic"
    automatic = _phonons(forceconstants, tmp_path / "auto")
    explicit = _phonons(
        forceconstants,
        tmp_path / "explicit",
        ifc_interpolation="periodic",
    )

    assert automatic.ifc_interpolation_resolved == "periodic"
    assert explicit.ifc_interpolation_resolved == "periodic"
    assert automatic.ifc_cache_key != explicit.ifc_cache_key


def test_cache_identity_refreshes_when_lazy_ifc3_support_is_loaded(tmp_path):
    """Loading literal IFC3 support invalidates the earlier lazy namespace."""
    forceconstants = _forceconstants(third_supercell=(2, 1, 1))
    phonons = _phonons(forceconstants, tmp_path)
    lazy_key = phonons.ifc_cache_key
    grid = SupercellGrid(np.diag([2, 1, 1]))
    forceconstants._third = SimpleNamespace(
        translation_support=TranslationSupport(
            [[0, 0, 0], [2, 0, 0]], grid, provenance="file"
        )
    )

    assert phonons.ifc_cache_key != lazy_key
    assert phonons.ifc_cache_key in phonons._add_grid_components("")


def test_periodic_ifc3_lazy_load_adds_exact_support_identity(tmp_path):
    """Even compact support order is part of persistent numerical identity."""
    forceconstants = _forceconstants(third_supercell=(2, 2, 1))
    phonons = _phonons(forceconstants, tmp_path)
    lazy_key = phonons.ifc_cache_key
    forceconstants._third = SimpleNamespace(
        translation_support=TranslationSupport.periodic(
            SupercellGrid(np.diag([2, 2, 1]), order="C")
        )
    )

    assert phonons.ifc_cache_key != lazy_key

    other_forceconstants = _forceconstants(third_supercell=(2, 2, 1))
    other_forceconstants._third = SimpleNamespace(
        translation_support=TranslationSupport.periodic(
            SupercellGrid(np.diag([2, 2, 1]), order="F")
        )
    )
    other = _phonons(other_forceconstants, tmp_path / "other")
    assert other.ifc_cache_key != phonons.ifc_cache_key


def test_wigner_seitz_rejects_nonperiodic_lattice_direction(tmp_path):
    """Shortest-image interpolation is defined only for a 3-D lattice."""
    forceconstants = _forceconstants(pbc=(True, True, False))

    with pytest.raises(ValueError, match="periodic boundary conditions.*three"):
        _phonons(
            forceconstants,
            tmp_path,
            ifc_interpolation="wigner-seitz",
        )
