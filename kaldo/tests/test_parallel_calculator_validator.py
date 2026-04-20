"""
Tests for ``validate_parallel_calculator`` and the API-compatibility
guarantees around it.

Covers:

1. Picklable calculators (EMT classes, instances, and ``functools.partial``) pass the validator.
2. Non-picklable instances raise ``TypeError`` with a ``functools.partial``
   hint when ``n_workers > 1``, at every user-facing API entry point.
3. Serial-mode (``n_workers=1``, default) execution is validator-free and
   preserves the dominant kaldo-examples pattern
   (``atoms.calc = calc; fc.second.calculate(calc)``).
4. ``SecondOrder.calculate`` and ``ThirdOrder.calculate`` behave symmetrically:
   both accept ``calculator=None`` (relying on ``replicated_atoms.calc``),
   both auto-resolve ``scratch_dir`` in parallel runs only.
"""

import functools
import os
import tempfile

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.controllers.displacement import calculate_second, calculate_third
from kaldo.forceconstants import ForceConstants
from kaldo.parallel import validate_parallel_calculator


class _UnpicklableCalculator(EMT):
    """EMT subclass that cannot be pickled, as a stand-in for ML calculators
    (MatterSim, MACE, orb, CPUNEP) that hold weak refs, PyTorch models, or
    C handles."""

    def __reduce__(self):
        raise TypeError("this calculator is intentionally non-picklable (test stand-in)")


@pytest.fixture(scope="module")
def al_atoms():
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated = atoms.repeat((1, 1, 2))
    return atoms, replicated


# -- Validator unit tests ----------------------------------------------------

@pytest.mark.parametrize('calc', [
    None,
    EMT,                                            # class / factory
    functools.partial(EMT, asap_cutoff=False),      # partial factory
    EMT(),                                          # picklable instance
])
def test_validator_accepts_valid_calculators(calc):
    validate_parallel_calculator(calc, method='x')


def test_validator_rejects_non_picklable_instance():
    with pytest.raises(TypeError) as excinfo:
        validate_parallel_calculator(
            _UnpicklableCalculator(), method='SecondOrder.calculate'
        )
    msg = str(excinfo.value)
    assert 'SecondOrder.calculate' in msg
    assert 'functools' in msg and 'partial' in msg
    assert '_UnpicklableCalculator' in msg
    assert 'n_workers=1' in msg


# -- Validator integration: rejects at every parallel entry point ------------

@pytest.mark.parametrize('calculate_fn', [calculate_second, calculate_third],
                         ids=['second', 'third'])
def test_parallel_rejects_non_picklable_calculator(al_atoms, calculate_fn):
    atoms, replicated = al_atoms
    with pytest.raises(TypeError, match='functools'):
        calculate_fn(atoms, replicated, 1e-5,
                     n_workers=2,
                     calculator=_UnpicklableCalculator())


def test_parallel_rejects_non_picklable_atoms_calc(al_atoms):
    """When calculator=None and replicated_atoms.calc is non-picklable, the
    parallel guard must still fire (the attached instance gets pickled with
    replicated_atoms when shipped to workers)."""
    atoms, _ = al_atoms
    replicated = atoms.repeat((1, 1, 2))
    replicated.calc = _UnpicklableCalculator()
    with pytest.raises(TypeError, match='functools'):
        calculate_second(atoms, replicated, 1e-5, n_workers=2, calculator=None)


# -- Parallel correctness: non-picklable calc via functools.partial ---------

@pytest.mark.parametrize('calculate_fn,densify,atol', [
    (calculate_second, lambda x: x,          1e-9),
    (calculate_third,  lambda x: x.todense(), 0.0),
], ids=['second', 'third'])
def test_parallel_via_partial_matches_serial(al_atoms, calculate_fn, densify, atol):
    """The full fix path: wrap a non-picklable calculator in functools.partial
    so each worker builds its own instance. Parallel must match serial."""
    atoms, _ = al_atoms
    replicated_serial = atoms.repeat((1, 1, 2))
    replicated_serial.calc = EMT()
    serial = calculate_fn(atoms, replicated_serial, 1e-5,
                          n_workers=1, calculator=EMT())

    replicated_parallel = atoms.repeat((1, 1, 2))
    parallel = calculate_fn(atoms, replicated_parallel, 1e-5,
                            n_workers=2,
                            calculator=functools.partial(_UnpicklableCalculator))

    np.testing.assert_allclose(
        densify(serial), densify(parallel), rtol=1e-7, atol=atol,
        err_msg="Parallel via functools.partial diverges from serial.",
    )


# -- Serial-mode guarantees --------------------------------------------------

def test_serial_instance_does_not_rebind_per_atom(monkeypatch, tmp_path):
    """Serial + calculator instance must attach the calculator to atoms once,
    not re-attach per atom iteration. Some calculator libraries require a
    calculator to stay bound to a single atoms object over its lifetime.
    """
    import ase.atoms

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=str(tmp_path))
    target = fc.second.replicated_atoms
    rebind_log = []
    original_setattr = ase.atoms.Atoms.__setattr__

    def tracking_setattr(self, name, value):
        if name == 'calc' and self is target:
            rebind_log.append(type(value).__name__)
        original_setattr(self, name, value)

    monkeypatch.setattr(ase.atoms.Atoms, '__setattr__', tracking_setattr)
    fc.second.calculate(calculator=EMT(), delta_shift=1e-5, is_storing=False, n_workers=1)
    assert len(rebind_log) == 1, (
        f"Serial-mode SecondOrder.calculate rebound replicated_atoms.calc "
        f"{len(rebind_log)} times; expected 1. Log: {rebind_log}"
    )


# -- API symmetry between SecondOrder and ThirdOrder -------------------------

_OBSERVABLES = [
    ('second', 'second_order'),
    ('third',  'third_order'),
]


@pytest.mark.parametrize('observable,scratch_name', _OBSERVABLES)
@pytest.mark.parametrize('n_workers,expected', [
    (1, None),                    # serial: no auto-scratch
    (2, 'SCRATCH_AUTO'),          # parallel: auto-resolves to {folder}/{scratch_name}
], ids=['serial', 'parallel'])
def test_auto_scratch_dir_symmetry(monkeypatch, tmp_path, observable, scratch_name,
                                   n_workers, expected):
    """Serial runs must not auto-resolve scratch_dir; parallel runs must
    resolve to ``{folder}/{observable}_order``. Applied to both second and third."""
    from kaldo.observables import secondorder as secondorder_mod
    from kaldo.observables import thirdorder as thirdorder_mod

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    folder = tmp_path / "kaldo_out"
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=str(folder))
    n_atoms = len(atoms.numbers)

    captured = {}

    def fake_second(*args, **kwargs):
        captured['scratch_dir'] = kwargs.get('scratch_dir')
        return np.zeros((1, n_atoms, 3, 2, n_atoms, 3))

    def fake_third(*args, **kwargs):
        captured['scratch_dir'] = kwargs.get('scratch_dir')
        import sparse
        shape = (n_atoms * 3, 2 * n_atoms * 3, 2 * n_atoms * 3)
        return sparse.COO(np.empty((3, 0), dtype=np.int64), np.array([]), shape=shape)

    monkeypatch.setattr(secondorder_mod, 'calculate_second', fake_second)
    monkeypatch.setattr(thirdorder_mod,  'calculate_third',  fake_third)

    calc_arg = EMT if n_workers > 1 else EMT()  # factory for parallel, instance for serial
    getattr(fc, observable).calculate(
        calculator=calc_arg, delta_shift=1e-5, is_storing=False, n_workers=n_workers,
    )

    want = None if expected is None else os.path.join(str(folder), scratch_name)
    assert captured['scratch_dir'] == want


def test_second_calculate_accepts_calculator_none():
    """``SecondOrder.calculate(calculator=None)`` with a pre-attached
    ``replicated_atoms.calc`` must complete and write replicated_atoms.xyz
    without crashing in the post-calc ``get_forces()`` step.
    """
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    with tempfile.TemporaryDirectory() as folder:
        fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=folder)
        fc.second.replicated_atoms.calc = EMT()
        fc.second.calculate(
            calculator=None, delta_shift=1e-5, is_storing=True, n_workers=1,
        )
        assert os.path.exists(os.path.join(folder, 'replicated_atoms.xyz'))


def test_third_calculate_accepts_calculator_none():
    """``ThirdOrder.calculate(calculator=None)`` must not raise. When None,
    ``replicated_atoms.calc`` supplies the calculator — consistent with
    ``SecondOrder.calculate`` and ``calculate_third``.
    """
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    with tempfile.TemporaryDirectory() as folder:
        fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=folder)
        fc.third.replicated_atoms.calc = EMT()
        fc.third.calculate(
            calculator=None, delta_shift=1e-4, is_storing=False, n_workers=1,
        )
