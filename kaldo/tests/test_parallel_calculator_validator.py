"""
Tests for ``validate_parallel_calculator`` and the API-compatibility
guarantees around it.

Covers:

1. Picklable calculators (EMT) pass the validator.
2. Callable factories (class or ``functools.partial``) pass the validator.
3. Non-picklable instances raise ``TypeError`` with a ``functools.partial``
   hint when ``n_workers > 1``, at every user-facing API entry point.
4. Serial-mode (``n_workers=1``, default) execution is validator-free and
   preserves the dominant kaldo-examples pattern
   (``atoms.calc = calc; fc.second.calculate(calc)``).
"""

import functools
import os

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.controllers.displacement import calculate_second, calculate_third
from kaldo.parallel import validate_parallel_calculator


class _UnpicklableCalculator(EMT):
    """EMT subclass that cannot be pickled, as a stand-in for ML calculators
    (MatterSim, MACE, orb, CPUNEP) that hold weak refs, PyTorch models, or
    C handles."""

    def __reduce__(self):
        raise TypeError("this calculator is intentionally non-picklable (test stand-in)")


def test_validator_accepts_none():
    validate_parallel_calculator(None, method='x')


def test_validator_accepts_callable_class():
    validate_parallel_calculator(EMT, method='x')


def test_validator_accepts_partial_factory():
    validate_parallel_calculator(
        functools.partial(EMT, asap_cutoff=False), method='x'
    )


def test_validator_accepts_picklable_instance():
    validate_parallel_calculator(EMT(), method='x')


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


@pytest.fixture(scope="module")
def al_atoms():
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated = atoms.repeat((1, 1, 2))
    return atoms, replicated


def test_calculate_second_rejects_non_picklable_parallel(al_atoms):
    atoms, replicated = al_atoms
    with pytest.raises(TypeError, match='functools'):
        calculate_second(
            atoms, replicated, 1e-5,
            n_workers=2,
            calculator=_UnpicklableCalculator(),
        )


def test_calculate_third_rejects_non_picklable_parallel(al_atoms):
    atoms, replicated = al_atoms
    with pytest.raises(TypeError, match='functools'):
        calculate_third(
            atoms, replicated, 1e-5,
            n_workers=2,
            calculator=_UnpicklableCalculator(),
        )


def test_calculate_second_accepts_non_picklable_serial(al_atoms):
    """Serial mode (n_workers=1) must not trip the validator — the whole point
    of the validator is that it only fires when multiprocessing is actually used.
    """
    atoms, replicated = al_atoms
    result = calculate_second(
        atoms, replicated, 1e-5,
        n_workers=1,
        calculator=_UnpicklableCalculator(),
    )
    assert result.shape == (1, 4, 3, 2, 4, 3)


def test_calculate_second_parallel_via_partial_matches_serial(al_atoms):
    """The full fix path: wrap a non-picklable calculator in functools.partial
    so each worker builds its own instance. Parallel must produce the same
    tensor as serial.
    """
    atoms, _ = al_atoms
    replicated_serial = atoms.repeat((1, 1, 2))
    replicated_serial.calc = EMT()
    serial = calculate_second(
        atoms, replicated_serial, 1e-5, n_workers=1, calculator=EMT(),
    )

    replicated_parallel = atoms.repeat((1, 1, 2))
    parallel = calculate_second(
        atoms, replicated_parallel, 1e-5,
        n_workers=2,
        calculator=functools.partial(_UnpicklableCalculator),
    )
    np.testing.assert_allclose(
        serial, parallel, rtol=1e-7, atol=1e-9,
        err_msg="Parallel via functools.partial diverges from serial.",
    )


def test_calculate_third_parallel_via_partial_matches_serial(al_atoms):
    atoms, _ = al_atoms
    replicated_serial = atoms.repeat((1, 1, 2))
    replicated_serial.calc = EMT()
    serial = calculate_third(
        atoms, replicated_serial, 1e-5, n_workers=1, calculator=EMT(),
    )

    replicated_parallel = atoms.repeat((1, 1, 2))
    parallel = calculate_third(
        atoms, replicated_parallel, 1e-5,
        n_workers=2,
        calculator=functools.partial(_UnpicklableCalculator),
    )
    np.testing.assert_allclose(
        serial.todense(), parallel.todense(), rtol=1e-7,
        err_msg="Parallel third via functools.partial diverges from serial.",
    )


def test_calculate_second_detects_unpicklable_atoms_calc(al_atoms):
    """When calculator=None and replicated_atoms.calc is non-picklable, the
    parallel guard must still fire (the attached instance gets pickled with
    replicated_atoms when shipped to workers)."""
    atoms, _ = al_atoms
    replicated = atoms.repeat((1, 1, 2))
    replicated.calc = _UnpicklableCalculator()
    with pytest.raises(TypeError, match='functools'):
        calculate_second(atoms, replicated, 1e-5, n_workers=2, calculator=None)


def _reference_second(atoms, replicated):
    """Compute the reference second-order tensor in strictly serial mode."""
    replicated_ref = replicated.copy()
    replicated_ref.calc = EMT()
    return calculate_second(
        atoms, replicated_ref, 1e-5, n_workers=1, calculator=EMT(),
    )


def test_kaldo_examples_pattern_second_is_byte_compatible(al_atoms):
    """Reproduce the dominant kaldo-examples usage exactly:

        atoms.calc = calc
        fc.second.calculate(calc, delta_shift=...)

    (which reduces to calculate_second(..., n_workers=1, calculator=calc))
    and verify the tensor matches a clean serial baseline.
    """
    atoms, replicated_pristine = al_atoms
    replicated_user = replicated_pristine.copy()
    replicated_user.calc = EMT()  # user attaches to atoms
    user_result = calculate_second(
        atoms, replicated_user, 1e-5, n_workers=1, calculator=EMT(),
    )
    reference = _reference_second(atoms, replicated_pristine)
    np.testing.assert_allclose(
        user_result, reference, rtol=1e-7, atol=1e-9,
        err_msg="Serial kaldo-examples pattern regressed vs reference.",
    )


def test_serial_instance_does_not_rebind_per_atom(monkeypatch, tmp_path):
    """Serial + calculator instance must attach the calculator to atoms once,
    not re-attach per atom iteration. Some calculator libraries require a
    calculator to stay bound to a single atoms object over its lifetime.
    """
    import ase.atoms

    from kaldo.forceconstants import ForceConstants

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


def test_third_order_serial_does_not_auto_resolve_scratch_dir(monkeypatch, tmp_path):
    """Serial-mode ``ThirdOrder.calculate`` must not silently pass a
    ``{folder}/third_order/`` path to ``calculate_third``. Verified by
    monkeypatching ``calculate_third`` to capture its ``scratch_dir`` arg.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables import thirdorder as thirdorder_mod

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    folder = tmp_path / "kaldo_out"
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=str(folder))

    captured = {}

    def fake_calc_third(*args, **kwargs):
        captured['scratch_dir'] = kwargs.get('scratch_dir')
        captured['n_workers'] = kwargs.get('n_workers')
        # return a placeholder sparse tensor of the expected shape
        import sparse
        import numpy as np
        n_atoms = len(atoms.numbers)
        n_replicas = 2
        shape = (n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3)
        return sparse.COO(np.empty((3, 0), dtype=np.int64), np.array([]), shape=shape)

    monkeypatch.setattr(thirdorder_mod, 'calculate_third', fake_calc_third)

    fc.third.calculate(
        calculator=EMT(),
        delta_shift=1e-5,
        is_storing=False,
        n_workers=1,
    )
    assert captured['n_workers'] == 1
    assert captured['scratch_dir'] is None, (
        f"Serial-mode ThirdOrder.calculate should pass scratch_dir=None to "
        f"calculate_third, got {captured['scratch_dir']!r}."
    )


def test_third_order_parallel_auto_resolves_scratch_dir(monkeypatch, tmp_path):
    """Parallel-mode ``ThirdOrder.calculate`` must auto-resolve ``scratch_dir``
    to ``{folder}/third_order`` so workers have a place to flush chunks."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables import thirdorder as thirdorder_mod

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    folder = tmp_path / "kaldo_out"
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=str(folder))

    captured = {}

    def fake_calc_third(*args, **kwargs):
        captured['scratch_dir'] = kwargs.get('scratch_dir')
        import sparse
        import numpy as np
        n_atoms = len(atoms.numbers)
        shape = (n_atoms * 3, 2 * n_atoms * 3, 2 * n_atoms * 3)
        return sparse.COO(np.empty((3, 0), dtype=np.int64), np.array([]), shape=shape)

    monkeypatch.setattr(thirdorder_mod, 'calculate_third', fake_calc_third)

    fc.third.calculate(
        calculator=EMT,  # factory so the parallel guard is happy
        delta_shift=1e-5,
        is_storing=False,
        n_workers=2,
    )
    assert captured['scratch_dir'] == os.path.join(str(folder), 'third_order')


def test_second_order_parallel_auto_resolves_scratch_dir(monkeypatch, tmp_path):
    """Parallel-mode ``SecondOrder.calculate`` must auto-resolve ``scratch_dir``
    to ``{folder}/second_order`` for symmetry with the third-order API."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables import secondorder as secondorder_mod

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    folder = tmp_path / "kaldo_out"
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=str(folder))

    captured = {}

    def fake_calc_second(*args, **kwargs):
        captured['scratch_dir'] = kwargs.get('scratch_dir')
        n_atoms = len(atoms.numbers)
        return np.zeros((1, n_atoms, 3, 2, n_atoms, 3))

    monkeypatch.setattr(secondorder_mod, 'calculate_second', fake_calc_second)

    fc.second.calculate(
        calculator=EMT,
        delta_shift=1e-5,
        is_storing=False,
        n_workers=2,
    )
    assert captured['scratch_dir'] == os.path.join(str(folder), 'second_order')


def test_second_order_serial_does_not_auto_resolve_scratch_dir(monkeypatch, tmp_path):
    """Serial-mode ``SecondOrder.calculate`` must not silently pass a
    ``{folder}/second_order/`` path to ``calculate_second``, matching the
    third-order behaviour."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables import secondorder as secondorder_mod

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    folder = tmp_path / "kaldo_out"
    fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=str(folder))

    captured = {}

    def fake_calc_second(*args, **kwargs):
        captured['scratch_dir'] = kwargs.get('scratch_dir')
        n_atoms = len(atoms.numbers)
        return np.zeros((1, n_atoms, 3, 2, n_atoms, 3))

    monkeypatch.setattr(secondorder_mod, 'calculate_second', fake_calc_second)

    fc.second.calculate(
        calculator=EMT(),
        delta_shift=1e-5,
        is_storing=False,
        n_workers=1,
    )
    assert captured['scratch_dir'] is None


def test_third_order_accepts_calculator_none():
    """``ThirdOrder.calculate(calculator=None)`` must not raise. When None,
    ``replicated_atoms.calc`` supplies the calculator — consistent with how
    ``SecondOrder.calculate`` behaves and how ``calculate_third`` works."""
    from kaldo.forceconstants import ForceConstants
    import tempfile

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    with tempfile.TemporaryDirectory() as folder:
        fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=folder)
        fc.third.replicated_atoms.calc = EMT()
        # Must not raise; result correctness is covered elsewhere.
        fc.third.calculate(
            calculator=None,
            delta_shift=1e-4,
            is_storing=False,
            n_workers=1,
        )


def test_second_order_accepts_calculator_none():
    """``SecondOrder.calculate(calculator=None)`` with a pre-attached
    ``replicated_atoms.calc`` must complete and write ``replicated_atoms.xyz``
    without crashing in the post-calc ``get_forces()`` step.
    """
    from kaldo.forceconstants import ForceConstants
    import tempfile

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    with tempfile.TemporaryDirectory() as folder:
        fc = ForceConstants(atoms=atoms, supercell=(1, 1, 2), folder=folder)
        fc.second.replicated_atoms.calc = EMT()
        # Must not raise; previously this re-attached ``None`` to ``.calc`` and
        # then called ``get_forces()`` on an empty calculator.
        fc.second.calculate(
            calculator=None,
            delta_shift=1e-5,
            is_storing=True,
            n_workers=1,
        )
        assert os.path.exists(os.path.join(folder, 'replicated_atoms.xyz'))
