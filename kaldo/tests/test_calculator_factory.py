"""Unit tests for `kaldo.parallel.CalculatorFactory`."""

import pickle
import pytest
from kaldo.parallel import CalculatorFactory


class _Stub:
    """Minimal stand-in for a calculator class (picklable, constructor-argful)."""

    def __init__(self, model='default', device='cpu'):
        self.model = model
        self.device = device


class _PositionalStub:
    def __init__(self, path):
        self.path = path


def test_kwargs_only():
    factory = CalculatorFactory(_Stub, kwargs={'model': 'm1', 'device': 'cuda'})
    instance = factory()
    assert instance.model == 'm1'
    assert instance.device == 'cuda'


def test_args_only():
    factory = CalculatorFactory(_PositionalStub, args=('nep.txt',))
    instance = factory()
    assert instance.path == 'nep.txt'


def test_args_and_kwargs():
    factory = CalculatorFactory(_Stub, args=('mymodel',), kwargs={'device': 'cuda'})
    instance = factory()
    assert instance.model == 'mymodel'
    assert instance.device == 'cuda'


def test_validate_true_raises_on_bad_kwargs():
    with pytest.raises(TypeError):
        CalculatorFactory(_Stub, kwargs={'not_a_real_kwarg': 42})


def test_validate_false_defers_error_to_call():
    factory = CalculatorFactory(_Stub, kwargs={'not_a_real_kwarg': 42}, validate=False)
    with pytest.raises(TypeError):
        factory()


def test_non_callable_class_rejected():
    with pytest.raises(TypeError, match="calculator_class must be callable"):
        CalculatorFactory(42)


def test_pickle_round_trip():
    factory = CalculatorFactory(_Stub, kwargs={'model': 'm1'})
    restored = pickle.loads(pickle.dumps(factory))
    assert restored.calculator_class is _Stub
    assert restored.args == ()
    assert restored.kwargs == {'model': 'm1'}
    instance = restored()
    assert instance.model == 'm1'


def test_two_calls_produce_distinct_instances():
    factory = CalculatorFactory(_Stub)
    a = factory()
    b = factory()
    assert a is not b
