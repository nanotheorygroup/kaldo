"""Picklable factory for deferred ASE calculator construction.

`CalculatorFactory` wraps ``(calculator_class, args, kwargs)`` into a
zero-argument callable that builds a fresh calculator instance on demand. It
is picklable, so it survives transport across `ProcessPoolExecutor` and
`MPIPoolExecutor` workers that cannot pickle live calculator instances (e.g.
C++-backed or GPU-bound calculators).

It drops into `calculate_second` / `calculate_third`'s existing
``calculator=`` slot with no other changes::

    from kaldo.parallel import CalculatorFactory
    from pynep.calculate import NEP

    calculator = CalculatorFactory(NEP, args=('nep.txt',),
                                   kwargs={'kwarg1': 'value1', ...})
    second_order.calculate(calculator=calculator, n_workers=None)

Best practice is to fit all your arguments into the kwarg dictionary,
but if you know the argument order exactly you can put them in the args
list.
"""


class CalculatorFactory:
    """Callable wrapper that constructs a fresh calculator on each call.

    Parameters
    ----------
    calculator_class : type or callable
        The calculator class (or any zero-or-more-arg callable) that, when
        invoked with ``args`` and ``kwargs``, returns a calculator instance.
    args : tuple, optional
        Positional arguments forwarded to ``calculator_class`` on every call.
        Default ``()``.
    kwargs : dict, optional
        Keyword arguments forwarded to ``calculator_class`` on every call.
        Default ``None`` (treated as empty dict).
    validate : bool, optional
        If True (default), construct one instance in ``__init__`` and discard
        it so that bad arguments raise on the main process. Set False to
        defer any construction errors to worker-side invocation — useful when
        construction is slow (heavy ML calculators) or has side effects
        (GPU allocation) you want to happen only inside workers.

    Examples
    --------
    >>> from ase.calculators.emt import EMT
    >>> factory = CalculatorFactory(EMT)
    >>> calc = factory()             # fresh EMT instance

    >>> factory = CalculatorFactory(EMT, kwargs={'asap_cutoff': True})
    >>> factory = CalculatorFactory(MyCalc, args=('model.pt',),
    ...                             kwargs={'device': 'cuda'})
    """

    def __init__(self, calculator_class, args=(), kwargs=None, validate=True):
        if not callable(calculator_class):
            raise TypeError(
                f"calculator_class must be callable, got "
                f"{type(calculator_class).__name__}"
            )
        self.calculator_class = calculator_class
        self.args = tuple(args)
        self.kwargs = dict(kwargs) if kwargs else {}
        if validate:
            self.calculator_class(*self.args, **self.kwargs)

    def __call__(self):
        return self.calculator_class(*self.args, **self.kwargs)

    def __repr__(self):
        cls_name = getattr(self.calculator_class, '__name__',
                           repr(self.calculator_class))
        return (f"CalculatorFactory({cls_name}, args={self.args!r}, "
                f"kwargs={self.kwargs!r})")
