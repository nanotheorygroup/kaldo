---
root: false
targets: ["*"]
description: Python code organization
globs: ["**/*.py"]
---

# Code Organization

## One responsibility per module

Each module should have one clear purpose, expressible in a one-line description. Module names are nouns describing what the module *contains*, not what it *does*: `transport.py`, `forces.py`, `harmonic.py`, not `compute_transport.py`.

If you cannot summarize a module in one sentence, it's doing too much. Split it.

## Package `__init__.py`

The package `__init__.py` re-exports the public API. Keep it short.

```python
from .transport import compute_conductivity
from .forces import ForceConstants

__all__ = ["compute_conductivity", "ForceConstants"]
```

Do not put implementation logic in `__init__.py`. If you need helper code, put it in a dedicated module (`_internal.py`, `_utils.py`) and import from there.

## When to split a module

Split when any of these happens:

- The module exceeds ~400 lines and isn't a single tight algorithm.
- A reader needs to scroll past unrelated code to find what they're looking for.
- Two distinct subjects share the file (e.g., parsers + solvers + I/O).

Split by **responsibility**, not by technical layer. `parsers/quantum_espresso.py` + `parsers/vasp.py` is good. `models.py` + `views.py` + `controllers.py` for a numerical library is not.

## Imports

- Standard library first, then third-party, then first-party. One blank line between groups.
- Avoid wildcard imports (`from foo import *`) outside of `__init__.py` re-exports.
- Avoid circular imports by keeping domain modules independent. If two modules need to import each other, a third module probably belongs between them.

## Private vs public

- Prefix internal-only names with a single underscore (`_load_config`, `_DEFAULT_CUTOFF`).
- Prefix dunders only when overriding Python protocols (`__init__`, `__repr__`).
- Don't expose internals through `__init__.py`.

## Project conventions (kaldo)

- Line length is **119** characters (see `setup.cfg` `[flake8]` and `[yapf]`). Don't reflow to 80 or 100.
- Linting is flake8; formatting is yapf. Don't introduce ruff/black/isort without an explicit decision — they would churn every file.
- Subpackages already reflect responsibility-based organization: `controllers/` (algorithms), `observables/` (tensor objects), `interfaces/` (external code I/O), `parallel/` (worker pools), `helpers/` (utilities). Place new modules in the subpackage that matches their role.
