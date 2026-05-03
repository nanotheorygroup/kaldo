---
root: false
targets: ["*"]
description: Dependency management
globs: ["**/*.py", "**/requirements*.txt", "**/pyproject.toml", "**/setup.py", "**/setup.cfg"]
---

# Dependencies

## The dependency manifest is the source of truth

The project's dependency manifest (`requirements.txt`, `pyproject.toml`, or `setup.py`) lists every package the code depends on. If you add an import, add the package to the manifest in the same change.

If you remove the last user of a package, remove it from the manifest.

## Don't use a package without listing it

Avoid this trap: a heavy framework (TensorFlow, PyTorch, JAX) is installed in your environment but not in the manifest. Your code imports it; tests pass for you; another developer with a fresh environment hits an `ImportError`.

If your editor offers an autocomplete from a package not in the manifest, stop and add it first.

## Optional dependencies

If a package is genuinely optional (e.g., GPU acceleration that not every user installs), wrap the import:

```python
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
```

Document the optional dependency in the manifest under an extras section (`pyproject.toml`'s `[project.optional-dependencies]` or `setup.py`'s `extras_require`).

When a function requires the optional dep, fail fast with a clear error:

```python
def fancy_thing():
    if not HAS_TF:
        raise ImportError("fancy_thing requires tensorflow. Install with: pip install <pkg>[gpu]")
    ...
```

## Version pinning

- Library code (a package others install): use `>=` lower bounds, not `==`. Don't pin tighter than your code actually requires.
- Application code (a service deployed somewhere): pin exactly via a lockfile (`uv.lock`, `poetry.lock`, or `pip-tools` output). Reproducible installs matter.

## Project conventions (kaldo)

- The manifest is `requirements.txt` at the repo root; `setup.py` reads it via `f.read().splitlines()`. Add new runtime deps to `requirements.txt`, not `setup.py`.
- kaldo is a library: keep `>=` lower bounds, not `==` pins.
- TensorFlow is required (not optional) per current `requirements.txt`. If you make a code path TF-only, document the GPU vs CPU behavior in the docstring; do not gate the import behind `try/except` and pretend it's optional.
