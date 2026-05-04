---
root: false
targets: ["*"]
description: Testing discipline (pytest)
globs: ["**/*.py", "**/test_*.py", "**/*_test.py"]
---

# Testing

The default test runner is **pytest**. Run with `pytest` from the project root or `python -m pytest` if pytest isn't on PATH.

## Naming and structure

- Test files: `test_<module>.py` or `<module>_test.py`.
- Test functions: `test_<behavior>` describing the behavior under test, not the implementation.
- One assertion concept per test. Multiple `assert` lines are fine if they verify the same concept.

## Fixtures over setup/teardown

Prefer `@pytest.fixture` over `setUp`/`tearDown`. Fixtures compose; class-based setup does not.

```python
@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_something(sample_data):
    assert sample_data["key"] == "value"
```

## Filesystem isolation

Use the built-in `tmp_path` fixture for any test that creates files. Never write into the working directory or hardcoded `/tmp` paths.

```python
def test_writes_output(tmp_path):
    out = tmp_path / "result.txt"
    write_result(out)
    assert out.read_text() == "expected"
```

## Parametrize for variants

When the same logic should work for many inputs, use `@pytest.mark.parametrize` instead of duplicating tests.

```python
@pytest.mark.parametrize("temp,expected", [(0, 0.0), (300, 0.025), (1000, 0.087)])
def test_thermal_conductivity(temp, expected):
    assert abs(conductivity(temp) - expected) < 1e-3
```

## Real I/O over mocks

For code that touches files (HDF5, NumPy `.npy`, plain text, JSON), prefer writing real fixture files in `tmp_path` over mocking the filesystem. Mocks of file APIs are brittle and hide real bugs.

Mock only what you cannot exercise: external HTTP services, expensive third-party APIs, hardware that isn't available on a test machine.

## What not to test

- Don't test third-party libraries. Trust that numpy, scipy, etc. work.
- Don't test trivial getters/setters or pure delegation.
- Don't write tests purely for coverage. Each test should describe a real behavior.

## Project conventions (kaldo)

- Tests live in `kaldo/tests/`, named `test_*.py`. Reference materials sit in sibling directories (e.g. `kaldo/tests/si-crystal/`, `kaldo/tests/mgo/`).
- Reference data files (force constants, dynamical matrices) are read directly in tests — do not mock the filesystem to substitute for them.
- Numerical comparisons use `np.testing.assert_allclose` with explicit `rtol`/`atol`. Pick tolerances that reflect the physics, not whatever makes the test pass today.
- Coverage config lives in `setup.cfg` `[coverage:run]`; the test directory and `kaldo/_version.py` are omitted.
