---
root: false
targets: ["*"]
description: Add pytest tests for a function or module
---

# /add-test

Given a function or module path, add pytest tests covering the happy path and at least one edge case.

## Inputs

- Path to a Python file or a `module.function` reference.
- Optional: a brief description of what behavior to focus on.

## Process

1. Read the target. Identify its inputs, outputs, and any documented edge cases.
2. Locate the existing test file under `kaldo/tests/test_<module>.py`. If none exists, create one there (kaldo keeps tests inside the package, not at repo root).
3. Write at least:
   - One happy-path test demonstrating the documented behavior.
   - One edge-case test (empty input, boundary value, error condition).
4. Use `tmp_path` for any filesystem interaction.
5. Use `@pytest.mark.parametrize` if the function naturally takes a range of inputs.
6. Run the new tests; confirm they pass against the current implementation.
7. Optionally, write a test that *should* fail given a likely bug (off-by-one, missing nan handling, etc.) and report whether the implementation catches it.

## Output

- The test file with new tests added.
- The pytest output showing the new tests passing.
- A short note on any tests that revealed unexpected behavior.

## What to avoid

- Mocking things that can be exercised directly with `tmp_path` or fixture data.
- Asserting on private internal state. Test the public contract.
- Tests that just exercise lines without asserting anything meaningful.
