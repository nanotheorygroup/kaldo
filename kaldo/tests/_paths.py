"""
Centralized environment-variable lookup for production-only test fixtures.

Several non-diagonal (SNF) supercell tests need DFT-quality Si IFCs
(n_uc=2, rhombohedral primitive + cubic conventional ssposcar) or a Ne LJ
TDEP run. These fixtures are too large to vendor in the repo (hundreds of
MB for IFC4 alone). The tests are gated on env vars set in your shell:

    KALDO_TEST_SI_PROD          # path to a non-diagonal Si TDEP folder
    KALDO_TEST_NE_REF           # path to a non-diagonal Ne TDEP folder

When unset, tests skip cleanly. To run the full suite with production
fixtures available, set these in your environment before invoking pytest.

This is a tracked technical debt: future work is to vendor smaller fixtures
or to host the production fixtures in a stable location (e.g. Zenodo) and
download in conftest.py.
"""
from __future__ import annotations

import os
from pathlib import Path


_NONEXISTENT = Path("/__kaldo_test_fixture_unset__")


def _path_from_env(var: str) -> Path:
    """Return Path from env var; returns a definitely-nonexistent sentinel
    Path when unset (so .exists() reliably returns False)."""
    val = os.environ.get(var, "")
    if not val:
        return _NONEXISTENT
    return Path(val)


SI_PROD = _path_from_env("KALDO_TEST_SI_PROD")
NE_REF = _path_from_env("KALDO_TEST_NE_REF")
