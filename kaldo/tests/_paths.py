"""
Centralized environment-variable lookup for production-only test fixtures.

Several cumulant and non-diagonal SNF tests need DFT-quality Si IFCs (n_uc=2,
production 25^3 supercell) or Ethan Meitz's Ne LJ IFCs from his TDEP run.
These fixtures are too large to vendor in the repo (hundreds of MB for IFC4
alone) and are not portable: they live on the maintainer's davinci-labs
workstation. The tests below are gated on env vars set in your shell init:

    KALDO_TEST_SI_PROD          # path to reference_si/T300_0
    KALDO_TEST_NE_REF           # path to thermo_out_full
    KALDO_TEST_SI_NE_H5         # path to ifcs_sc_remapped.h5
    KALDO_TEST_CUMULANT_SAMPLES # path to phase5_our_samples.npz

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
NE_IFC_H5 = _path_from_env("KALDO_TEST_SI_NE_H5")
CUMULANT_SAMPLES = _path_from_env("KALDO_TEST_CUMULANT_SAMPLES")
