"""
Pin the kaldo.cumulant public API surface.

This test documents which symbols are exported from kaldo.cumulant today.
A future cleanup that removes or renames a public name must update this
test as well, making accidental API breakage visible in the diff.
"""
from __future__ import annotations


EXPECTED_PUBLIC_SYMBOLS = {
    # cumulant-specific constants
    "NE_MASS_AMU", "FREQ_TOL_THZ",
    # one-shot TDEP folder reader
    "load_tdep_folder",
    # kaldo-native entry points (recommended)
    "F1_from_fc", "F2_from_fc",
    # legacy list-based math kernels (kept as cross-check)
    "F1_vectorized", "F2_vectorized",
    # MC pipeline
    "SCSampler", "SCContractors",
    "calculate_cumulants", "bootstrap_corrections",
    # top-level API
    "cumulant_thermo", "CumulantResult", "print_thermo_table",
}


def test_cumulant_package_public_surface_stable():
    """kaldo.cumulant.__all__ contains exactly the expected public names."""
    import kaldo.cumulant as cum
    actual = set(cum.__all__)
    missing = EXPECTED_PUBLIC_SYMBOLS - actual
    extra = actual - EXPECTED_PUBLIC_SYMBOLS
    assert not missing, f"missing public symbols: {sorted(missing)}"
    assert not extra, (
        f"unexpected public symbols (remove or add to EXPECTED): {sorted(extra)}"
    )


def test_cumulant_package_all_symbols_importable():
    """Every __all__ entry must actually be importable from the package."""
    import kaldo.cumulant as cum
    for name in cum.__all__:
        assert hasattr(cum, name), f"kaldo.cumulant.{name} not defined"
