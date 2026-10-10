"""Opt-in comparison with saved amorphous QHGK arrays, without TensorFlow.

Run ``python -m kaldo.amorphous.validate_512 EXAMPLE_PATH --volume ANGSTROM3``.
The root may contain nested kALDo property folders, but each required property
name must be unique. The saved ``bandwidth`` is divided by two, matching the
legacy QHGK fallback; use ``--direct-bandwidth`` if it already represents the
explicit ``diffusivity_bandwidth`` input. This script never writes benchmark
data and is intentionally excluded from the default test suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import resource

import ase.units as units
import numpy as np
from scipy import sparse

from .qhgk import (NpyEigensystemSource, calculate_qhgk_isotropic_blocked,
                   generalized_heat_capacity_block)
from .sparse_harmonic import SparseAmorphousHarmonic


def _file(root, stem, overrides=None):
    if overrides and stem in overrides:
        chosen = Path(overrides[stem])
        chosen = chosen if chosen.is_absolute() else root / chosen
        if not chosen.is_file():
            raise ValueError(f"explicit --file selection does not exist: {chosen}")
        return chosen
    matches = list(root.rglob(stem + ".npy"))
    if len(matches) > 1:
        raise ValueError(
            f"expected one {stem}.npy below {root}, found {len(matches)}; "
            f"use --file {stem}=RELATIVE_PATH to select one")
    if not matches:
        raise FileNotFoundError(f"no {stem}.npy below {root}")
    return matches[0]


def _error(actual, reference):
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("comparison arrays contain nonfinite values")
    difference = np.asarray(actual, dtype=np.float64) - reference
    scale = max(np.max(np.abs(reference)), 1e-30)
    return {"max_abs": float(np.max(np.abs(difference))),
            "max_rel_to_peak": float(np.max(np.abs(difference)) / scale),
            "rms_rel_to_peak": float(np.sqrt(np.mean(difference ** 2)) / scale)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--volume", type=float, required=True, help="cell volume in Å³")
    parser.add_argument("--temperature", type=float, default=300.)
    parser.add_argument("--shape", choices=("triangle", "gauss", "lorentz"), default="lorentz")
    parser.add_argument("--classic", action="store_true")
    parser.add_argument("--direct-bandwidth", action="store_true")
    parser.add_argument("--file", action="append", default=[], metavar="STEM=PATH",
                        help="select a property file explicitly, relative to PATH")
    parser.add_argument("--sparse-harmonic", type=Path,
                        help="optional atom-block sparse directory to compare")
    parser.add_argument("--saved-rtol", type=float, default=2e-6,
                        help="max error relative to saved peak (default: 2e-6)")
    args = parser.parse_args(argv)
    root = args.path
    if not root.is_dir():
        parser.error(f"saved-data directory is absent: {root}")
    overrides = {}
    for item in args.file:
        stem, separator, name = item.partition("=")
        if not separator or not stem or not name:
            parser.error("--file requires STEM=PATH")
        overrides[stem] = name
    selected = lambda stem: _file(root, stem, overrides)
    source = NpyEigensystemSource(selected("_eigensystem"))
    try:
        omega = np.load(selected("omega"), mmap_mode="r").reshape(-1)
    except FileNotFoundError:
        omega = np.load(selected("frequency"), mmap_mode="r").reshape(-1) * (2 * np.pi)
    physical = np.load(selected("physical_mode"), mmap_mode="r").reshape(-1)
    bandwidth = np.load(selected("bandwidth"), mmap_mode="r").reshape(-1)
    if not args.direct_bandwidth:
        bandwidth = bandwidth / 2
    operators = tuple(np.load(selected("_dynmat_derivatives_" + axis), mmap_mode="r")
                      for axis in "xyz")
    saved_flux = tuple(np.load(selected("_sij_" + axis), mmap_mode="r")
                       for axis in "xyz")
    saved_heat = np.load(selected("heat_capacity_2d"), mmap_mode="r")
    saved_conductivity = np.load(selected("conductivity"), mmap_mode="r")
    expected = np.trace(saved_conductivity.reshape(-1, 3, 3), axis1=1, axis2=2) / 3
    if source.n_modes != len(omega) or saved_heat.shape != (len(omega), len(omega)):
        raise ValueError("saved arrays have inconsistent mode counts")

    # Compare sampled 64-column/row blocks; no projected global flux is made.
    e0 = source.read_modes(0, min(64, len(omega)), dtype=np.float64)
    e1 = source.read_modes(64, min(128, len(omega)), dtype=np.float64)
    for axis, operator, flux in zip("xyz", operators, saved_flux):
        projected = e0.T @ (operator @ e1)
        error = _error(projected, flux[:e0.shape[1], 64:64+e1.shape[1]])
        print(f"flux {axis}: {error}")
        if error["max_rel_to_peak"] > args.saved_rtol:
            raise AssertionError(f"saved flux {axis} mismatch")
    heat = generalized_heat_capacity_block(
        omega[:64], omega[64:128], physical[:64], physical[64:128],
        temperature=args.temperature, is_classic=args.classic, dtype=np.float64)
    heat_error = _error(heat * (units.kB / units.J), saved_heat[:64, 64:128])
    print("heat_capacity_2d:", heat_error)
    if heat_error["max_rel_to_peak"] > args.saved_rtol:
        raise AssertionError("saved heat capacity mismatch")

    common = dict(omega=omega, diffusivity_bandwidth=bandwidth,
                  physical_mode=physical, temperature=args.temperature,
                  is_classic=args.classic, volume=args.volume, eigenvectors=source,
                  harmonic_operator=operators, broadening_shape=args.shape)
    results = {}
    for size in (16, 64, 257, 512, len(omega)):
        result = calculate_qhgk_isotropic_blocked(**common, block_size=size,
                                                   compute_dtype=np.float64)[0]
        results[size] = result
        print(f"float64 block={size}: saved={_error(result, expected)} "
              f"block64={_error(result, results.get(64, result))}")
        if _error(result, expected)["max_rel_to_peak"] > args.saved_rtol:
            raise AssertionError(f"saved conductivity mismatch at block {size}")
        if 64 in results and _error(result, results[64])["max_rel_to_peak"] > 1e-10:
            raise AssertionError(f"block-size mismatch at block {size}")
    sparse_operators = tuple(sparse.csr_matrix(operator) for operator in operators)
    sparse_result = calculate_qhgk_isotropic_blocked(
        **{**common, "harmonic_operator": sparse_operators}, block_size=64,
        compute_dtype=np.float64)[0]
    print("CSR versus dense:", _error(sparse_result, results[64]))
    if _error(sparse_result, results[64])["max_rel_to_peak"] > 1e-10:
        raise AssertionError("CSR versus dense mismatch")
    mixed_result = calculate_qhgk_isotropic_blocked(
        **{**common, "harmonic_operator": sparse_operators}, block_size=64,
        compute_dtype=np.float32)[0]
    print("mixed versus float64:", _error(mixed_result, results[64]))
    total_error = abs(np.sum(mixed_result) - np.sum(results[64])) / max(abs(np.sum(results[64])), 1e-30)
    print("mixed total relative error:", total_error)
    if total_error > 1e-6:
        raise AssertionError("mixed-precision total mismatch")
    if args.sparse_harmonic is not None:
        atom_blocks = SparseAmorphousHarmonic.load(args.sparse_harmonic)
        print("atom-block validation:", atom_blocks.validation_report())
        for axis, operator, flux in zip("xyz", atom_blocks.operators, saved_flux):
            projected = e0.T @ (operator @ e1)
            error = _error(projected, flux[:e0.shape[1], 64:64+e1.shape[1]])
            print(f"atom-block flux {axis}: {error}")
            if error["max_rel_to_peak"] > args.saved_rtol:
                raise AssertionError(f"atom-block flux {axis} mismatch")
        atom_result = calculate_qhgk_isotropic_blocked(
            **{**common, "harmonic_operator": atom_blocks.operators},
            block_size=64, compute_dtype=np.float64)[0]
        if _error(atom_result, results[64])["max_rel_to_peak"] > 1e-10:
            raise AssertionError("atom-block conductivity mismatch")
    print("peak process RSS (platform units):", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    main()
