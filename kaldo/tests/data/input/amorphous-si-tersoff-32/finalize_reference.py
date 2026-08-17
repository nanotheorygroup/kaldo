#!/usr/bin/env python3
"""Measure and record the amorphous-Si pair-image regression references.

Run this script from the repository root after ``generate_fixture.py`` has
created ``second.npy`` and ``third.npz``.  The physical oracle is invariance
under a rigid origin shift, not the values produced by kALDo itself.  The
stored numbers pin the magnitude of the exposed legacy error and make later
numerical drift visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Use the checkout containing this script, not a different editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from sparse import COO

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.thirdorder import _rank8_ifc3
from kaldo.phonons import Phonons

import kaldo

if Path(kaldo.__file__).resolve().parents[1] != REPOSITORY_ROOT:
    raise RuntimeError("reference finalizer imported kALDo from another checkout")

# These shifts are fixed before inspecting the transport result.  The first
# one that wraps more than one subset of atoms is used, so the regression does
# not optimize the apparent size of the bug.
ORIGIN_SHIFT_CANDIDATES = (
    (0.53, 0.37, 0.29),
    (0.71, 0.43, 0.19),
    (0.47, 0.61, 0.73),
)
IFC3_PROBE_Q_POINTS = ((0.17, 0.11, 0.07), (-0.09, 0.13, 0.05))


def sha256(path):
    """Return the hexadecimal SHA-256 digest of one fixture artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_forceconstants(folder):
    """Load the compact periodic IFC2/IFC3 fixture through the public API."""
    return ForceConstants.from_folder(
        folder=str(folder),
        format="numpy",
        supercell=(1, 1, 1),
        third_supercell=(1, 1, 1),
        is_acoustic_sum=True,
    )


def choose_origin_shift(atoms):
    """Choose the first predeclared shift that crosses multiple cell faces."""
    fractional = atoms.get_scaled_positions(wrap=False)
    for candidate in ORIGIN_SHIFT_CANDIDATES:
        crossed = np.floor(fractional + np.asarray(candidate)).astype(np.int64)
        if np.unique(crossed, axis=0).shape[0] > 1:
            return np.asarray(candidate, dtype=float)
    raise RuntimeError("predeclared origin shifts do not wrap distinct atom subsets")


def move_compact_origin(forceconstants, shift):
    """Wrap the basis after a rigid shift without modifying physical IFCs."""
    for observable in (forceconstants.second, forceconstants.third):
        scaled = observable.atoms.get_scaled_positions(wrap=False) + shift
        observable.atoms.set_scaled_positions(np.mod(scaled, 1.0))
        observable.replicated_positions = observable.atoms.positions[np.newaxis, :, :]
        observable._replicated_atoms = None


def boundary_weight(forceconstants):
    """Measure nonzero IFC weight whose compact pair crosses a cell face."""
    atoms = forceconstants.atoms
    fractional = atoms.get_scaled_positions(wrap=False)
    pair_shift = -np.rint(
        fractional[np.newaxis, :, :] - fractional[:, np.newaxis, :]
    ).astype(np.int64)
    boundary_pair = np.any(pair_shift != 0, axis=-1)

    second = np.asarray(forceconstants.second.dynmat)[0, :, :, 0, :, :]
    second_block_weight = np.sum(np.abs(second), axis=(1, 3))
    second_total = float(np.sum(second_block_weight))

    third = forceconstants.third
    rank8 = _rank8_ifc3(third.value, len(atoms), third.n_translations)
    sparse = rank8 if isinstance(rank8, COO) else COO.from_numpy(np.asarray(rank8))
    coords = np.asarray(sparse.coords)
    boundary_entry = (
        boundary_pair[coords[0], coords[3]] | boundary_pair[coords[0], coords[6]]
    )
    third_abs = np.abs(np.asarray(sparse.data))
    return {
        "ifc2_nonzero_boundary_pairs": int(
            np.count_nonzero((second_block_weight > 0.0) & boundary_pair)
        ),
        "ifc2_boundary_l1_fraction": float(
            np.sum(second_block_weight[boundary_pair]) / second_total
        ),
        "ifc3_nonzero_boundary_entries": int(np.count_nonzero(boundary_entry)),
        "ifc3_boundary_l1_fraction": float(
            np.sum(third_abs[boundary_entry]) / np.sum(third_abs)
        ),
    }


def harmonic(forceconstants, mode):
    """Construct the Gamma harmonic object used by amorphous transport."""
    return HarmonicWithQ(
        q_point=np.zeros(3),
        second=forceconstants.second,
        storage="memory",
        is_amorphous=True,
        ifc_interpolation=mode,
    )


def flux(harmonic_object):
    """Return the three Cartesian heat-flux matrices."""
    return np.stack(
        (harmonic_object._sij_x, harmonic_object._sij_y, harmonic_object._sij_z)
    )


def phonons(forceconstants, mode):
    """Construct the exact Phonons configuration used by the regression."""
    return Phonons(
        forceconstants=forceconstants,
        kpts=(1, 1, 1),
        temperature=300.0,
        is_classic=False,
        third_bandwidth=0.05 / 4.135,
        broadening_shape="triangle",
        storage="memory",
        ifc_interpolation=mode,
    )


def qhgk_tensor(forceconstants, mode):
    """Return the total QHGK conductivity tensor in W/(m K)."""
    conductivity = Conductivity(
        phonons=phonons(forceconstants, mode),
        method="qhgk",
        storage="memory",
    )
    tensor = np.asarray(conductivity.conductivity).sum(axis=0)
    if not np.isfinite(tensor).all():
        raise RuntimeError("QHGK reference contains a non-finite value")
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    metadata_path = args.folder / "expected.json"
    metadata = json.loads((args.folder / "generation.json").read_text())
    reference_fc = load_forceconstants(args.folder)
    shift = choose_origin_shift(reference_fc.atoms)
    shifted_fc = load_forceconstants(args.folder)
    move_compact_origin(shifted_fc, shift)

    corrected_reference_flux = flux(harmonic(reference_fc, "auto"))
    corrected_shifted_flux = flux(harmonic(shifted_fc, "auto"))
    np.testing.assert_allclose(
        corrected_shifted_flux,
        corrected_reference_flux,
        rtol=2e-11,
        atol=2e-11,
    )
    reference_flux = flux(harmonic(reference_fc, "periodic"))
    shifted_flux = flux(harmonic(shifted_fc, "periodic"))
    flux_change = np.linalg.norm(shifted_flux - reference_flux) / np.linalg.norm(
        reference_flux
    )
    if flux_change <= 0.05:
        raise RuntimeError(
            "the predeclared origin shift does not expose a 5% IFC2 first-moment error"
        )

    corrected_tensor = qhgk_tensor(reference_fc, "auto")
    corrected_shifted_tensor = qhgk_tensor(shifted_fc, "auto")
    np.testing.assert_allclose(
        corrected_shifted_tensor, corrected_tensor, rtol=2e-10, atol=2e-12
    )
    legacy_reference_tensor = qhgk_tensor(reference_fc, "periodic")
    legacy_shifted_tensor = qhgk_tensor(shifted_fc, "periodic")
    legacy_qhgk_change = abs(
        np.trace(legacy_shifted_tensor) - np.trace(legacy_reference_tensor)
    ) / abs(np.trace(legacy_reference_tensor))
    if legacy_qhgk_change <= 0.01:
        raise RuntimeError(
            "the predeclared origin shift does not expose a 1% QHGK error"
        )

    measured_boundary_weight = boundary_weight(reference_fc)
    if measured_boundary_weight["ifc2_nonzero_boundary_pairs"] == 0:
        raise RuntimeError("fixture has no nonzero boundary-crossing IFC2 pair")
    if measured_boundary_weight["ifc3_nonzero_boundary_entries"] == 0:
        raise RuntimeError("fixture has no nonzero boundary-crossing IFC3 entry")

    metadata.update(
        {
            "reference_finalizer_version": 1,
            "origin_shift_fractional": shift.tolist(),
            "boundary_weight": measured_boundary_weight,
            "ifc3_probe_q_points": [list(q) for q in IFC3_PROBE_Q_POINTS],
            "references": {
                "legacy_origin_flux_relative_change": float(flux_change),
                "qhgk_tensor_W_mK": corrected_tensor.tolist(),
                "legacy_origin_qhgk_trace_relative_change": float(legacy_qhgk_change),
            },
        }
    )
    retained = (
        "amorphous_si_32.extxyz",
        "replicated_atoms.xyz",
        "replicated_atoms_third.xyz",
        "second.npy",
        "third.npz",
    )
    metadata["sha256"] = {
        name: sha256(args.folder / name)
        for name in retained
        if (args.folder / name).is_file()
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata["boundary_weight"], indent=2))
    print(f"legacy flux change: {flux_change:.8%}")
    print(f"legacy QHGK trace change: {legacy_qhgk_change:.8%}")
    print("corrected QHGK tensor [W/(m K)]:")
    print(corrected_tensor)


if __name__ == "__main__":
    main()
