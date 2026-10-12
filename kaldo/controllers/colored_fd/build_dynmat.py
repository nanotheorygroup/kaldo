"""Build a restartable, sparse Gamma dynamical matrix from colored IFC2.

Example (submit each ``run`` index as a separate scheduler job)::

    python -m kaldo.controllers.colored_fd.build_dynmat plan \
        --workdir /scratch/colored-si --structure amorphous.extxyz \
        --calculator my_potential:make_calculator --delta-shift 0.01 --shards 16
    python -m kaldo.controllers.colored_fd.build_dynmat run \
        --workdir /scratch/colored-si --index 0
    python -m kaldo.controllers.colored_fd.build_dynmat finalize \
        --workdir /scratch/colored-si

The factory must be importable and take no arguments. It must return an ASE
calculator. ``finalize`` writes ``merged/`` (the processed IFC2, loadable with
``ForceConstants.from_folder(folder, format='colored', only_second=True)``),
``dynmat_bsr.npz`` (a SciPy 3x3 BSR matrix in kALDo dynamical units), and
``harmonic/`` (the atom-block input of the distributed QHGK reducer).
``second.npy`` is never written.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import ase.io
import numpy as np
from scipy.sparse import save_npz

from kaldo.helpers.atomic_io import atomic_save_npy, atomic_write, atomic_write_json, sha256_file
from kaldo.helpers.logger import get_logger
from ._common import atoms_extra_arrays_hash, atoms_hash, atoms_info_hash, calculator_instance
from .assembly import merge_colored_second_order
from . import estimate as estimate_module
from .estimate import estimate_interaction_range
from .plan import load_colored_second_order_plan, plan_colored_second_order
from .run import run_colored_second_order
from .structure import perpendicular_cell_widths


_LOG = get_logger()
_FORMAT = "kaldo-colored-dynmat-run-v1"
_DYNMAT_FORMAT = "kaldo-colored-dynmat-bsr-v1"


def _factory(import_path):
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute or any(not part.isidentifier()
                                                               for part in attribute.split(".")):
        raise ValueError("calculator must be an importable module:factory path")
    factory = importlib.import_module(module_name)
    for name in attribute.split("."):
        factory = getattr(factory, name)
    if not callable(factory):
        raise ValueError("calculator factory must be callable")
    return factory


def _atoms(path, expected_atoms=None):
    atoms = ase.io.read(path, index=0)
    if expected_atoms is not None and len(atoms) != expected_atoms:
        raise ValueError(f"expected {expected_atoms} atoms, found {len(atoms)}")
    if atoms.constraints:
        raise ValueError("colored force derivatives require unconstrained atoms")
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if masses.shape != (len(atoms),) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("structure must provide positive finite atomic masses")
    return atoms, masses


def _configuration(workdir):
    config = json.loads((workdir / "run.json").read_text())
    if config.get("format") != _FORMAT:
        raise ValueError("unsupported colored dynamical-matrix run configuration")
    plan = load_colored_second_order_plan(workdir / "plan")
    if (config["plan_sha256"] != plan.plan_hash or
            config["calculator_description"] != plan.metadata["calculator_description"] or
            config["calculator_description"] != _description(
                config["calculator"], config["model_description"]) or
            config["atoms_sha256"] != plan.metadata["atoms_sha256"] or
            config["extra_arrays_sha256"] != plan.metadata["extra_arrays_sha256"] or
            config["atoms_info_sha256"] != plan.metadata["atoms_info_sha256"] or
            config["color_radius_angstrom"] != plan.metadata["color_radius_angstrom"] or
            config["store_radius_angstrom"] != plan.metadata["store_radius_angstrom"] or
            config["delta_shift_angstrom"] != plan.delta_shift):
        raise ValueError("run configuration does not match colored IFC plan")
    return config, plan


def _description(factory_path, model_description):
    return factory_path if model_description is None else f"{factory_path} | {model_description}"


def plan(args):
    workdir = args.workdir.resolve()
    structure_path = args.structure.resolve()
    atoms, masses = _atoms(structure_path, args.expected_atoms)
    factory = _factory(args.calculator)
    if args.calculator_description is not None and not args.calculator_description.strip():
        raise ValueError("calculator description must be nonempty")
    description = _description(args.calculator, args.calculator_description)
    fingerprint = {
        "format": _FORMAT,
        "structure": str(structure_path),
        "atoms_sha256": atoms_hash(atoms),
        "extra_arrays_sha256": atoms_extra_arrays_hash(atoms)[0],
        "atoms_info_sha256": atoms_info_hash(atoms),
        "calculator": args.calculator,
        "model_description": args.calculator_description,
        "calculator_description": description,
        "n_atoms": len(atoms),
        "n_shards": args.shards,
        "delta_shift_angstrom": args.delta_shift,
        "probe_tolerance": args.tolerance,
        "n_probe": args.n_probe,
        "seed": args.seed,
        "bin_width_angstrom": args.bin_width,
        "cutoff_hint_angstrom": args.cutoff_hint,
        "cutoff_multiplier": args.cutoff_multiplier,
    }
    config_path = workdir / "run.json"
    if config_path.exists():
        existing, _ = _configuration(workdir)
        if any(existing.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("existing run configuration differs; use a new workdir")
        if sha256_file(workdir / "masses.npy") != existing["masses_sha256"]:
            raise ValueError("saved masses have changed")
        if not np.array_equal(masses, np.load(workdir / "masses.npy", allow_pickle=False)):
            raise ValueError("current structure masses differ from the planned masses")
        _LOG.info("Reusing colored IFC plan: %s", workdir / "plan")
        return
    if not np.isfinite(args.delta_shift) or args.delta_shift <= 0:
        raise ValueError("delta_shift must be positive and finite")
    if not 0 < args.tolerance < 1 or not np.isfinite(args.tolerance):
        raise ValueError("probe tolerance must be between zero and one")
    if args.shards < 1 or args.n_probe < 2 or not np.isfinite(args.cutoff_multiplier) or args.cutoff_multiplier < 2:
        raise ValueError("shards and n_probe must be positive; cutoff_multiplier must be at least 2")
    calculator = calculator_instance(factory)
    detected_cutoff = estimate_module.resolve_cutoff_hint(calculator, None)
    supplied_cutoff = (estimate_module.resolve_cutoff_hint(calculator, args.cutoff_hint)
                       if args.cutoff_hint is not None else None)
    cutoff = max((value for value in (detected_cutoff, supplied_cutoff) if value is not None), default=None)
    if cutoff is None:
        raise ValueError("a finite calculator cutoff is required before force probes; provide --cutoff-hint")
    workdir.mkdir(parents=True, exist_ok=True)
    # The lower-level probe checkpoint hashes geometry and description, but
    # calculators may also read ASE arrays or info. Bind them into this run's
    # probe identity so a changed structure cannot reuse stale force columns.
    probe_description = (f"{description}|arrays={fingerprint['extra_arrays_sha256']}"
                         f"|info={fingerprint['atoms_info_sha256']}")
    estimate = estimate_interaction_range(
        atoms, calculator, delta_shift=args.delta_shift, n_probe=args.n_probe,
        seed=args.seed, checkpoint_dir=workdir / "probes",
        calculator_description=probe_description, bin_width=args.bin_width,
        tolerances=(args.tolerance,), cutoff_hint=cutoff,
        cutoff_multiplier=args.cutoff_multiplier, check_h_convergence=True,
        json_path=workdir / "probe_estimate.json",
    )
    if estimate.h_noise_ratio is not None and estimate.h_noise_ratio > 1.5:
        raise ValueError("finite-difference probe is noise dominated; choose a larger --delta-shift "
                         "and inspect probe_estimate.json")
    recommendation = estimate.recommendations[0]
    if (recommendation.radius is None or estimate.structural_upper_bound is None or
            not np.isfinite(estimate.structural_upper_bound)):
        raise ValueError("automatic radii require a resolved probe radius and a finite calculator cutoff; "
                         "check probe_estimate.json and provide --cutoff-hint if needed")
    store_radius = float(recommendation.radius)
    color_radius = float(np.nextafter(max(store_radius, estimate.structural_upper_bound), np.inf))
    widths = perpendicular_cell_widths(atoms)
    if len(widths) and color_radius > float(widths.min()) / 2:
        raise ValueError("automatic coloring radius exceeds half the smallest periodic cell width; "
                         "the physical cell must be large enough for an unambiguous finite-range run")
    colored_plan = plan_colored_second_order(
        atoms, color_radius, store_radius, delta_shift=args.delta_shift,
        plan_dir=workdir / "plan", calculator_description=description)
    masses_path = workdir / "masses.npy"
    atomic_save_npy(masses_path, masses)
    config = {**fingerprint, "plan_sha256": colored_plan.plan_hash,
              "masses_sha256": sha256_file(masses_path),
              "color_radius_angstrom": color_radius,
              "store_radius_angstrom": store_radius,
              "resolved_cutoff_angstrom": cutoff,
              "structural_upper_bound_angstrom": estimate.structural_upper_bound}
    atomic_write_json(config_path, config)
    _LOG.info("Plan: %d groups, %d tasks, %d stored blocks", colored_plan.n_groups,
              len(colored_plan.tasks), colored_plan.store.nnz)
    _LOG.info("R_color=%.6g A R_store=%.6g A shards=%d workdir=%s", color_radius, store_radius,
              args.shards, workdir)
    if args.calculator_description is None:
        _LOG.info("Calculator identity uses the factory path; pass --calculator-description "
                  "when external model files can change")


def run(args):
    workdir = args.workdir.resolve()
    config, colored_plan = _configuration(workdir)
    if not 0 <= args.index < config["n_shards"]:
        raise ValueError(f"shard index must be in [0, {config['n_shards']})")
    atoms, masses = _atoms(Path(config["structure"]), config["n_atoms"])
    if (atoms_hash(atoms) != config["atoms_sha256"] or
            atoms_extra_arrays_hash(atoms)[0] != config["extra_arrays_sha256"] or
            atoms_info_hash(atoms) != config["atoms_info_sha256"] or
            not np.array_equal(masses, np.load(workdir / "masses.npy", allow_pickle=False))):
        raise ValueError("current structure or masses differ from the planned force input")
    if sha256_file(workdir / "masses.npy") != config["masses_sha256"]:
        raise ValueError("saved masses have changed")
    plan_dir = colored_plan.directory
    del colored_plan
    directory = workdir / f"shard_{args.index:04d}_of_{config['n_shards']:04d}"
    _LOG.info("Running shard %d/%d: %s", args.index, config["n_shards"], directory)
    run_colored_second_order(
        plan_dir, _factory(config["calculator"]),
        shard=(args.index, config["n_shards"]), workdir=directory,
        calculator_description=config["calculator_description"], atoms=atoms,
        flush_tasks=args.flush_tasks)


def finalize(args):
    workdir = args.workdir.resolve()
    config, colored_plan = _configuration(workdir)
    response_threshold = (config["probe_tolerance"] if args.response_threshold is None
                          else args.response_threshold)
    if (not np.isfinite(response_threshold) or response_threshold <= 0 or
            not np.isfinite(args.residual_threshold) or args.residual_threshold <= 0):
        raise ValueError("diagnostic thresholds must be positive and finite")
    masses_path = workdir / "masses.npy"
    if sha256_file(masses_path) != config["masses_sha256"]:
        raise ValueError("saved masses have changed")
    masses = np.load(masses_path, allow_pickle=False)
    if masses.shape != (colored_plan.n_atoms,):
        raise ValueError("saved masses do not match the colored IFC plan")
    plan_dir = colored_plan.directory
    plan_hash = colored_plan.plan_hash
    n_atoms = colored_plan.n_atoms
    atoms = colored_plan.atoms.copy()
    atoms.set_masses(masses)
    del colored_plan
    shard_dirs = [workdir / f"shard_{index:04d}_of_{config['n_shards']:04d}"
                  for index in range(config["n_shards"])]
    missing = [str(directory) for directory in shard_dirs if not (directory / "manifest.json").is_file()]
    if missing:
        raise ValueError(f"missing shard manifests: {missing}")
    sparse_ifc = merge_colored_second_order(
        plan_dir, shard_dirs, output_dir=workdir / "merged",
        warning_threshold=response_threshold, apply_acoustic_sum=True,
        asr_mode="projection", symmetrize=True, atoms=atoms)
    diagnostics_path = workdir / "merged" / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    for key in ("relative_outside_response", "relative_truncation_response"):
        if not np.isfinite(diagnostics[key]) or diagnostics[key] > response_threshold:
            raise RuntimeError(f"{key}={diagnostics[key]:.3g} exceeds {response_threshold:g}; "
                               f"inspect {diagnostics_path} before using the matrix")
    residual = diagnostics["processing_report"]["after"]
    for key in ("relative_asymmetry", "relative_row_sum"):
        if not np.isfinite(residual[key]) or residual[key] > args.residual_threshold:
            raise RuntimeError(f"{key}={residual[key]:.3g} exceeds {args.residual_threshold:g}")
    output = workdir / "dynmat_bsr.npz"
    metadata_path = workdir / "dynmat_metadata.json"
    source = {"format": _DYNMAT_FORMAT, "shape": list(sparse_ifc.matrix.shape),
              "blocksize": [3, 3], "dtype": "float64", "units": "kaldo_10J_per_mol",
              "mass_weighted": True, "block_rows": "responding_atom",
              "block_columns": "displaced_atom",
              "plan_sha256": plan_hash,
              "ifc2_arrays_sha256": sha256_file(workdir / "merged" / "ifc2.npz"),
              "masses_sha256": config["masses_sha256"],
              "processing_sha256": sparse_ifc.metadata["processing_sha256"],
              "diagnostics_sha256": sha256_file(diagnostics_path)}
    if metadata_path.exists() and output.exists():
        existing = json.loads(metadata_path.read_text())
        if (any(existing.get(key) != value for key, value in source.items()) or
                existing.get("arrays_sha256") != sha256_file(output)):
            raise ValueError("existing dynamical matrix does not match verified inputs; "
                             "use a new workdir")
        _LOG.info("Reusing sparse dynamical matrix: %s", output)
    else:
        weighted = sparse_ifc.dynmat
        atomic_write(output, lambda stream: save_npz(stream, weighted, compressed=True))
        atomic_write_json(metadata_path, {**source, "n_atoms": n_atoms,
                                          "nnz_blocks": sparse_ifc.nnz_blocks,
                                          "storage_bytes": weighted.data.nbytes + weighted.indices.nbytes +
                                                           weighted.indptr.nbytes,
                                          "arrays_sha256": sha256_file(output)})
        _LOG.info("Saved %d x %d BSR dynamical matrix (%d blocks): %s", weighted.shape[0],
                  weighted.shape[1], sparse_ifc.nnz_blocks, output)
    _write_sparse_harmonic(workdir / "harmonic", sparse_ifc, source)


def _write_sparse_harmonic(directory, sparse_ifc, source):
    """Write the mass-weighted atom-block input for the distributed QHGK reducer, once.

    Pair displacements are minimum-image vectors, which the planning check
    (coloring radius at most half the narrowest cell width) makes unique.
    """
    provenance = {key: source[key] for key in ("plan_sha256", "ifc2_arrays_sha256", "masses_sha256")}
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()).get("colored_ifc_provenance") != provenance:
            raise ValueError("existing sparse harmonic does not match verified inputs; use a new workdir")
        _LOG.info("Reusing sparse harmonic: %s", directory)
        return
    harmonic = sparse_ifc.to_sparse_harmonic()
    harmonic.metadata["colored_ifc_provenance"] = provenance
    harmonic.save(directory)
    _LOG.info("Saved sparse harmonic for distributed QHGK: %s", directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="probe range and create the immutable coloring plan")
    planning.add_argument("--workdir", type=Path, required=True)
    planning.add_argument("--structure", type=Path, required=True, help="ASE-readable physical Gamma cell")
    planning.add_argument("--calculator", required=True, help="importable module:factory")
    planning.add_argument("--calculator-description", help="stable model/version identity for checkpoint safety")
    planning.add_argument("--delta-shift", type=float, required=True, help="central-difference shift in Angstrom")
    planning.add_argument("--shards", type=int, default=1)
    planning.add_argument("--expected-atoms", type=int,
                          help="fail unless the structure has exactly this many atoms")
    planning.add_argument("--n-probe", type=int, default=12)
    planning.add_argument("--tolerance", type=float, default=1e-5)
    planning.add_argument("--seed", type=int, default=0)
    planning.add_argument("--bin-width", type=float, default=0.5)
    planning.add_argument("--cutoff-hint", type=float, help="calculator descriptor cutoff in Angstrom")
    planning.add_argument("--cutoff-multiplier", type=float, default=2.0)
    worker = commands.add_parser("run", help="evaluate one checkpointed shard")
    worker.add_argument("--workdir", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--flush-tasks", type=int, default=1)
    merging = commands.add_parser("finalize", help="merge, validate, mass-weight, and save BSR")
    merging.add_argument("--workdir", type=Path, required=True)
    merging.add_argument("--response-threshold", type=float,
                         help="maximum relative omitted response; defaults to the probe tolerance")
    merging.add_argument("--residual-threshold", type=float, default=1e-8)
    args = parser.parse_args(argv)
    {"plan": plan, "run": run, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
