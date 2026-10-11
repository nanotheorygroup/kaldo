"""Probe finite-difference IFC columns to choose sparse coloring radii."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from ase.neighborlist import neighbor_list

from kaldo.helpers.atomic_io import atomic_savez_compressed, atomic_write_json
from kaldo.helpers.logger import get_logger
from ._common import atoms_hash, calculator_instance, describe_calculator
from .structure import _DEFAULT_PAIR_LIMIT, build_neighbor_structure, color_structure, perpendicular_cell_widths


_LOG = get_logger()


@dataclass(frozen=True)
class RangeRecommendation:
    """Empirical radius and the cost of using it for color and storage."""

    tolerance: float
    radius: float | None
    n_groups: int | None
    force_calls: int | None
    nnz_blocks: int | None
    bsr_gb: float | None
    expected_contamination: float | None


@dataclass(frozen=True)
class RangeEstimate:
    """Maximum sampled block norm in each distance bin, normalized by onsite."""

    probe_atoms: tuple[int, ...]
    bin_width: float
    bin_edges: tuple[float, ...]
    normalized_max: tuple[float, ...]
    median_self_norm: float
    recommendations: tuple[RangeRecommendation, ...]
    structural_upper_bound: float | None
    cutoff_hint: float | None
    h_noise_ratio: float | None
    warnings: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self.to_dict())


def _probe_fingerprint(atoms, delta_shift, calculator_description):
    payload = [atoms_hash(atoms), repr(float(delta_shift)), calculator_description]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _cutoff_hint(calculator, explicit):
    if explicit is not None:
        candidate = explicit
    else:
        candidate = None
        parameters = getattr(calculator, "parameters", {})
        for source in (parameters, calculator):
            for key in ("rc", "cutoff", "r_cut"):
                value = source.get(key) if hasattr(source, "get") else getattr(source, key, None)
                if isinstance(value, (int, float, np.number)) and np.isfinite(value) and value > 0:
                    candidate = value
                    break
            if candidate is not None:
                break
    if candidate is None:
        return None
    candidate = float(candidate)
    if not np.isfinite(candidate) or candidate <= 0:
        raise ValueError("cutoff_hint must be positive")
    return candidate


def _select_probe_atoms(atoms, n_probe, seed, reference_radius):
    n_atoms = len(atoms)
    if n_probe < 1 or not n_atoms or reference_radius <= 0:
        raise ValueError("n_probe, atom count, and reference_radius must be positive")
    if n_atoms == 1:
        return (0,)
    if n_probe < 2:
        raise ValueError("n_probe must be at least 2 when the structure has multiple atoms")
    row, _ = neighbor_list("ij", atoms, reference_radius, self_interaction=False)
    coordination = np.bincount(row, minlength=n_atoms)
    low = int(np.argmin(coordination))
    high = n_atoms - 1 - int(np.argmax(coordination[::-1]))
    chosen = [low]
    if high != low:
        chosen.append(high)
    rng = np.random.default_rng(seed)
    for atom in rng.permutation(n_atoms):
        if len(chosen) >= min(n_probe, n_atoms):
            break
        if int(atom) not in chosen:
            chosen.append(int(atom))
    return tuple(chosen)


def _probe_columns(atoms, calculator, probe, delta_shift):
    work = atoms.copy()
    work.calc = calculator
    initial = work.positions.copy()
    columns = np.empty((len(work), 3, 3), dtype=np.float64)
    try:
        for axis in range(3):
            work.positions[:] = initial
            work.positions[probe, axis] += delta_shift
            plus = np.asarray(work.get_forces(), dtype=np.float64)
            work.positions[:] = initial
            work.positions[probe, axis] -= delta_shift
            minus = np.asarray(work.get_forces(), dtype=np.float64)
            if plus.shape != (len(work), 3) or minus.shape != plus.shape:
                raise ValueError("calculator returned forces with the wrong shape")
            columns[:, :, axis] = -(plus - minus) / (2 * delta_shift)
    finally:
        work.positions[:] = initial
    if not np.isfinite(columns).all():
        raise ValueError("calculator returned nonfinite finite-difference columns")
    return columns


def _load_or_probe(atoms, calculator, probe, delta_shift, checkpoint_dir, fingerprint):
    if checkpoint_dir is None:
        return _probe_columns(atoms, calculator, probe, delta_shift)
    path = checkpoint_dir / f"probe_{probe:09d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["fingerprint"]) != fingerprint or int(saved["probe"]) != probe or
                    saved["blocks"].shape != (len(atoms), 3, 3)):
                raise ValueError(f"probe checkpoint is incompatible: {path}")
            blocks = np.asarray(saved["blocks"], dtype=np.float64)
            if not np.isfinite(blocks).all():
                raise ValueError(f"probe checkpoint has nonfinite blocks: {path}")
            return blocks
    blocks = _probe_columns(atoms, calculator, probe, delta_shift)
    atomic_savez_compressed(path, fingerprint=np.array(fingerprint), probe=np.array(probe), blocks=blocks)
    return blocks


def estimate_interaction_range(atoms, calculator, *, delta_shift=1e-3, n_probe=12,
                               seed=0, checkpoint_dir=None, calculator_description=None,
                               reference_radius=3.0, bin_width=0.5,
                               tolerances=(1e-3, 1e-4, 1e-5), coloring_seeds=(0, 1, 2),
                               cutoff_hint=None, cutoff_multiplier=2.0,
                               pair_memory_limit=_DEFAULT_PAIR_LIMIT, check_h_convergence=False,
                               json_path=None):
    """Estimate the finite-range IFC tail from exact single-atom FD columns.

    Recommended radii are empirical: they are the first bin edges beyond all
    sampled blocks at or above each tolerance. The optional structural bound
    uses ``cutoff_multiplier * cutoff_hint``; the default factor two is
    conservative for descriptor-based many-body potentials. For pairwise
    potentials, set ``cutoff_multiplier=1``. Checkpoint files contain derived
    probe IFC columns, never raw force arrays.
    """
    if calculator is None:
        raise ValueError("calculator is required")
    if not np.isfinite(delta_shift) or delta_shift <= 0:
        raise ValueError("delta_shift must be positive")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if not np.isfinite(cutoff_multiplier) or cutoff_multiplier <= 0:
        raise ValueError("cutoff_multiplier must be positive")
    tolerances = tuple(float(tolerance) for tolerance in tolerances)
    if not tolerances or any(not 0 < tolerance < 1 for tolerance in tolerances):
        raise ValueError("tolerances must lie strictly between zero and one")
    description = calculator_description or describe_calculator(calculator)
    fingerprint = _probe_fingerprint(atoms, delta_shift, description)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probes = _select_probe_atoms(atoms, n_probe, seed, reference_radius)
    engine = calculator_instance(calculator)
    norms = []
    distances = []
    self_norms = []
    first_blocks = None
    for probe in probes:
        blocks = _load_or_probe(atoms, engine, probe, delta_shift, checkpoint_dir, fingerprint)
        if first_blocks is None:
            first_blocks = blocks
        self_norms.append(float(np.linalg.norm(blocks[probe])))
        norms.append(np.linalg.norm(blocks, axis=(1, 2)))
        distances.append(np.asarray(atoms.get_distances(probe, np.arange(len(atoms)), mic=True), dtype=np.float64))
        _LOG.info("Colored IFC range probe %d/%d: atom %d", len(self_norms), len(probes), probe)
    median_self = float(np.median(self_norms))
    if not np.isfinite(median_self) or median_self <= 0:
        raise ValueError("probe onsite blocks have zero or nonfinite median norm")
    distances = np.concatenate(distances)
    normalized = np.concatenate(norms) / median_self
    bins = np.floor(distances / bin_width).astype(np.int64)
    if bins.max() > 1_000_000:
        raise ValueError("bin_width creates too many distance bins")
    profile = np.zeros(int(bins.max()) + 1, dtype=np.float64)
    np.maximum.at(profile, bins, normalized)
    edges = np.arange(len(profile) + 1, dtype=np.float64) * bin_width
    notes = []
    widths = perpendicular_cell_widths(atoms)
    minimum_width = float(widths.min()) if len(widths) else None
    raw_hint = _cutoff_hint(engine, cutoff_hint)
    upper_bound = None if raw_hint is None else raw_hint * cutoff_multiplier
    h_noise_ratio = None
    if check_h_convergence:
        half = _probe_columns(atoms, engine, probes[0], delta_shift / 2)
        double = _probe_columns(atoms, engine, probes[0], delta_shift * 2)
        small_error = float(np.linalg.norm(half - first_blocks))
        large_error = float(np.linalg.norm(double - first_blocks))
        h_noise_ratio = small_error / max(large_error, np.finfo(float).tiny)
        if h_noise_ratio > 1.5:
            notes.append("h/2 changes the probe more than 2h; force noise may dominate")
    radii = []
    for tolerance in tolerances:
        significant = np.flatnonzero(profile >= tolerance)
        if len(profile) > 1 and len(significant) and significant[-1] == len(profile) - 1:
            notes.append(f"no decay below {tolerance:g} within sampled radius; consider a separate long-range term")
            radii.append(None)
        else:
            radii.append(float(edges[significant[-1] + 1]) if len(significant) else float(bin_width))
    # One neighbor search at the widest radius; narrower graphs are distance filters of it.
    unique_radii = sorted({radius for radius in radii if radius is not None}, reverse=True)
    costs = {}
    if unique_radii:
        widest = build_neighbor_structure(atoms, unique_radii[0], pair_memory_limit=pair_memory_limit)
        for radius in unique_radii:
            structure = widest.restrict(radius)
            costs[radius] = (color_structure(structure, seeds=coloring_seeds).n_groups, structure.nnz_blocks)
    recommendations = []
    for tolerance, radius in zip(tolerances, radii):
        if radius is None:
            recommendations.append(RangeRecommendation(tolerance, None, None, None, None, None, None))
            continue
        n_groups, nnz_blocks = costs[radius]
        tail = float(np.max(normalized[distances >= radius], initial=0.0))
        bsr_bytes = nnz_blocks * (9 * 8 + 4) + (len(atoms) + 1) * 4
        recommendations.append(RangeRecommendation(tolerance, radius, n_groups, 6 * n_groups, nnz_blocks,
                                                    bsr_bytes / 1e9, tail))
        if minimum_width is not None and radius > minimum_width / 2:
            notes.append(f"radius {radius:g} Å exceeds half the smallest perpendicular cell width")
    for note in dict.fromkeys(notes):
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    result = RangeEstimate(probes, float(bin_width), tuple(map(float, edges)), tuple(map(float, profile)),
                           median_self, tuple(recommendations), upper_bound, raw_hint, h_noise_ratio,
                           tuple(dict.fromkeys(notes)))
    if json_path is not None:
        result.save_json(json_path)
    return result
