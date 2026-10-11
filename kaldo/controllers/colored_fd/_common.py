"""Format tags, input hashes, and calculator resolution shared by colored IFC2 stages."""

import hashlib
import json

import numpy as np


PLAN_FORMAT = "kaldo-colored-ifc2-plan-v1"
SHARD_FORMAT = "kaldo-colored-ifc2-shard-v1"
MERGE_FORMAT = "kaldo-colored-ifc2-merge-v1"


def is_calculator_factory(calculator):
    """True for a class or no-argument callable that builds an ASE calculator."""
    return isinstance(calculator, type) or (callable(calculator) and not hasattr(calculator, "get_forces"))


def calculator_instance(calculator):
    """Return an ASE calculator, calling ``calculator()`` when it is a factory."""
    if is_calculator_factory(calculator):
        calculator = calculator()
    if not hasattr(calculator, "get_forces"):
        raise ValueError("calculator must be an ASE calculator or a no-argument factory")
    return calculator


def describe_calculator(calculator):
    """Default description: the calculator's import path plus its ASE parameters."""
    if is_calculator_factory(calculator):
        return f"factory:{calculator.__module__}.{calculator.__qualname__}"
    cls = type(calculator)
    return f"{cls.__module__}.{cls.__qualname__}:{repr(getattr(calculator, 'parameters', None))}"


def _arrays_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atoms_hash(atoms):
    """SHA-256 of the exact atomic input used for force evaluations."""
    return _arrays_hash(np.asarray(atoms.numbers, dtype=np.int32),
                        np.asarray(atoms.positions, dtype=np.float64),
                        np.asarray(atoms.cell, dtype=np.float64),
                        np.asarray(atoms.pbc, dtype=np.bool_))


def atoms_extra_arrays_hash(atoms):
    """Hash ASE arrays beyond positions and numbers, without serializing them."""
    digest = hashlib.sha256()
    names = sorted(set(atoms.arrays) - {"numbers", "positions"})
    for name in names:
        array = np.ascontiguousarray(atoms.arrays[name])
        if array.dtype.hasobject:
            raise ValueError(f"ASE array {name!r} has unsupported object dtype")
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest(), names


def _json_info_value(value):
    """Canonicalize common ASE info scalars and containers for a stable hash."""
    if isinstance(value, np.ndarray):
        return _json_info_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_info_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("ASE info contains a nonfinite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_info_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_info_value(item) for key, item in sorted(value.items())}
    raise ValueError("ASE info must contain JSON-compatible values for colored IFC planning")


def atoms_info_hash(atoms):
    info = _json_info_value(atoms.info)
    return hashlib.sha256(json.dumps(info, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
