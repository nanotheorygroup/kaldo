"""Content hashing and crash-safe file replacement for checkpoint files."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path, *, chunk_bytes=4 * 1024 * 1024):
    """Hex SHA-256 digest of a file, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays):
    """Hex SHA-256 of each array's dtype, shape, and C-ordered bytes, in order."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_json(value):
    """Hex SHA-256 of ``value`` serialized as JSON with sorted keys."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_write(path, write):
    """Call ``write(stream)`` on ``path.tmp``, then rename it over ``path``.

    The binary stream is passed to ``write``, so writers such as ``np.save``
    use ``path`` verbatim and never append a second suffix.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        write(stream)
    os.replace(temporary, path)


def atomic_write_json(path, value):
    """Write sorted, indented JSON to ``path.tmp``, then rename it over ``path``."""
    text = json.dumps(value, indent=2, sort_keys=True).encode()
    atomic_write(path, lambda stream: stream.write(text))


def atomic_save_npy(path, array):
    """Write one ``.npy`` array atomically, without pickled objects."""
    atomic_write(path, lambda stream: np.save(stream, array, allow_pickle=False))


def atomic_savez(path, **arrays):
    """Write an uncompressed ``.npz`` archive atomically."""
    atomic_write(path, lambda stream: np.savez(stream, **arrays))


def atomic_savez_compressed(path, **arrays):
    """Write a compressed ``.npz`` archive atomically."""
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
