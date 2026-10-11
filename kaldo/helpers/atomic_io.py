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


def atomic_write_json(path, value):
    """Write sorted, indented JSON to ``path.tmp``, then rename it over ``path``."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_savez_compressed(path, **arrays):
    """Write a compressed ``.npz`` archive to ``path.tmp``, then rename it over ``path``.

    The archive is written through a file handle, so ``path`` is used verbatim
    and NumPy never appends a second ``.npz`` suffix.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
