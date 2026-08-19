#!/usr/bin/env python3
"""Pack the direct Fortran rgd_blk output as the checked-in NumPy fixture."""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np

SHA = "c2da58dfa6849c4edb6f96ad8ad2be58834ae23142bdac2ee33eeebf1eb21e88"
NAMES = (
    "finite_3d",
    "gamma_direct_3d",
    "gamma_directional_3d",
    "finite_negative_3d",
    "finite_2d",
)


def parse(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    name: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith("["):
            name = line[1:-1]
            values[name] = np.zeros((3, 3, 3, 3), complex)
            continue
        assert name is not None
        i, j, na, nb, re, im = line.split()
        values[name][int(na) - 1, int(i) - 1, int(nb) - 1, int(j) - 1] = float(
            re
        ) + 1j * float(im)
    assert tuple(values) == NAMES
    return values


def main() -> None:
    source, output = map(Path, sys.argv[1:3])
    arrays = parse(source)
    arrays.update(
        qe_tag=np.array("qe-7.6"),
        rigid_f90_sha256=np.array(SHA),
        q_reduced_3d=np.array([0.173, -0.119, 0.087]),
        gamma_direction_reduced=np.array([1.0, 2.0, -1.0]),
        q_reduced_2d=np.array([0.137, -0.083, 0.0]),
    )
    np.savez(output, **arrays)


if __name__ == "__main__":
    main()
