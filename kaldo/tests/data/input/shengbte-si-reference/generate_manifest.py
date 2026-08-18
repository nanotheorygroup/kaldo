"""Regenerate hashes and fixed provenance for the ShengBTE Si references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_ROOT = ROOT.parents[2]

SOURCES = {
    "vasp": [
        ROOT / "vasp" / "CONTROL",
        TEST_ROOT / "si-crystal" / "vasp" / "FORCE_CONSTANTS",
        TEST_ROOT / "si-crystal" / "vasp" / "FORCE_CONSTANTS_3RD",
    ],
    "qe": [
        ROOT / "qe" / "CONTROL",
        TEST_ROOT / "si-crystal" / "qe" / "espresso.ifc2",
        TEST_ROOT / "si-crystal" / "qe" / "FORCE_CONSTANTS_3RD",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.relative_to(TEST_ROOT))


def main() -> None:
    manifest = {
        "schema": 1,
        "shengbte": {
            "repository": "https://bitbucket.org/sousaw/shengbte.git",
            "commit": "b0d209068239c37fc86d2021efda131ad854f1c1",
            "describe": "v1.5.1-3-gb0d2090",
        },
        "settings": {
            "q_grid": [3, 3, 3],
            "temperature_k": 300.0,
            "scalebroad": 1.0,
            "convergence": False,
            "nonanalytic": False,
            "isotopes": False,
            "mpi_ranks": 2,
            "openmp_threads_per_rank": 1,
        },
        "slurm_job": 5416312,
        "cases": {},
    }
    for case, source_paths in SOURCES.items():
        result_paths = sorted((ROOT / case / "reference").rglob("BTE.*"))
        manifest["cases"][case] = {
            "sources": {
                _display_path(path): _sha256(path) for path in source_paths
            },
            "results": {
                _display_path(path): _sha256(path) for path in result_paths
            },
        }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
