"""Rebuild the checked-in Gonze/Phonopy inputs from pinned source archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    """Return the SHA-256 checksum used by the fixture manifest."""

    return hashlib.sha256(data).hexdigest()


def main(archive_directory: str) -> None:
    """Extract and verify the checked-in YAML fixtures from source archives.

    ``archive_directory`` must contain the archive filenames recorded in
    ``manifest.json``. Both each downloaded archive and its extracted YAML
    payload are verified before the fixture is replaced.
    """

    archives = Path(archive_directory)
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        source = archives / case["archive"]
        payload = source.read_bytes()
        if digest(payload) != case["archive_sha256"]:
            raise RuntimeError(f"archive checksum mismatch: {source}")
        with zipfile.ZipFile(source) as bundle:
            members = [
                name
                for name in bundle.namelist()
                if name.endswith("phonopy_params.yaml")
            ]
            if len(members) != 1:
                raise RuntimeError(f"expected one phonopy_params.yaml in {source}")
            yaml = bundle.read(members[0])
        if digest(yaml) != case["yaml_sha256"]:
            raise RuntimeError(f"YAML fixture checksum mismatch for {case['id']}")
        target = ROOT / case["id"] / "phonopy_params.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(yaml)
        print(target)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARCHIVE_DIRECTORY")
    main(sys.argv[1])
