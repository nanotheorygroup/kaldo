"""Ensure the offline Phonopy inputs for Gonze validation remain pinned."""

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "input" / "gonze-phonopy"


def test_gonze_phonopy_inputs_cover_seven_lattice_systems() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    systems = [case["lattice_system"] for case in manifest["cases"]]
    assert systems == [
        "cubic",
        "hexagonal",
        "tetragonal",
        "orthorhombic",
        "trigonal",
        "monoclinic",
        "triclinic",
    ]
    for case in manifest["cases"]:
        source = DATA / case["id"] / "phonopy_params.yaml"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["yaml_sha256"]


def test_gonze_inputs_include_an_offline_rebuild_script() -> None:
    script = (DATA / "generate_inputs.py").read_text(encoding="utf-8")
    assert "archive_sha256" in script
    assert "zipfile.ZipFile" in script
    assert 'target = ROOT / case["id"] / "phonopy_params.yaml"' in script
