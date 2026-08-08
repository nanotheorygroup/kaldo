"""Ensure the offline Phonopy inputs for Gonze validation remain pinned."""

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "input" / "gonze-phonopy"


def test_gonze_phonopy_inputs_preserve_selected_26_campaign() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 26
    assert manifest["bravais_coverage"].startswith("13 of 14")
    for case in manifest["cases"]:
        source = DATA / case["id"] / "phonopy_params.yaml"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["yaml_sha256"]


def test_gonze_inputs_record_official_downloads() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert "8cabbe0aee3dde48a19f1235864b0d3ab5cb9330" in manifest["source_index"]
    assert manifest["download_base"] == "https://mdr.nims.go.jp/download_all/"
    for case in manifest["cases"]:
        assert case["source_dataset"].startswith(
            "https://mdr.nims.go.jp/concern/datasets/"
        )
        assert case["source_archive"].startswith(manifest["download_base"])
        assert case["source_archive"].endswith(".zip")
    assert not (DATA / "generate_inputs.py").exists()
