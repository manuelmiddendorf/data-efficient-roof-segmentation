"""Small, strict training-run artifact contract and compatibility checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .integrity import sha256_file


REQUIRED_COMPLETED_FILES = {
    "config.json", "metadata.json", "summary.json", "history.csv", "metrics.csv", "checkpoint.pt"
}


def json_digest(value: object) -> str:
    """Hash canonical compact JSON for a stable identity."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def selected_input_digest(manifest: dict, sample_ids: Iterable[str]) -> str:
    """Hash selected IDs and their provider image/mask checksums."""
    records = {sample["id"]: sample for sample in manifest["samples"]}
    identity = [{"id": sample_id, "image": records[sample_id]["image_sha256"],
                 "mask": records[sample_id]["mask_sha256"]} for sample_id in sample_ids]
    return json_digest(identity)


def write_json(path: Path, value: object) -> None:
    """Write JSON atomically within an already selected run directory."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a rectangular CSV atomically."""
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_run(directory: Path, config: dict, metadata: dict) -> bool:
    """Create an uncompleted run or validate and reuse a matching completed run.

    Returns ``True`` only for a complete compatible run. Any incomplete or
    conflicting directory raises instead of being overwritten or resumed.
    """
    if directory.exists() and any(directory.iterdir()):
        available = {path.name for path in directory.iterdir() if path.is_file()}
        if not {"config.json", "metadata.json", "summary.json"} <= available:
            raise ValueError(f"Incomplete run lacks standard JSON files: {directory}")
        saved_config = json.loads((directory / "config.json").read_text())
        saved_metadata = json.loads((directory / "metadata.json").read_text())
        saved_summary = json.loads((directory / "summary.json").read_text())
        differences = []
        if saved_config != config:
            differences.append("config.json")
        if saved_metadata.get("compatibility_identity") != metadata.get("compatibility_identity"):
            differences.append("metadata.compatibility_identity")
        if differences:
            raise ValueError(f"Run conflicts with current inputs: {', '.join(differences)}")
        if saved_summary.get("status") != "completed" or not REQUIRED_COMPLETED_FILES <= available:
            raise ValueError(f"Matching run is incomplete and will not be resumed automatically: {directory}")
        return True
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config)
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "summary.json", {"status": "running"})
    return False


def verify_completed_run(directory: Path) -> dict:
    """Require the complete standard artifact set and checkpoint identity."""
    available = {path.name for path in directory.iterdir() if path.is_file()}
    if not REQUIRED_COMPLETED_FILES <= available:
        raise ValueError(f"Run artifact set is incomplete: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "completed":
        raise ValueError(f"Run summary is not complete: {directory}")
    if sha256_file(directory / "checkpoint.pt") != summary["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash differs from summary: {directory}")
    return summary
