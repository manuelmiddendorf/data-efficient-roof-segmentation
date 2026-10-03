"""Hashing and immutable-reference checks for downloaded RID files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_provider_checksums(path: Path) -> dict[str, str]:
    """Parse the RID provider's GNU-style ``Checksums.sha256`` file.

    Parameters
    ----------
    path:
        Unmodified checksum file downloaded with RID release 1.0.

    Returns
    -------
    dict
        Project-release-relative POSIX paths mapped to lowercase SHA-256 values.
    """
    references: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split(" *", 1)
        except ValueError as error:
            raise ValueError(f"Malformed provider checksum line {line_number}.") from error
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"Invalid SHA-256 on provider checksum line {line_number}.")
        if relative in references:
            raise ValueError(f"Duplicate provider checksum path: {relative}")
        references[relative] = digest.lower()
    return references


def verify_files(root: Path, references: dict[str, str], relative_paths: Iterable[str]) -> dict[str, str]:
    """Verify selected files against an externally supplied checksum mapping.

    The function never creates or updates references. Missing, unreferenced or
    changed files stop the audit immediately.
    """
    verified: dict[str, str] = {}
    for relative in sorted(relative_paths):
        if relative not in references:
            raise ValueError(f"No provider checksum for {relative}.")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required RID file is missing: {relative}")
        actual = sha256_file(path)
        if actual != references[relative]:
            raise ValueError(f"Provider checksum mismatch for {relative}.")
        verified[relative] = actual
    return verified


def write_new_json(path: Path, value: Any) -> None:
    """Write a new JSON reference and refuse to replace different content."""
    serialized = json.dumps(value, indent=2, sort_keys=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"Refusing to replace changed reference: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))
