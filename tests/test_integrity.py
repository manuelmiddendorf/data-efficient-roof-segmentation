from pathlib import Path

import pytest

from roofseg.integrity import sha256_file, verify_files, write_new_json


def test_checksum_verification_detects_changed_bytes(tmp_path: Path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"original")
    reference = {"sample.bin": sha256_file(path)}
    verify_files(tmp_path, reference, ["sample.bin"])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_files(tmp_path, reference, ["sample.bin"])


def test_saved_reference_is_not_silently_replaced(tmp_path: Path):
    path = tmp_path / "reference.json"
    write_new_json(path, {"value": 1})
    with pytest.raises(FileExistsError, match="Refusing"):
        write_new_json(path, {"value": 2})
