"""Verify RID release 1.0 and create the two shared data references."""

from pathlib import Path
import json

from roofseg.audit import build_data_manifest, build_report, build_splits
from roofseg.integrity import write_new_json


ROOT = Path(__file__).resolve().parents[1]
manifest = build_data_manifest(ROOT)
splits = build_splits(manifest, ROOT)
write_new_json(ROOT / "data/metadata/data_manifest.json", manifest)
write_new_json(ROOT / "data/metadata/splits.json", splits)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/data_audit.json").write_text(
    json.dumps(build_report(manifest, splits), indent=2) + "\n", encoding="utf-8"
)
print(f"Verified {manifest['audit']['pair_count']} image/mask pairs.")
print("Final roles:", splits["counts"])
