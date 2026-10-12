"""Audit WHU Aerial and create its canonical manifest and split record."""

from pathlib import Path
import json

from roofseg.integrity import write_new_json
from roofseg.whu_data import build_whu_manifest, build_whu_report, build_whu_splits


ROOT = Path(__file__).resolve().parents[1]
manifest = build_whu_manifest(ROOT)
splits = build_whu_splits(manifest)
metadata_root = ROOT / "data/metadata/whu"
write_new_json(metadata_root / "data_manifest.json", manifest)
write_new_json(metadata_root / "splits.json", splits)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/whu_data_audit.json").write_text(
    json.dumps(build_whu_report(manifest, splits), indent=2) + "\n", encoding="utf-8"
)
print(f"Verified {manifest['audit']['pair_count']} WHU image/label pairs.")
print("Provider roles:", splits["counts"])
