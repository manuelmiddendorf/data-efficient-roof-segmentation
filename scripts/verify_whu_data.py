"""Recompute WHU references and require agreement with saved metadata."""

from pathlib import Path

from roofseg.integrity import load_json
from roofseg.whu_data import build_whu_manifest, build_whu_splits


ROOT = Path(__file__).resolve().parents[1]
metadata_root = ROOT / "data/metadata/whu"
manifest = build_whu_manifest(ROOT)
if manifest != load_json(metadata_root / "data_manifest.json"):
    raise SystemExit("Current WHU files differ from data_manifest.json; reference not changed.")
splits = build_whu_splits(manifest)
if splits != load_json(metadata_root / "splits.json"):
    raise SystemExit("Current WHU role derivation differs from splits.json; reference not changed.")
print(f"WHU integrity and provider roles verified for {len(manifest['samples'])} samples.")
