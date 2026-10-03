"""Recompute RID references and require exact agreement with saved metadata."""

from pathlib import Path

from roofseg.audit import build_data_manifest, build_splits
from roofseg.integrity import load_json


ROOT = Path(__file__).resolve().parents[1]
manifest = build_data_manifest(ROOT)
saved_manifest = load_json(ROOT / "data/metadata/data_manifest.json")
if manifest != saved_manifest:
    raise SystemExit("Current RID files differ from data_manifest.json; reference not changed.")
splits = build_splits(manifest, ROOT)
saved_splits = load_json(ROOT / "data/metadata/splits.json")
if splits != saved_splits:
    raise SystemExit("Current split derivation differs from splits.json; reference not changed.")
print(f"RID integrity and geographic split verified for {len(manifest['samples'])} samples.")
