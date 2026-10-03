import json
from pathlib import Path

from roofseg.split_proposal import build_split_proposal


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_split_is_complete_disjoint_and_not_shaped_by_prior_use():
    manifest = json.loads((ROOT / "data/metadata/data_manifest.json").read_text())
    old_splits = json.loads((ROOT / "data/metadata/splits.json").read_text())
    old_training_ids = old_splits["training_subsets"]["repetitions"]["1"]["subsets"]["25"]
    usage = {
        "model_or_qualitative_ids": old_training_ids + old_splits["roles"]["validation"],
        "model_training_ids": old_training_ids,
        "model_validation_ids": old_splits["roles"]["validation"],
        "qualitative_development_examples": [],
    }

    proposal = build_split_proposal(manifest, old_splits, usage)
    roles = proposal["roles"]
    assigned = [sample_id for ids in roles.values() for sample_id in ids]

    assert len(assigned) == len(set(assigned)) == len(manifest["samples"])
    assert proposal["cross_role_positive_area_overlaps"] == {
        "training_validation": 0,
        "training_test": 0,
        "validation_test": 0,
    }
    assert len(roles["training"]) >= 500
    assert set(("146", "1762", "1782")) <= set(roles["test"])
    assert not set(roles["test"]) & set(old_splits["roles"]["validation"])
    for group in manifest["audit"]["exact_duplicate_image_groups"]:
        assert len(set(group) - set(roles["excluded"])) <= 1
