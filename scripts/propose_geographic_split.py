"""Create a non-binding metadata-only proposal for a compact test region."""

from pathlib import Path
import json

from roofseg.integrity import load_json
from roofseg.split_proposal import build_split_proposal, plot_split_proposal, prior_development_usage


ROOT = Path(__file__).resolve().parents[1]
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
old_splits = load_json(ROOT / "data/metadata/splits.json")
usage = prior_development_usage(ROOT, manifest, old_splits)
proposal = build_split_proposal(manifest, old_splits, usage)

payload = {"usage": usage, "proposal": proposal}
output = ROOT / "reports/geographic_split_proposal.json"
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

figure = plot_split_proposal(manifest, proposal)
figure.savefig(ROOT / "reports/figures/geographic_split_proposal.png", dpi=180, bbox_inches="tight")

duplicate_exclusions = sum(
    count for reason, count in proposal["exclusion_reason_counts"].items()
    if reason.startswith("exact_duplicate_of_")
)
distances = {row["roles"]: row for row in proposal["distance_summary_m"]}
qualitative_ids = ", ".join(row["id"] for row in usage["qualitative_development_examples"])
run_statuses = {}
for run in usage["executed_runs"]:
    run_statuses[run["status"]] = run_statuses.get(run["status"], 0) + 1
report = f"""# Candidate geographic evaluation split

**Status: proposal only. It has not replaced `data/metadata/splits.json` and must
not be used to start training before review.**

## Prior development use

The saved artifacts show {run_statuses.get('completed', 0)} completed
random-initialized run and {run_statuses.get('interrupted', 0)} interrupted
ImageNet-initialized attempts under the old split. They used the same 25-image
Seed 17 training subset, and their evaluations exposed all 289 old validation
images to checkpoint selection. The real-data preflight used IDs 8 and 144 for
its small learning check; these are already in that 25-image subset. The executed
Stage 1 gallery opened six development images: {qualitative_ids}.

All 1,880 samples also participated in checksum, pairing, coordinate, label-code
and aggregate label-area checks. These technical audits are recorded separately:
they did not fit a model, select a checkpoint or inspect prediction errors.

## Preferred proposal

The candidate holds out a compact southwest coordinate window in EPSG:25832,
from easting {proposal['test_area_bounds_m'][0]:.0f} to
{proposal['test_area_bounds_m'][2]:.0f} m and northing
{proposal['test_area_bounds_m'][1]:.0f} to
{proposal['test_area_bounds_m'][3]:.0f} m. Only complete image footprints inside
the window are eligible. The boundary was chosen from coordinates and sample
counts, outside the existing northern validation block; no image content, mask
distribution, score or prediction error was used.

| Proposed role | Images |
| --- | ---: |
| Training | {proposal['counts']['training']} |
| Validation | {proposal['counts']['validation']} |
| Test | {proposal['counts']['test']} |
| Excluded | {proposal['counts']['excluded']} |

The old northern validation set remains the development validation set because
it has already influenced model selection. The training pool remains large enough
for nested 25, 100 and 500-image subsets. Of the former provider test images,
{proposal['old_provider_test_transition']['training']} would become development
training material, {proposal['old_provider_test_transition']['test']} fall inside
the new geographic test area, and none were previously used for model fitting or
selection. This role change must be explicit if the proposal is adopted.

![Proposed geographic roles and full-footprint boundary](figures/geographic_split_proposal.png)

The left panel shows the complete municipality and the southwest selection
window. The right panel displays full image footprints near the boundary: green
footprints are proposed test images, blue footprints remain training material,
and grey footprints are excluded. The rectangle is a selection boundary, not an
additional distance buffer.

## Leakage and exclusions

Historical use is disclosed but is not an exclusion rule for the restarted
study. The proposed test set contains three images used by the discarded pilot
for training (IDs 146, 1762 and 1782), no old validation image, and no qualitative
gallery example. The new split therefore must not be described as untouched
since the beginning of the project. Its selection remains independent of old
scores, predictions, errors and label distributions.

| Exclusion basis | Images |
| --- | ---: |
| Footprint intersects retained northern validation | {proposal['exclusion_reason_counts']['footprint_intersects_validation']} |
| Footprint straddles the southwest boundary | {proposal['exclusion_reason_counts']['southwest_test_boundary_straddler']} |
| Exact duplicate copy | {duplicate_exclusions} |

Positive-area footprint overlaps are 0 for training–validation, training–test
and validation–test. Nearest complete-footprint distances are summarized below;
the small training–validation minimum reflects the explicit decision not to add
a buffer.

| Roles | Minimum (m) | 10th percentile (m) | Median (m) |
| --- | ---: | ---: | ---: |
| Training–validation | {distances['training–validation']['minimum']:.1f} | {distances['training–validation']['p10']:.1f} | {distances['training–validation']['median']:.1f} |
| Training–test | {distances['training–test']['minimum']:.1f} | {distances['training–test']['p10']:.1f} | {distances['training–test']['median']:.1f} |
| Validation–test | {distances['validation–test']['minimum']:.1f} | {distances['validation–test']['p10']:.1f} | {distances['validation–test']['median']:.1f} |

## Recommendation and limitation

This is the preferred proposal because it creates one compact, geographically
defined test area without old validation images, preserves the already exposed
northern block for validation, and leaves ample training data. It measures transfer to a held-out part of
Wartenberg, not transfer to another city. Spatial proximity remains because no
buffer is requested. Prior pilot use is a historical limitation, not a geometric
hole or a permanent role constraint in the restarted design.

If approved, the next implementation step is to create a new versioned split,
regenerate nested training subsets from its training role, update the method
documentation and start every model from fresh random or original ImageNet
weights with a newly initialized decoder. Old checkpoints remain historical and
must not be reused.
"""
(ROOT / "reports/geographic_split_proposal.md").write_text(report, encoding="utf-8")

print(json.dumps({
    "status": proposal["status"],
    "counts": proposal["counts"],
    "overlaps": proposal["cross_role_positive_area_overlaps"],
    "prior_use_intersection": proposal["prior_use_intersection"],
}, indent=2))
