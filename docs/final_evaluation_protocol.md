# Research framing and final evaluation protocol

**Status:** fixed on 13 October 2026 after exploratory training and validation,
before any final test prediction. This is a prospective test protocol, not a
preregistration of the whole study. It reviews code commit
`0129ae72fe17935c3cf9ad7f2d5b18fd04d634e1`.

## Scientific contribution

The study asks how ImageNet encoder initialization, labelled training-image
count and optimization budget affect low-data aerial segmentation with one
EfficientNet-B0 U-Net. RID supplies visible roof targets in Wartenberg; WHU
supplies building-footprint targets in Christchurch. Replication concerns the
initialization comparison, not zero-shot transfer of a RID-trained model.

The main report has three questions:

1. Does the paired ImageNet advantage persist across 25, 100 and 500 training
   images and three saved training-image selections on each dataset?
2. How do performance gains and overfitting change with training-image count
   under the common 4,000-update recipe?
3. How sensitive is the n=500 comparison to extending training to 8,000 updates?

The RID n=100 random-versus-spatial selection comparison is secondary.
Weight decay, colour, BatchNorm, dropout and boundary-weighted loss experiments
remain exploratory validation analyses. They are not additional test candidates.
No new architecture or loss function is claimed. Publication framing should
emphasize the empirical comparisons and their limitations, rather than a new
algorithm or a universal minimum annotation requirement.

## Focused methodological review

[Review checks](../reports/final_review_checks.json) cover 52 completed runs and
62 checkpoint entries. Canonical configurations, input identities, paired
data/augmentation schedules and decoder initialization, learning-rate phases,
validation-selection rules, saved CSV/summary metrics and checkpoint hashes
agree. Every checkpoint entry was strictly loaded on CPU, without inference.
Recorded initial encoders and decoders match reconstructed initial states; the
cached ImageNet file matches its recorded full hash and torchvision hash prefix.
The active RID roles have zero positive-area cross-role footprint overlaps
under the existing 0.01 m² tolerance. Twenty-nine focused synthetic unit tests
passed. No new training, raw test image/mask reading or test prediction was run.

No blocking numerical inconsistency was found in this scope. This is an
internal, focused review, not independent peer review or a complete software
audit. Important limitations remain:

- Only training-image selection varies across repetitions. Model, data-order
  and augmentation seeds are fixed; all repetitions reuse one validation region
  per dataset and their training subsets can intersect.
- Validation was repeatedly used for exploration and checkpoint selection.
- The historical RID pilot exposed part of today's test region; the evaluation
  policy below explicitly accounts for that history.
- WHU TIFFs omit coordinates, so provider-documented area separation cannot be
  independently reconstructed for every tile. Exact duplicate checks are weaker
  than a geographic overlap check.
- Primary RID subsets use spatial maximin selection, WHU subsets uniform random
  selection. Targets, resolution and empty-image proportions also differ.
- Equal optimizer updates are not equal convergence. n=25 runs process 14,287
  examples in 4,000 updates, versus 16,000 at n=100/500, because incomplete
  batches are retained. Timings are local observations, not universal compute
  benchmarks; reported run totals exclude model construction and weight download.
- Validation-selected checkpoint weights are available. Final-update weights
  are not generally saved. Test endpoints or five-checkpoint ensembles must not
  be inferred from the validation-history CSVs.

## Locked checkpoint matrix

[The checkpoint manifest](../reports/final_evaluation_checkpoints.csv) enumerates
every permitted checkpoint, run path, selected update and hash. Existing runs
are reused; no refitting on training plus validation is planned.

| Analysis | Dataset and scope | Checkpoint entries |
| --- | --- | ---: |
| Primary learning curves | RID and WHU; n=25/100/500; seeds 17/29/43; both initializations; best validation checkpoint through update 4,000 | 36 |
| Secondary image selection | RID n=100 uniform-random selections, seeds 17/29/43, both initializations; paired spatial references are already in the primary matrix | 6 |
| Secondary training budget | RID n=500, seeds 17/29, both initializations; each new 8,000-update run's best checkpoint through 4,000 and through 8,000 | 8 |
| Secondary training budget | WHU n=500, seeds 17/29/43, both initializations; the same two selection ranges | 12 |
| Total | 52 runs; some entries may contain equivalent model states | 62 |

For budget comparisons, pair the two selection ranges **within the same extended
run**, not a historical 4,000-update run with a new 8,000-update run. Test results
describe validation-selected performance under the available budget, including
40 versus 80 selection opportunities. Fixed-update endpoints and final-five
means remain validation-only diagnostics. These are distinct estimands.

The common training recipe is float32, batch size four, AdamW with weight decay
1e-4 excluding biases and one-dimensional parameters, joint D4 augmentation,
normal BatchNorm adaptation and 0.5 BCE + 0.5 soft Dice loss. Learning rate is
1e-3 through update 2,000 and 1e-4 thereafter. The exact saved configuration,
rather than this summary, remains authoritative.

## Frozen test populations

**RID primary:** 254 images from the existing southwest test role. Historical
configs list 25 old training and 289 old validation IDs. The saved development-
usage ledger also covers preflight learning and six qualitative examples;
their inclusion does not change the following exclusions. Three old training IDs
occur directly in the current test role: 146, 1762 and 1782. Metadata geometry
also shows positive-area overlap with test IDs 1402 and 1404. Exclude all five
from the primary evaluation before reading test pixels or making predictions.
This is a conservative history-based policy, not a performance-based exclusion.

**RID sensitivity:** additionally report all 259 images in the original test
region, with the historical exposure disclosed. It cannot replace the primary
population based on which results look better. The original split file and
training inputs are unchanged. [RID eligibility](../reports/final_evaluation_rid_test_ids.csv)
freezes IDs and reasons; the review JSON records the historical config sources
and overlap pairs. Absence of recorded overlap does not prove complete absence
of indirect experimental influence.

**WHU primary:** all 2,416 images in the unchanged provider test role. No
label-based exclusions or foreground balancing. Positive/empty groups are
defined from references during evaluation and reported with their counts.

Use the same population for every comparison within a dataset. Frozen test-input
identities are derived from already existing manifest records in the review
JSON; these records do not substitute for checking file identities at execution.

## Model selection and inference

Select checkpoints solely by the highest mean per-image validation IoU at
threshold 0.5, excluding update zero; earliest update wins ties. Do not select a
seed, training size or initialization based on test results. Restore the exact
manifest checkpoint with strict state/config/input verification.

Use original-orientation native 512 × 512 RGB inputs, divide uint8 by 255 and
apply the saved ImageNet mean/std for both initializations. RID foreground is
`mask != 17`; WHU foreground is `mask == 0`. These products have no declared
invalid mask code. Use float32, batch size four, eval mode and inference mode;
BatchNorm buffers remain fixed during inference. Record the actual device,
software versions and executed code commit. No silent device fallback.

Apply sigmoid to logits and classify foreground with `probability >= 0.5`.
No test-time augmentation, ensembling, calibration, morphology, threshold
selection, component filtering or resizing is introduced.

## Metrics and reporting

Primary endpoint is equally weighted mean per-image foreground IoU. Secondary
metrics are equally weighted per-image Dice and mean non-interpolated pixel AP
over positive-reference images. Do not replace these with pooled-pixel scores
or trapezoidal PR AUC.

For IoU/Dice, empty reference and empty prediction score one; an empty reference
with any predicted foreground scores zero. AP is undefined for an empty target
and excluded with its count reported. Scores use the current canonical metric
implementations; the foreground meaning is stated separately per dataset.

For **both datasets**, additionally report IoU/Dice/AP on positive-reference
images, and the mean predicted foreground fraction and fraction with any
foreground prediction on empty-reference images. If a group is absent, report
count zero and null/undefined metrics, not a failure or an invented zero score.
The existing reference-group helper requires both groups, so the final evaluator
must explicitly handle absent groups before calling it. No minimum component
size is applied to the "any prediction" diagnostic.

For each dataset, size and repetition report both initializations and their
paired difference. Summarize the three subset selections with means, sample SD
(ddof=1) and visible individual values. RID budget comparisons have two
selections. SD reflects subset selection on shared geography, not confidence
intervals or all training randomness. Do not bootstrap pixels or overlapping
tiles as independent observations, perform significance tests, pool the two
datasets into one score, or fit a scaling law from three sizes.

Report all predeclared entries regardless of their performance. Do not choose
the best seed or present test-selected maxima. A numeric requirement for
"reliable" segmentation was not specified externally; no retrospective IoU
cutoff or claim of a universal minimum label budget is introduced. State that
validation annotations and external ImageNet pretraining are additional to the
reported target-task training-image count.

Qualitative test figures are descriptive, never input to optimization. For the
primary n=100 comparison use seed 17 on each dataset. Among positive-reference
images select up to two highest and two lowest per-image **paired mean IoU across
both initializations**, plus the example closest to the median. If empty-reference
images exist, also show the one with the highest paired mean predicted foreground
fraction. Break ties by saved test-ID order. Show both
predictions on identical images and identify these as metric-selected examples.
Any source-image redistribution requires the separately documented permission.

## Execution boundary

This stage fixes the framing, review and protocol only. Commit this document,
the two CSVs and review JSON before implementing/running final inference. The
next stage must validate the exact manifest and evaluation populations, handle
undefined groups, then write per-image results, summaries and an execution
record. No test evaluator has been implemented or run here.

After test access, preserve the protocol and report failures or negative
findings. A technical correction must be documented and applied consistently
to all affected entries; test outcomes must not drive another model, seed,
threshold or training-budget search. Any later research is explicitly labelled
post-test exploration.
