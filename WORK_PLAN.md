# Research plan and project status

## Research question

How many labelled roof images are needed for useful segmentation on spatially
separated imagery, and how do pretraining and training choices affect that need?

The intended contribution is a reproducible empirical study with readable code,
learning curves, paired comparisons and an honest account of uncertainty.
The initial architecture is an EfficientNet-B0 U-Net. Random initialization and
ImageNet encoder initialization are the main comparison; both start with a fresh
segmentation decoder.

## Current status

Stage 1 is complete on branch `codex/data-foundation`. RID 1.0 was downloaded
from the provider, every required file was checked against the provider SHA-256
list, and the data contract was verified for all 1,880 image/mask pairs. The
binary target uses roof codes 0–16 and background code 17, as confirmed from the
provider's mask-generation code. This corrects an initially attempted reversed
mapping before any result was committed.

The active split `rid_southwest_test_north_validation_v2` holds out a compact
259-image southwest test area and retains 289 northern validation images. Strict
complete-footprint checks and exact-duplicate removal leave 1,210 training and
122 excluded images, with no additional distance buffer and no positive-area
cross-role overlap. Three deterministic nested training-subset repetitions use
seeds 17, 29 and 43.

An early 25-image pilot under the former provider-test design was discarded when
the evaluation question changed to transfer into a held-out part of Wartenberg.
One random run completed and two ImageNet attempts were interrupted; these remain
historical artifacts and are not reused. Three images in the active test region
(146, 1762 and 1782) occurred in that old training subset. This history is not
erased, but it did not determine the approved geographic boundary. The restarted
Stage 2 pilot is complete: all six runs began from fresh random parameters or the
original verified ImageNet weights with a newly initialized decoder. The 259
active test images remained outside all new training, model selection and
qualitative analysis.

Under the common 2,000-update recipe, ImageNet initialization achieved validation
IoU 0.834, 0.862 and 0.870 at 25, 100 and 500 images; random initialization
achieved 0.785, 0.813 and 0.826. The corresponding paired gains were 0.049,
0.050 and 0.044. All values are exploratory validation results from repetition 1
(subset seed 17), not independent test performance. The six runs required 61.8
minutes of optimization and 77.9 minutes including evaluation on MPS.

The first Stage 3 block compares learning rates 1e-4, 3e-4 and 1e-3 at 100
training images. Four fresh runs were paired with the two verified Stage 2
references. The best observed validation IoU is 0.817 for random initialization
and 0.865 for ImageNet initialization, both at 1e-3. ImageNet remains ahead at
every matched rate by 0.048–0.064 IoU. The four new runs required 42.2 minutes
of optimization and 52.7 minutes including evaluation on MPS. These are still
single-repetition, validation-selected results; no test image was evaluated.

The training-horizon block then ran four fresh 4,000-update configurations at
`1e-4` and `1e-3`. Random initialization improved from best IoU 0.793 to 0.811
at `1e-4` and from 0.817 to 0.831 at `1e-3` when comparing the first 2,000
updates with the full curve. ImageNet `1e-4` improved from 0.857 to 0.863;
ImageNet `1e-3` retained its 0.865 step-2,000 maximum. The four runs required
92.5 minutes of optimization and 113.1 minutes including evaluation on MPS.
The results remain exploratory validation evidence from repetition 1; the
locked test role was not evaluated.

## Dataset and scientific decisions

Start by evaluating RID (2022), not automatically substituting RID2:

- [Dataset and download](https://mediatum.ub.tum.de/1655470)
- [Paper](https://mediatum.ub.tum.de/doc/1735413/document.pdf)
- [Upstream repository](https://github.com/TUMFTM/RID)

The dataset record describes 1,880 roof-centred aerial images and reviewed roof
segment masks. The paper locates them in Wartenberg, Germany, and warns about
overlapping images. Verify the downloaded release and metadata before relying
on these descriptions. Inspect the exact label codes, annotation coverage,
coordinate system, footprint geometry and supplied splits. Derive binary roof
targets from roof-segment labels, not roof-superstructure labels.

The dataset record lists CC BY-NC 4.0; imagery has separate source notices.
Record attribution and applicable terms. Reference the original download rather
than adding raw images to Git. Public example figures need compatible usage and
attribution. Do not import application-task images or private documents.

Choose train/validation/test regions before training. Start from the supplied
geographic split information, but verify it: remove crossing footprints or use
a documented spatial buffer where needed. Report retained and excluded counts.
Do not force arbitrary percentages at the cost of geographic separation.

Create exact nested subsets only inside the training pool. Proposed sizes are
25, 50, 100, 250, 500 and the complete retained training pool, subject to available
data and spatial redundancy. Three paired repetitions are the starting proposal,
not a substitute for evaluating independent regions.

Validation annotations are additional to the stated training-label budget.
Multiple crops or overlapping views are not independent labelled examples.
Count and describe this limitation. Keep test performance and test error
inspection out of development decisions.

## Implementation stages

| Stage | Deliverable | Discussion point |
| --- | --- | --- |
| 1. Foundation and data | Local environment, data import, geographic splits, shared data records, meaningful tests and one executed data notebook | Are labels, coverage and spatial separation suitable? |
| 2. Training and pilot | Reused training components, simple run artifacts, restoration checks and paired pilot runs at 25/100/500 images | Does each initialization learn, how long does it take, and what errors dominate? |
| 3. Optimization | Small comparisons of learning rate, schedule and weight decay | Which adequately trained protocols support a fair comparison? |
| 4. Targeted experiments | Selected blocks for BatchNorm, augmentation, boundary objectives and further regularization | Which effects repeat and which merit follow-up? |
| 5. Learning curves | Confirmed settings over training sizes and repeated subset/seed choices | What can we conclude about target-task label requirements? |
| 6. Final evaluation and presentation | Locked test evaluation, concise paper-style README, executed notebooks and reproduction commands | Final technical/scientific review before publication |

Stages 1 and 2 must be working and understood before a broad experiment series.
Stage 1 includes the data records; training-run JSON writers and checkpoint
management belong to Stage 2. Do not scaffold unused later modules or notebooks.

## Experiment backlog

This is a prioritized menu, not a Cartesian grid or an instruction to run
everything. Confirm settings after pilot evidence. Initially screen one factor
at a time at two training sizes, usually 100 and 500, then confirm promising
effects with additional seeds. Combinations need a specific reason and a paired
reference. The baseline appropriate to each block must be explicit.

| Block | Initial candidates | Interpretation to preserve |
| --- | --- | --- |
| Initialization | Random versus ImageNet encoder | Same architecture; adequate convergence and comparable tuning effort |
| Learning rate | 1e-4, 3e-4, 1e-3 | A fine-tuning rate need not suit training from scratch |
| Schedule | Constant versus cosine decay | Specify total updates/horizon and relation to early stopping |
| Weight decay | 0, 1e-4, 1e-3, 1e-2 | Keep parameter-group policy fixed; record learning rate and update count |
| BatchNorm | Update encoder running statistics versus freeze them for pretrained models | Separate running buffers from trainable affine parameters |
| Batch size | 4 versus 8 if memory permits | Changes both optimization and BatchNorm; record updates and runtime |
| Colour | D4 alone versus mild brightness/contrast | Alter raw RGB before fixed encoder normalization |
| Geometry | Additional free rotations or conservative resized crops | Transform labels and validity jointly; avoid artificial boundary targets |
| Boundary A | BCE + Dice with extra band-pixel BCE weight, divided by valid-pixel count | Images with more contour pixels contribute more additional loss |
| Boundary B | BCE + Dice plus BCE averaged within the band | Study normalization across images; not a wholly unrelated algorithm |
| Boundary C | BCE + Dice plus signed-distance boundary term | Specify distance units, normalization, validity and empty-mask behavior |
| Decoder dropout | No additional dropout versus Dropout2d around 0.1 | Fix its placement; evaluate with dropout disabled |
| Transfer regularization | Partial encoder freezing or L2-SP | Explicitly preserve some pretrained information |
| Decoder capacity | Standard versus reduced channels | A separate capacity comparison, not only an optimizer change |
| Training diversity | Random versus spatially distributed selection | Select only within the training pool using permitted information |

For a boundary block, begin with at most two strengths per variant and a fixed
band definition. Do not assume equal coefficients mean equal influence for
different terms. Do not mistake image/crop borders or missing observations for
roof boundaries.

For the new baseline, the proposed decay policy excludes biases and normalization
parameters; verify and record this policy once. All numerical choices above remain
proposals until a block configuration is agreed and saved.

Learning-rate schedules, stopping rules and maximum updates must give the small
subsets and random initialization enough opportunity to learn. Report any
initialization-specific settings and define whether the comparison concerns a
common recipe or separately tuned recipes with equal search budgets.

## Evaluation and iteration

The proposed primary metric is equally weighted mean per-image roof IoU at a
fixed threshold, initially 0.5. Secondary diagnostics include Dice, AP, boundary
IoU with a fixed tolerance, per-image failures and runtime. Specify empty-mask
and validity conventions before implementation. Predeclare any success criterion
for answering "how many labels?" before confirmatory evaluation.

For diagnosing overfitting, compare train and validation metrics under matching
evaluation preprocessing and model mode. Training-mode loss with augmentation
or dropout is not directly equivalent to validation-mode loss.

After a pilot or experiment block, record only:
1. The hypothesis and configuration actually tested.
2. A compact result table, curves and paired image examples.
3. What the evidence supports and its limitations.
4. The next proposed block and estimated run/runtime budget.

Allow new hypotheses from training/validation evidence; do not silently expand
the execution scope. Confirm small gains across repetitions. Keep a brief
decision record here rather than creating a document for every trial.
Freeze the final comparison before test evaluation; do not use test feedback
to select another reported winner.

## Decision record

- The early provider-test pilot was stopped when the study adopted a compact
  southwest test region. Its artifacts remain historical and were not reused.
- The active split was fixed before the restart. Exact-role reconstruction,
  zero positive-area cross-role overlap and test exclusion were verified before
  training.
- The paired Stage 2 pilot used the unchanged common recipe. ImageNet
  initialization led at every budget, but n=25 ImageNet overfit after an early
  best checkpoint and several runs selected late checkpoints. Equal updates did
  not establish equal convergence or an optimal learning rate.
- The first Stage 3 block tested learning rates 1e-4, 3e-4 and 1e-3 at n=100.
  ImageNet led at every matched rate. Both 1e-4 runs selected step 2,000 and
  still had higher training objectives, so their lower scores reflect
  underoptimization at this horizon rather than established lower attainable
  performance.
- Thresholds 0.1–0.9 were evaluated only for the stored 3e-4 references.
  ImageNet remained ahead by 0.050–0.070 IoU; changing the threshold therefore
  did not explain the reference gap. The primary threshold remains 0.5.
- The training-horizon block completed four fresh 4,000-update runs at `1e-4`
  and `1e-3`. Longer training raised both random-initialization maxima and the
  ImageNet `1e-4` maximum, but ImageNet `1e-3` remained the overall leader with
  its step-2,000 IoU of 0.865. Endpoints and second-half variation show that a
  higher selected maximum alone is not evidence of stable convergence.
- The next recommended block is confirmation on another predeclared subset
  repetition, pairing random and ImageNet initialization at `1e-3` under the
  common 4,000-update cap. This would test repeatability before introducing a
  scheduler or changing weight decay. It is a proposal and has not been run.

## Git and delivery

Plan four PR-sized stages:

1. Project foundation, dataset handling and geographic separation.
2. Training foundation, run artifacts and pilot results.
3. Optimization and targeted experiments.
4. Learning curves, final evaluation and public presentation.

Use roughly two to four coherent commits per PR where useful, not one per tiny
edit or training run. For a new repository, a minimal initial documentation commit
may establish the main branch before the first implementation branch. Review the
training foundation broadly and perform another broader review before publication.
Use focused checks between those points.

The first task should initialize local Git if necessary and create the Stage 1
branch. Use a configured remote if available; otherwise finish and commit local
work before requesting the destination repository. Do not invent an account or
publish a repository merely to resolve missing remote configuration.

Repository-root files state durable project facts. Personal paths and copied
task prompts remain in ignored .local/. A future docs/setup.md should list only
working setup and execution commands.

## Method references

- [RID](https://mediatum.ub.tum.de/doc/1735413/document.pdf)
- [AdamW](https://arxiv.org/abs/1711.05101)
- [BatchNorm behavior](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm2d.html)
- [Boundary loss](https://proceedings.mlr.press/v102/kervadec19a.html)
- [Boundary IoU](https://arxiv.org/abs/2103.16562)
- [L2-SP](https://proceedings.mlr.press/v80/li18a.html)
