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

A late-trajectory analysis then separated linear change from residual variation
over steps 2,100–4,000. One predeclared schedule block ran exactly two fresh
configurations, using `1e-3` through step 2,000 and `1e-4` thereafter. The drop
reduced late IoU residual SD from 0.0075 to 0.0012 for random initialization and
from 0.0055 to 0.0010 for ImageNet. ImageNet best IoU rose from 0.865 to 0.868;
random best IoU was effectively tied (0.831 constant versus 0.830 drop), while
its endpoint rose from 0.806 to 0.830. The two fresh runs required 51.3 minutes
including evaluation. These remain single-repetition validation results; the
locked test role was not accessed.

The selected fixed-drop recipe was then repeated on the saved Seed 29 n=100
subset while model, order and augmentation seeds remained fixed. The new random
and ImageNet runs reached best IoU 0.831 and 0.863, compared with 0.830 and
0.868 on Seed 17. The paired ImageNet-minus-random advantage repeated at +0.032
versus +0.038. Seed 17 and Seed 29 share 24 training IDs; both use the same
validation region. The two new runs required 51.3 minutes including evaluation.
No constant-rate Seed 29 control or test-role evaluation was performed.

A preliminary fixed-recipe data-efficiency curve then compared the nested Seed
17 subsets at 25, 100 and 500 images. Best validation IoU rose from 0.781 to
0.830 to 0.840 for random initialization and from 0.837 to 0.868 to 0.879 for
ImageNet. The paired pretraining differences were +0.056, +0.038 and +0.039.
The four fresh runs required 102.3 minutes including evaluation. The n=25 runs
show strong train–validation gaps and late loss deterioration, while n=500 still
improves after the step-2,000 rate drop. This is one nested subset repetition on
one reused validation region; the test role remained locked.

The data-efficiency curve is now complete for the saved Seed 29 subset family.
Four fresh n=25 and n=500 runs joined the existing n=100 pair; they required
80.1 minutes of optimization and 99.6 minutes including evaluation on MPS.
Random/ImageNet best validation IoU was 0.783/0.842 at n=25 and 0.842/0.877 at
n=500, with paired advantages of +0.059 and +0.035. Together, both repetitions
show positive but smaller 100-to-500 gains, n=25 train–validation separation and
continued late learning for n=500 random initialization. Seeds 17 and 29 share
4, 24 and 293 training IDs at n=25, 100 and 500. Only training-image selection
varies; both repetitions use the same validation region and all other seeds.
No test-role image was accessed.

The first targeted regularization block increased AdamW weight decay from
`1e-4` to `1e-2` at n=25 for both initializations and subset seeds 17 and 29.
The four fresh runs required 77.3 minutes of optimization and 96.6 minutes
including evaluation on MPS. Strong-minus-reference best-IoU changes were
+0.0035 and −0.0049 for random initialization and −0.0026 and −0.0063 for
ImageNet. The train–validation IoU gap increased in all four pairs, and late
loss/stability changes were inconsistent. The stronger decay is not adopted;
`1e-4` remains the supported baseline. The locked test role was not accessed.

The next targeted block added mild whole-image brightness and contrast at n=25,
using independent `Uniform(0.85, 1.15)` factors on every training occurrence
with a dedicated seed. Four fresh runs required 91.7 minutes of optimization and
113.1 minutes including evaluation on MPS. Random initialization improved by
+0.0076 and +0.0074 best validation IoU across subset seeds 17 and 29; ImageNet
changed by −0.0071 and −0.0007. The comparable train–validation gap increased
slightly in all four pairs. The rule is therefore retained only as a promising
Random-specific candidate, not adopted as the shared recipe. The locked test
role was not accessed.

The encoder-BatchNorm block then kept the original ImageNet running statistics
fixed during training for the two n=25 subset selections. The two fresh runs
required 36.7 minutes of optimization and 46.1 minutes including evaluation on
MPS. Best validation IoU fell from 0.837 to 0.791 for Seed 17 and from 0.842 to
0.803 for Seed 29; endpoint IoU, Dice and AP also fell, while the comparable
train–validation gap increased in both pairs. Fixed ImageNet statistics are not
adopted. This result does not establish that BatchNorm caused any earlier
augmentation result. The locked test role was not accessed.

The decoder channel-dropout block then applied `p=0.1` after the final decoder
block and before the prediction head for all four n=25 pairs. Its dedicated RNG
seed did not alter other recorded random streams. The four fresh runs required
78.6 minutes of optimization and 97.3 minutes including evaluation on MPS.
Dropout-minus-reference best-IoU changes were −0.0009/+0.0037 for Random and
−0.0025/+0.0050 for ImageNet across subset seeds 17/29. Endpoint IoU, validation
loss, fit gaps and late stability were inconsistent across pairs. Do not adopt
this dropout setting as the shared recipe. The locked test role was not accessed.

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
- Late linear trends and residual SD were computed separately for validation
  IoU and validation loss over steps 2,100–4,000. Residual SD is descriptive,
  not a confidence interval or convergence threshold; adjacent checkpoints are
  dependent.
- Exactly two fresh runs tested the predeclared `1e-3 → 1e-4` change at step
  2,001. The drop substantially reduced late IoU variation for both
  initializations, improved the ImageNet best and endpoint scores, and preserved
  the random best while improving its endpoint.
- The common fixed-drop recipe was repeated on the saved Seed 29 n=100 subset.
  The ImageNet-minus-random best-IoU difference remained positive (+0.032 versus
  +0.038 on Seed 17), and all four late trajectories had low residual variation.
  This varies training-image selection only; both repetitions share one
  validation region and 24 training IDs.
- The fixed-recipe data-efficiency curve now covers saved subset seeds 17 and
  29 at n=25, 100 and 500. Performance increased with data in both repetitions,
  with smaller gains from 100 to 500; pretraining remained beneficial at every
  size. Both n=25 selections show overfitting, while both n=500 random runs
  improve late.
- The predeclared n=25 Weight Decay `1e-2` block completed four fresh runs
  against paired `1e-4` references. It improved only Seed 17 Random and worsened
  the other three best-IoU comparisons; the comparable train–validation gap grew
  in all four. Do not adopt `1e-2`. The next proposed block is predeclared mild
  brightness/contrast augmentation at n=25 across both subset selections.
- The mild colour block completed four fresh paired runs with independent
  brightness and contrast factors in `[0.85, 1.15]`. Random initialization
  improved by about +0.0075 best IoU in both subset selections, while ImageNet
  did not improve and the comparable train–validation gap increased in all four
  pairs. Do not adopt it as a shared recipe. A limited encoder-BatchNorm block
  for the two n=25 ImageNet selections was selected as the next block.
- The encoder-BatchNorm block completed two fresh ImageNet runs with fixed
  pretrained running statistics. Best IoU decreased by 0.0459 and 0.0386, and
  the comparable train–validation gap increased in both subset selections. Keep
  normal BatchNorm adaptation.
- The decoder channel-dropout block completed four fresh n=25 runs with `p=0.1`
  at the final decoder feature map. Best-IoU effects ranged from −0.0025 to
  +0.0050 and did not repeat consistently across initialization and subset
  selection; endpoint and stability diagnostics were also mixed. Keep the
  shared reference without decoder dropout. A boundary-aware objective is the
  next bounded candidate, subject to a fixed boundary definition and at most two
  predeclared variants.

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
