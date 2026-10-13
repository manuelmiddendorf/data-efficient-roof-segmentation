# Data-Efficient Roof Segmentation

*Pretraining, training-image count and optimization budget in aerial segmentation.*

## Abstract

This reproducible empirical study compares random initialization with an
ImageNet-pretrained encoder in one EfficientNet-B0 U-Net. Paired experiments
use 25, 100 and 500 labelled training images on the Roof Information Dataset
(RID, Wartenberg) and WHU Aerial (Christchurch). ImageNet initialization improves
validation IoU in every primary paired comparison. Training-budget diagnostics
show that the size of this advantage depends on optimization time. These are
validation findings; the final test protocol is fixed and test evaluation is
pending.

## Motivation

Pixel-level annotations are expensive. The question is how performance changes
with the number and selection of labelled images, and how much pretraining helps
under a limited training budget. We distinguish faster learning from higher
attainable performance and avoid treating repeated views as independent labels.

## Methods

RID supplies visible roof targets; WHU supplies building-footprint targets.
The saved RID split contains 1,210 training, 289 validation and 259 test images,
with no positive-area cross-role footprint overlap. WHU retains its provider's
4,736/1,036/2,416 roles; its prepared TIFFs omit coordinates, so complete spatial
separation cannot be independently reconstructed for every tile.

Each dataset has three saved nested training-image selections (seeds 17, 29,
43), three sizes and two initializations: 18 primary runs per dataset. RID
uses spatial maximin selection and WHU uses uniform random selection. Both
initializations receive the same images and augmentation schedule within a pair
and a fresh decoder with the same initial state. Model, ordering and augmentation
seeds remain fixed across subset repetitions.

The common recipe uses native 512 × 512 RGB inputs, ImageNet normalization,
joint D4 augmentation, ordinary BatchNorm adaptation, AdamW, batch size four,
and equally weighted BCE and soft Dice loss. Training lasts 4,000 updates with
learning rate 1e-3 through update 2,000 and 1e-4 thereafter. Checkpoints are
selected on validation mean per-image IoU at threshold 0.5. Separate n=500
experiments extend the same schedule to 8,000 updates.

## Main validation results

Mean selected IoU ± sample SD across three training-image selections:

| Dataset | Training images | Random | ImageNet |
| --- | ---: | ---: | ---: |
| RID | 25 | 0.781 ± 0.001 | 0.833 ± 0.010 |
| RID | 100 | 0.831 ± 0.001 | 0.866 ± 0.003 |
| RID | 500 | 0.840 ± 0.001 | 0.878 ± 0.002 |
| WHU | 25 | 0.636 ± 0.019 | 0.714 ± 0.011 |
| WHU | 100 | 0.717 ± 0.015 | 0.799 ± 0.017 |
| WHU | 500 | 0.749 ± 0.011 | 0.837 ± 0.015 |

Sources: [RID summary](reports/data_efficiency_repetition_summary.csv) and
[WHU summary](reports/whu_learning_curve_summary.csv). SD describes subset
selection on shared validation geography, not a confidence interval.

Performance increases with data for both initializations; ImageNet leads in all
nine primary pairs on each dataset. At 25 images, large train–validation gaps
and degradation after the selected checkpoint indicate overfitting. At 500,
late improvement motivates the separate training-budget analysis. Longer
training helps Random on average, but does not remove the observed ImageNet
advantage under the examined protocols; convergence is not established.

A secondary RID n=100 comparison finds a mean spatial-selection advantage of
0.0069 IoU for Random and approximately zero for ImageNet. It does not isolate
geographic coverage from changes in sampled image content. Targeted
regularization experiments found no consistent shared replacement for the
reference recipe and remain exploratory validation analyses.

## Final evaluation and limitations

[The final protocol](docs/final_evaluation_protocol.md) fixes 36 primary
checkpoint entries and 26 secondary entries for selection and budget comparisons.
[The internal review](reports/final_review_checks.json) verified 52 runs,
strictly loaded all 62 checkpoint entries and passed 29 focused unit tests.
No final test prediction has been made.

The primary RID test population is fixed at 254 images: three directly used
historical-pilot IDs and two footprint-overlapping neighbours are excluded from
the original 259-image region. All 259 remain a declared sensitivity analysis;
the saved split is unchanged. WHU uses all 2,416 provider test images. This
history and the differing spatial evidence are documented in the protocol.

Repeated validation use, shared geographic regions, overlapping subsets and
fixed non-subset seeds limit uncertainty claims. Training-image counts exclude
validation annotations and external ImageNet pretraining. Targets, ground
resolution and sampling differ between datasets, so absolute scores are not a
controlled cross-dataset comparison. The study does not establish a universal
minimum annotation budget or propose a new segmentation architecture.

## Reports and reproduction

- [Data and geographic split](notebooks/01_data_exploration.ipynb)
- [Training pilot](notebooks/02_training_pilot.ipynb)
- [Optimization](notebooks/03_optimization.ipynb)
- [RID data efficiency, image selection and budget](notebooks/04_data_efficiency.ipynb)
- [Exploratory targeted experiments](notebooks/05_targeted_experiments.ipynb)
- [WHU external-replication data](notebooks/06_external_replication_data.ipynb)
- [WHU data efficiency and training budget](notebooks/07_external_replication_pilot.ipynb)

[Setup](docs/setup.md) gives reproduction commands; [data documentation](docs/data.md)
records source and label interpretation. [The project plan](WORK_PLAN.md) keeps
internal decisions separate from the scientific report. Analysis notebooks load
saved results rather than silently launching training.

Download data from the [RID provider](https://mediatum.ub.tum.de/1655470) and
[WHU provider](https://gpcv.whu.edu.cn/data/building_dataset.html); raw data,
weights and environments are not bundled. Dataset and imagery permissions are
separate from the project code. Codex assisted with implementation, analysis,
tests and documentation; this is not an independent external peer review.
