# Data-Efficient Roof Segmentation

*How many labelled images do we need, and how much does pretraining help?*

This reproducible study examines binary roof segmentation in aerial imagery when
annotations are limited. It uses the 1,880-image [Roof Information Dataset
(RID)](https://mediatum.ub.tum.de/1655470) and compares one EfficientNet-B0 U-Net
initialized randomly or with an ImageNet-pretrained encoder. Training,
validation and test regions are geographically separated using full image
footprints; the compact southwest test region remains locked during development.

## Current evidence

RID 1.0 is checksum-verified. The active split contains 1,210 training images, a
289-image northern validation region and a 259-image southwest test region, with
no positive-area cross-role footprint overlap. The common recipe trains for
4,000 updates with AdamW, D4 augmentation, learning rate `1e-3` through update
2,000 and `1e-4` thereafter.

The fixed recipe was evaluated on three saved, nested training-image selections
(seeds 17, 29 and 43), three sizes and both initializations: 18 runs in total.
Mean best validation IoU across image selections increases from 0.781 to 0.831
to 0.840 for Random and from 0.833 to 0.866 to 0.878 for ImageNet at 25, 100
and 500 images. ImageNet leads in every paired comparison; best-IoU differences
range from +0.032 to +0.059. The observed sample SD across three image
selections is at most 0.0104. These SDs describe image-selection variation, not
confidence intervals or all training randomness.

All three n=25 selections show large train–validation separation and rising late
validation loss. All three n=500 Random runs improve near the 4,000-update limit,
so the common budget does not establish equal convergence. Gains from 100 to 500
images are consistently smaller than gains from 25 to 100.

Targeted n=25 experiments did not identify a universal replacement for the
common recipe. Stronger weight decay, fixed ImageNet encoder BatchNorm
statistics and decoder channel dropout were inconsistent or harmful. Mild
brightness/contrast helped Random but not ImageNet. A boundary-weighted BCE
experiment compared two normalizations. Image-balanced Variant B was repeated
on all three image selections: it improves Random best IoU by +0.0015 to +0.0037
and Random Boundary IoU by +0.0153 to +0.0190, but ImageNet regional-IoU effects
remain mixed (−0.0052, +0.0045 and −0.0033). Variant B is therefore not adopted
as the shared objective. All findings are provisional validation results from
one repeatedly used validation region; no locked-test result has been produced.

## Reports and reproduction

The executed notebooks form the scientific report:

- [Data and geographic split](notebooks/01_data_exploration.ipynb)
- [Training pilot](notebooks/02_training_pilot.ipynb)
- [Optimization](notebooks/03_optimization.ipynb)
- [Data efficiency across three image selections](notebooks/04_data_efficiency.ipynb)
- [Targeted experiments](notebooks/05_targeted_experiments.ipynb)

[docs/setup.md](docs/setup.md) gives exact environment, training, analysis and
notebook commands. [docs/data.md](docs/data.md) records source and label
interpretation, while [WORK_PLAN.md](WORK_PLAN.md) tracks scope and decisions.
Run artifacts retain exact configurations, compatibility identities, histories,
selected checkpoints and summaries. Analysis notebooks read those artifacts and
do not start training.

The dataset is downloaded from its provider and is not bundled here. The dataset
record lists CC BY-NC 4.0; imagery has separate source notices, which should be
reviewed before redistribution. Codex assisted with implementation, tests and
documentation; scientific choices and computed results remain explicit and
reviewable.
