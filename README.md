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

A bounded n=500 budget extension trained Random and ImageNet models for 8,000
updates on subset seeds 17 and 29. Random endpoint IoU improved by +0.0121 and
+0.0079 relative to each new run's 4,000-update prefix; ImageNet changed by
+0.0004 and −0.0005. The paired endpoint advantage therefore narrowed from
+0.0400 to +0.0284 and from +0.0322 to +0.0238. Final-five evaluation means
show the same pattern. Pretraining remains advantageous under both observed
budgets, while Random was more sensitive to the available optimization time.
This does not establish convergence or an initialization-independent optimum.

At 100 images, three fixed uniform-random subsets were compared with the three
spatial-maximin references under the same training recipe. Spatial selection
covered about 0.258 km² with no material footprint overlap, versus 0.220–0.227
km² and 12.1–14.5% overlap for random selection. Spatial-minus-random best-IoU
effects were consistently positive for Random initialization (+0.0059, +0.0122
and +0.0026; mean +0.0069) but mixed for ImageNet (+0.0047, −0.0038 and
−0.0010; mean −0.0001). This descriptive comparison does not isolate footprint
coverage from the changed roofs, backgrounds and image content.

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

The prepared 0.3 m WHU Aerial raster release contains 8,188 paired 512 × 512
RGB tiles and binary building-footprint masks in provider training, validation
and test roles. Nested n=25/100/500 learning curves were measured for three saved
uniform-random training selections. ImageNet improved selected validation IoU
in all nine pairs, by +0.0657 to +0.0983. Every n=25 model finished below its
selected maximum, whereas all six n=500 late IoU trends remained positive. The
100-to-500 gain varied substantially across selections: +0.0065 to +0.0562 for
Random and +0.0114 to +0.0677 for ImageNet. More data improved both positive-
building overlap and false-positive control on empty references. These runs vary
training-image selection while holding the other random seeds fixed and share
one provider validation role. They are repeated validation evidence rather than
independent test or geographic-generalization results. The ungeoreferenced
prepared TIFFs do not permit independent reconstruction of every tile footprint.
WHU test images were not used.

## Reports and reproduction

The executed notebooks form the scientific report:

- [Data and geographic split](notebooks/01_data_exploration.ipynb)
- [Training pilot](notebooks/02_training_pilot.ipynb)
- [Optimization](notebooks/03_optimization.ipynb)
- [Data efficiency across three image selections](notebooks/04_data_efficiency.ipynb)
- [Targeted experiments](notebooks/05_targeted_experiments.ipynb)
- [WHU external-replication data](notebooks/06_external_replication_data.ipynb)
- [WHU data efficiency and initialization](notebooks/07_external_replication_pilot.ipynb)

[docs/setup.md](docs/setup.md) gives exact environment, training, analysis and
notebook commands. [docs/data.md](docs/data.md) records source and label
interpretation, while [WORK_PLAN.md](WORK_PLAN.md) tracks scope and decisions.
Run artifacts retain exact configurations, compatibility identities, histories,
selected checkpoints and summaries. Analysis notebooks read those artifacts and
do not start training.

Raw datasets are downloaded from their providers and are not bundled here. The
RID record lists CC BY-NC 4.0 and a separate imagery notice. The WHU page and
prepared archive state no explicit reuse licence, so permission and underlying
LINZ imagery terms require review before redistribution. Codex assisted with
implementation, tests and documentation; scientific choices and computed
results remain explicit and reviewable.
