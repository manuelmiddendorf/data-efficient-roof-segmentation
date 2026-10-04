# Data-Efficient Roof Segmentation

*How many labelled images do we need, and how much does pretraining help?*

This project will study binary roof segmentation from aerial imagery under
limited annotation budgets. It will compare an EfficientNet-B0 U-Net initialized
randomly or with an ImageNet-pretrained encoder, followed by targeted experiments
on optimization, augmentation and regularization.

The planned dataset is the [Roof Information Dataset (RID)](https://mediatum.ub.tum.de/1655470).
Training, validation and test regions will be separated geographically, accounting
for overlapping image footprints. Learning curves will describe performance as
the number of labelled training images increases, alongside variation across
repetitions, computational cost and qualitative error analysis.

**Status:** Stage 2 is complete. RID 1.0 is checksum-verified and the active
split holds out a compact 259-image southwest test area and a 289-image northern
validation area, with 1,210 training images and no positive-area cross-role
footprint overlap. An early pilot under the former provider-based test design was
discarded before this split was adopted; all six reported Stage 2 models start
fresh. In the paired Seed 17 pilot, ImageNet initialization improved validation
IoU by 0.044–0.050 across 25, 100 and 500 training images. This is exploratory
validation evidence from one repetition; the locked test area has not been
evaluated.

The executed [data-exploration notebook](notebooks/01_data_exploration.ipynb)
presents the data and split evidence, and the executed
[training-pilot notebook](notebooks/02_training_pilot.ipynb) reports the six
paired runs. [docs/setup.md](docs/setup.md) gives exact reproduction commands,
[docs/data.md](docs/data.md) records source and label interpretation, and
[WORK_PLAN.md](WORK_PLAN.md) tracks the research stages.

The dataset remains available from its original provider and is not bundled with
this repository. Its usage terms and attribution are separate from those of the
project code. Codex assisted with implementation, tests and documentation; the
scientific decisions and computed outputs remain explicit and reviewable.
