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

**Status:** Four limited Stage 3 optimization blocks are complete. RID 1.0 is
checksum-verified and the active split holds out a compact 259-image southwest
test area and a 289-image northern validation area, with 1,210 training images
and no positive-area cross-role footprint overlap. In the paired Seed 17 pilot,
ImageNet initialization improved validation IoU by 0.044–0.050 across 25, 100 and
500 training images. At 100 images, `1e-3` produced the highest observed IoU for
both initializations. Extending fresh `1e-4` and `1e-3` runs to 4,000 updates
raised the random `1e-3` best IoU from 0.817 within 2,000 updates to 0.831, while
the ImageNet `1e-3` best remained 0.865 at step 2,000. A predeclared
`1e-3 → 1e-4` drop at step 2,001 then reduced late validation-IoU residual SD
from 0.0075 to 0.0012 for random initialization and from 0.0055 to 0.0010 for
ImageNet, while preserving or improving the achieved level. Repeating this
fixed-drop recipe on the saved Seed 29 n=100 subset produced best IoU 0.831 for
random and 0.863 for ImageNet, compared with 0.830 and 0.868 on Seed 17. The
paired ImageNet advantage repeated at +0.032 versus +0.038. These remain
exploratory validation results on one shared region. The locked test area has
not been evaluated.

The executed [data-exploration notebook](notebooks/01_data_exploration.ipynb)
presents the data and split evidence, and the executed
[training-pilot notebook](notebooks/02_training_pilot.ipynb) reports the six
paired runs. The executed [optimization notebook](notebooks/03_optimization.ipynb)
reports learning-rate, horizon, late-trajectory, fixed-drop and training-subset
sensitivity. [docs/setup.md](docs/setup.md)
gives exact reproduction commands,
[docs/data.md](docs/data.md) records source and label interpretation, and
[WORK_PLAN.md](WORK_PLAN.md) tracks the research stages.

The dataset remains available from its original provider and is not bundled with
this repository. Its usage terms and attribution are separate from those of the
project code. Codex assisted with implementation, tests and documentation; the
scientific decisions and computed outputs remain explicit and reviewable.
