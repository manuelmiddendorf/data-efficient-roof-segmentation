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

**Status:** planning documents are prepared. Dataset inspection, implementation
and experiments have not started; no results are claimed.

See [WORK_PLAN.md](WORK_PLAN.md) for the research stages and proposed experiments,
and [AGENTS.md](AGENTS.md) for development conventions. Setup instructions and
scientific notebooks will be added with the corresponding implementation.

The dataset remains available from its original provider and is not bundled with
this repository. Its usage terms and attribution are separate from those of the
project code. Codex assists with project planning and implementation; its actual
contributions will be documented as the work progresses.
