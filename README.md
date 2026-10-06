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

**Status:** RID 1.0 is checksum-verified and the active geographic split
holds out a compact 259-image southwest test area and a 289-image northern
validation area, with 1,210 training images and no positive-area cross-role
footprint overlap. The selected common recipe uses an EfficientNet-B0 U-Net for
4,000 updates, with learning rate `1e-3` through step 2,000 and `1e-4`
thereafter. Across two saved training-image selections, best validation IoU for
random/ImageNet initialization is 0.781/0.837 and 0.783/0.842 at 25 images,
0.830/0.868 and 0.831/0.863 at 100, and 0.840/0.879 and 0.842/0.877 at 500.
The paired ImageNet advantage remains positive at every size (+0.032 to +0.059).
Both n=25 repetitions show large train–validation gaps and rising late validation
loss; both n=500 random runs improve late. A targeted n=25 experiment then
increased AdamW weight decay from `1e-4` to `1e-2`. Best-IoU changes across
Seed 17/29 were +0.004/−0.005 for random initialization and −0.003/−0.006 for
ImageNet, while the train–validation gap increased in every pair. The stronger
decay is therefore not adopted. A second targeted block added whole-image
brightness and contrast factors in `[0.85, 1.15]` at n=25. Random initialization
improved by +0.0076 and +0.0074 best validation IoU across the two subset
selections; ImageNet changed by −0.0071 and −0.0007. The comparable
train–validation gap still increased in every pair, so the colour rule remains
a Random-only candidate rather than a shared recipe. These are exploratory results from one shared
validation region and two partially overlapping subset families. The locked test
area has not been evaluated.

The executed [data-exploration notebook](notebooks/01_data_exploration.ipynb)
presents the data and split evidence, and the executed
[training-pilot notebook](notebooks/02_training_pilot.ipynb) reports the six
paired runs. The executed [optimization notebook](notebooks/03_optimization.ipynb)
reports learning-rate, horizon, late-trajectory, fixed-drop and training-subset
sensitivity. The executed [data-efficiency notebook](notebooks/04_data_efficiency.ipynb)
reports the two-repetition fixed-recipe learning curves, and the executed
[targeted-experiments notebook](notebooks/05_targeted_experiments.ipynb) reports
the first regularization block. [docs/setup.md](docs/setup.md)
gives exact reproduction commands,
[docs/data.md](docs/data.md) records source and label interpretation, and
[WORK_PLAN.md](WORK_PLAN.md) tracks the research stages.

The dataset remains available from its original provider and is not bundled with
this repository. Its usage terms and attribution are separate from those of the
project code. Codex assisted with implementation, tests and documentation; the
scientific decisions and computed outputs remain explicit and reviewable.
