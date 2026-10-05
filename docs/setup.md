# Reproducing the Stage 1 data audit

The project uses [uv](https://docs.astral.sh/uv/) and the committed `uv.lock`.
Python 3.11–3.13 is supported; Stage 1 was executed on native arm64 CPython
3.12.13 on macOS. From the repository root:

```bash
uv sync --locked --dev
sh scripts/download_rid.sh
uv run python scripts/prepare_data.py
uv run pytest
uv run python scripts/run_notebook.py
```

The download script retrieves only the files used here from the public mediaTUM
rsync endpoint: the provider checksum list and README, all supplied split lists,
1,880 roof-centred GeoTIFFs and 1,880 reviewed roof-segment masks. Raw files stay
under ignored `data/raw/rid/`. `prepare_data.py` verifies every required file
against the provider's SHA-256 list before writing the shared manifest and split
record. Once present, changed reference records are refused. Use
`uv run python scripts/verify_data.py` for a full later verification.

The notebook runner starts fresh kernels, executes the available scientific
notebooks in order, stores their outputs, and exports matching HTML reports under
`reports/`. No notebook cell installs software or starts model training.

## Stage 2 training pilot

The pilot adds PyTorch 2.8 and torchvision 0.23 from the same lockfile. Training
uses MPS when it is available and otherwise reports CPU; it never changes device
silently during a run. First run the real-data preflight, which downloads and
verifies the selected torchvision EfficientNet-B0 weights if they are not yet in
the project-local cache:

```bash
uv sync --locked --dev
uv run python scripts/check_training.py
```

The preflight checks one real 512-pixel forward/backward path, paired model and
data initialization, BatchNorm behaviour, strict checkpoint restoration, a
small no-augmentation learning test and warmed timing. It writes the compact
report `reports/training_preflight.json`. The complete six-run pilot is an
explicit long-running command. On macOS, bind `caffeinate` to that process so
the machine cannot enter idle sleep during the runs; this changes no persistent
energy setting:

```bash
caffeinate -i -m uv run python scripts/run_training_pilot.py
```

Completed runs live under
`runs/training_pilot/rid_southwest_test_north_validation_v2/`. The split version
is part of the path and compatibility identity, so historical old-split runs
cannot be reused. A compatible completed run is reused; conflicting or
incomplete directories stop with an explanation. The analysis notebook reads
the saved CSV and JSON files and never launches training.

After all six runs complete, regenerate the small versioned result tables and
figures, then execute both notebooks from fresh kernels:

```bash
uv run python scripts/analyze_training_pilot.py
uv run python scripts/run_notebook.py
```

The analysis script verifies and reads the run summaries, histories and
per-image metrics. It loads selected checkpoints only to create the documented
validation examples; test IDs are not eligible for this analysis.

## Stage 3 learning-rate block

The first optimization block reuses the verified n=100 Stage 2 runs at `3e-4`
and creates four new runs at `1e-4` and `1e-3`. Run the focused MPS smoke check,
then start the explicit training process with sleep prevention bound to its
lifetime:

```bash
uv run python scripts/check_optimization.py
caffeinate -i -m uv run python scripts/run_learning_rate_block.py
```

New runs live under
`runs/optimization/rid_southwest_test_north_validation_v2/learning_rate_n100/`.
The runner verifies the original reference configurations, input identities and
checkpoint hashes before using them. It does not copy or rewrite those runs.
Generate the compact tables, threshold diagnostic and figures before executing
the notebooks:

```bash
uv run python scripts/analyze_optimization.py
uv run python scripts/run_notebook.py
```

The threshold analysis loads only the two stored `3e-4` validation checkpoints,
evaluates the fixed 0.1–0.9 list in memory and does not retain probability maps.
No command in this block evaluates the locked test role.

## Stage 3 training-horizon block

The horizon block creates four fresh 4,000-update runs at `1e-4` and `1e-3` for
both initializations. The preflight verifies the exact split and subset, constant
optimizer rates, paired initial states, the complete evaluation schedule and the
identity of the first 2,000 sample IDs and D4 transforms with the historical
schedule. Run the block explicitly with sleep prevention:

```bash
uv run python scripts/check_training_horizon.py
caffeinate -i -m uv run python scripts/run_training_horizon_block.py
```

Completed runs live under
`runs/optimization/rid_southwest_test_north_validation_v2/training_horizon_n100/`.
They are separate from every 2,000-update run. Generate the compact result files
and figures, then execute only the changed optimization notebook from a fresh
kernel:

```bash
uv run python scripts/analyze_training_horizon.py
uv run python scripts/run_notebook.py notebooks/03_optimization.ipynb
```

The four measured runs required 92.5 minutes of optimization and 113.1 minutes
including evaluation on MPS. The analysis reads completed saved artifacts; it
does not start training, repeat the threshold diagnostic or access the locked
test role.

## Stage 3 late-trajectory and fixed-drop block

First generate the descriptive late-trajectory analysis from the four completed
constant-rate horizon runs. The two-run schedule test then uses `1e-3` through
step 2,000 and `1e-4` from step 2,001 through step 4,000 for both
initializations:

```bash
uv run python scripts/check_late_lr_drop.py
caffeinate -i -m uv run python scripts/run_late_lr_drop_block.py
uv run python scripts/analyze_late_lr_drop.py
uv run python scripts/run_notebook.py notebooks/03_optimization.ipynb
```

The preflight performs a real two-step MPS forward/backward check across the
schedule boundary, verifies every optimizer group and confirms unchanged weight
decay. The runner reuses a completed compatible run, refuses conflicting or
incomplete directories and logs the applied learning rate with every validation
row. Exactly two fresh runs belong to this block; the constant-`1e-3` horizon
runs are read-only references. Analysis reads saved validation artifacts and does
not evaluate the locked test role.

## Stage 3 Seed 29 subset replication

This block reuses the saved repetition-2 n=100 subset and changes only the
training-image selection and required run identity relative to the Seed 17
fixed-drop references. It performs no additional learning smoke test:

```bash
uv run python scripts/check_seed29_replication.py
caffeinate -i -m uv run python scripts/run_seed29_replication.py
uv run python scripts/analyze_seed29_replication.py
uv run python scripts/run_notebook.py notebooks/03_optimization.ipynb
```

The preflight verifies MPS selection, exact Seed 29 IDs, role exclusion, paired
data schedules and unchanged Seed 17 reference configurations. Exactly two new
runs belong to this block. Completed compatible runs are reused; conflicting or
incomplete directories are refused. Analysis compares only saved validation
artifacts from the two subset repetitions and does not access the test role.

## Stage 5 data-efficiency curves across two subset selections

This block applies the selected fixed-drop recipe to the nested Seed 17 subsets
at 25 and 500 images. The compatible n=100 Seed 17 runs are verified and reused:

```bash
uv run python scripts/check_data_efficiency.py
caffeinate -i -m uv run python scripts/run_data_efficiency.py
uv run python scripts/analyze_data_efficiency.py
uv run python scripts/run_notebook.py notebooks/04_data_efficiency.ipynb
```

The first preflight verifies nesting, exact subset sizes, paired schedules,
role exclusion, unique run paths and unchanged n=100 references. Exactly four
fresh Seed 17 runs belong to that block.

The second saved subset family adds four fresh Seed 29 runs at 25 and 500 images;
the completed Seed 29 n=100 pair is reused:

```bash
uv run python scripts/check_data_efficiency_seed29.py
caffeinate -i -m uv run python scripts/run_data_efficiency_seed29.py
uv run python scripts/analyze_data_efficiency.py
uv run python scripts/run_notebook.py notebooks/04_data_efficiency.ipynb
```

The Seed 29 preflight verifies both nested subset families, all eight reference
runs, paired schedules, role exclusion, fresh run paths and unchanged non-subset
seeds. The shared analysis reads twelve compatible fixed-drop runs, does not mix
in the historical 2,000-step pilot and does not access the test role.

| Reference | Origin | Compared with | Check time |
| --- | --- | --- | --- |
| RID file SHA-256 values | Provider `Checksums.sha256` | Every required raw image, mask and split file | Once before the six-run block |
| Manifest and split SHA-256 values | Bytes of the committed Stage 1 JSON records | Records named by each run configuration | Run creation and reuse |
| Selected-input identity | Provider hashes stored in the manifest, keyed by exact sample IDs | Training subset and validation role in the resolved configuration | Run creation and reuse |
| Schedule identity | Canonical hash of the precomputed ID order and D4 codes | Schedule regenerated from the saved seeds | Before model training |
| EfficientNet weight prefix | Trusted hash prefix embedded in torchvision's weight URL | Downloaded checkpoint bytes, using torchvision's checked loader | ImageNet model construction |
| EfficientNet full SHA-256 | Locally computed checkpoint identity | The cached checkpoint subsequently used by the run | Recorded at model construction |
| Checkpoint SHA-256 | Locally computed run-output identity | The selected checkpoint file | Completion and later verification |

Reference hashes establish byte identity, not label quality or geographic
generalization. Git records the committed training source and `uv.lock` records
the environment resolution.

The dataset README states CC BY-NC terms. The aerial imagery was obtained through
Google Maps Static API and has a separate provider notice restricting the stated
use to non-commercial research, education and related fair-use contexts. Review
the current source terms before redistributing imagery or derived public figures.
