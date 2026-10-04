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
