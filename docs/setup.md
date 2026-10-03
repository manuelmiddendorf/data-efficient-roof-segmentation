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

The notebook runner starts a fresh kernel, executes
`notebooks/01_data_exploration.ipynb`, stores its outputs, and exports
`reports/01_data_exploration.html`. No notebook cell installs software or starts model
training.

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
