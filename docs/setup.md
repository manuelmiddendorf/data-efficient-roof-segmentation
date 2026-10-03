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

The dataset README states CC BY-NC terms. The aerial imagery was obtained through
Google Maps Static API and has a separate provider notice restricting the stated
use to non-commercial research, education and related fair-use contexts. Review
the current source terms before redistributing imagery or derived public figures.
