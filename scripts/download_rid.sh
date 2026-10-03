#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
destination="$project_root/data/raw/rid"
mkdir -p "$destination"

export RSYNC_PASSWORD=m1655470
server="rsync://m1655470@dataserv.ub.tum.de/m1655470"
rsync -rt "$server/Checksums.sha256" "$destination/"
rsync -rt "$server/RID_dataset/README_data.md" "$destination/RID_dataset/"
rsync -rt "$server/RID_dataset/filenames_train_val_test_split" "$destination/RID_dataset/"
rsync -rt "$server/RID_dataset/images_roof_centered_geotiff" "$destination/RID_dataset/"
rsync -rt "$server/RID_dataset/masks_segments_reviewed" "$destination/RID_dataset/"

echo "RID files downloaded. Run: uv run python scripts/prepare_data.py"
