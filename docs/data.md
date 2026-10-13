# RID data record and label interpretation

This project uses version 1.0 of the **Roof Information Dataset (RID)** by Krapf
et al., issued on 11 May 2022, DOI
[`10.14459/2022mp1655470`](https://doi.org/10.14459/2022mp1655470). The release is
downloaded from the provider and never committed. The exact public transport,
provider checksum-list hash, file-level checksums and upstream code revision are
recorded in `data/metadata/data_manifest.json`.

The selected product contains 1,880 paired 512 × 512 RGB GeoTIFFs and reviewed
roof-segment masks. All images declare EPSG:4326; no image has an alpha band,
nodata declaration or invalid dataset-mask pixel. Masks are 8-bit single-channel
PNGs. The official mask-generation code establishes the ordering that the
release README describes only tersely: codes 0–15 are azimuth bins, code 16 is
flat roof, and code 17 is background. Binary roof targets are therefore
`mask != 17`. There is no unknown code in this product. This code coverage does
not establish that annotations are complete or correct.

The active split retains the provider-D1 northern area as validation and holds
out one compact southwest area as test. The approved metric bounds are easting
720,200–720,800 m and northing 5,364,860–5,365,350 m in EPSG:25832. Only complete
image footprints inside this window enter the test role; boundary-straddling
images are excluded. Positive cross-role intersections larger than 0.01 m² are
removed, and exact byte-identical copies are reduced to one retained sample. No
additional distance buffer is applied. This leaves 1,210 training, 289
validation, 259 test and 122 excluded images. Exact IDs, reasons and newly
generated nested subsets are in `data/metadata/splits.json`.

The active test region is excluded from subset selection, preflight learning,
checkpoint selection and qualitative inspection under the restarted design.
Three active test images (146, 1762 and 1782) were training samples in a discarded
pilot under the old split. The pre-test metadata review additionally identifies
positive-area overlap with test IDs 1402 and 1404. The five images are excluded
from the fixed 254-image primary test population; all 259 remain a declared
full-region sensitivity analysis. This does not change the saved split or erase
the historical use, which did not determine the new geographic boundary.
See the [final evaluation protocol](final_evaluation_protocol.md) and
[eligibility CSV](../reports/final_evaluation_rid_test_ids.csv) for the rule and
exact IDs. Every model under the active design starts from fresh random
parameters or original ImageNet weights and a newly initialized decoder.

RID data and code have separate licenses. The dataset README states CC BY-NC;
the upstream code declares GPLv3. This project reimplements small data-contract
and spatial-audit operations and does not copy the upstream training pipeline.

## WHU Aerial external-replication data

The external data foundation uses only the provider's prepared **0.3 m Aerial
raster** archive from the [WHU Building Dataset page](https://gpcv.whu.edu.cn/data/building_dataset.html),
not the 0.2 m COCO product, satellite subsets or original full-resolution aerial
imagery. The archive is stored unchanged under `data/raw/whu/`; its locally
computed SHA-256 in `data/metadata/whu/data_manifest.json` identifies the bytes
but is not a provider-verified checksum.

The downloaded directories contain 4,736 training, 1,036 validation and 2,416
test pairs: 8,188 in total. This matches the tile-grid arithmetic on the source
page, although its descriptive text says 8,189. Images are 512 × 512 three-band
`uint8` TIFFs. Labels are 512 × 512 single-band `uint8` TIFFs with observed
values 0 and 1. Direct development-pair inspection shows that 0 denotes a
building footprint and 1 denotes background, so the binary target is
`mask == 0`. There is no alpha band,
explicit invalid label, nodata declaration or invalid dataset-mask pixel.

Training filenames are numeric, validation names use `val_`, and the two test
blocks use numeric and `2_` prefixes respectively. Pairing preserves these names
inside role-qualified IDs, so overlapping numeric stems across roles remain
unambiguous.

The audit finds 419 empty training masks, 299 empty validation masks and 685
empty test masks. Their identical all-background arrays form the only exact
mask-duplicate group. No RGB image and no complete image–mask pair is exactly
duplicated, including across provider roles. These byte checks do not detect
shifted, resampled or otherwise near-duplicate views.

WHU labels rasterized building footprints from edited Christchurch building
vectors. RID instead supplies roof-centred samples whose classes describe
visible roof segments. Both can support binary segmentation, but annotation
scope, sampling unit, city, imagery and spatial metadata differ. WHU results
therefore provide an external replication of the initialization comparison,
not a like-for-like estimate of the RID target.

The source page documents separate training, evaluation and two test areas with
tile-block dimensions 64 × 74, 14 × 74, and 17 × 42 plus 23 × 74. The prepared
TIFFs themselves contain no CRS, physical affine transform, GCPs or coordinate
table. Exact cross-role file duplicates can be tested, but individual ground
footprints and visible-area overlap cannot be reconstructed independently.
Numeric filenames are retained as identifiers and are not treated as coordinates.

The GPCV data index describes its datasets as freely downloadable, but neither
the dataset page nor this archive states an explicit reuse licence. Cite Ji,
Wei and Lu (2019), DOI `10.1109/TGRS.2018.2858817`, acknowledge Wuhan
University's GPCV Lab, identify New Zealand LINZ as the original imagery source,
and confirm permissions before redistributing the imagery or derived examples.
