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
checkpoint selection and qualitative inspection. Three active test images (146,
1762 and 1782) were training samples in a discarded pilot under the old split.
This historical use is disclosed but did not determine the new boundary. Every
model under the active design starts from fresh random parameters or the original
verified ImageNet encoder weights and a newly initialized decoder.

RID data and code have separate licenses. The dataset README states CC BY-NC;
the upstream code declares GPLv3. This project reimplements small data-contract
and spatial-audit operations and does not copy the upstream training pipeline.
