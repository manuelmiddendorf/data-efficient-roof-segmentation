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

The project adopts provider split D1 (the northern validation region) and the
provider's shared 154-image test set. Complete georeferenced image footprints are
projected to EPSG:25832. Any positive cross-role intersection larger than
0.01 m² causes the development image to be excluded; test IDs remain unchanged.
Exact byte-identical copies are also reduced to one retained development sample.
The resulting exact IDs, exclusions, reasons, nested training subsets and seeds
are in `data/metadata/splits.json`.

The locked test region is excluded from subset selection and qualitative image
inspection. Stage 1 checks its file and spatial contracts because those checks
are necessary to establish the split; it does not use test appearance or labels
to choose a model or training protocol.

RID data and code have separate licenses. The dataset README states CC BY-NC;
the upstream code declares GPLv3. This project reimplements small data-contract
and spatial-audit operations and does not copy the upstream training pipeline.
