# Data readiness — metadata only

Checked on `liekkas`, 2026-09-22. No new scene image, mesh, sparse point or laser
reference content was downloaded or evaluated. `data_probe.py` reads bounded
public metadata and refuses full responses to ZIP range requests.

Locked roles: scan40 adaptation; scan55/65/69 confirmation. Scene choice uses
availability/provenance and historical exposure records, not method scores.
scan63 is omitted because a historical archive stream probe passed through it.

| Scene | Mesh bytes | Reference wire bytes | Masks/plane wire bytes | Reference uncompressed bytes |
|---|---:|---:|---:|---:|
| 40 | 35,348,253 | 65,455,003 | 1,304,250 | 118,039,091 |
| 55 | 28,104,340 | 50,318,799 | 1,067,404 | 88,009,421 |
| 65 | 38,942,933 | 52,511,307 | 849,846 | 92,130,944 |
| 69 | 32,663,480 | 48,350,769 | 773,004 | 84,916,598 |

The shared photo/calibration archive is **3,561,809,006 bytes**, named
`dtu.tar.gz`. It is a gzip stream, so individual new-scene member sizes were NOT
inferred from a nonexistent ZIP directory. An initial ZIP metadata attempt
correctly failed (`BadZipFile`, 65,580 bytes read); reading the historical
downloader established tar.gz, then a 16-byte range confirmed the gzip signature
and HTTP Content-Range size. No decompression or scene extraction occurred.

Total planned transfer with one retained image archive and selected meshes,
reference members and masks is **3,917,498,394 bytes = 3.918 GB decimal**, excluding
small metadata/local headers. Final successful metadata pass read 313,229 bytes;
the prior unsuccessful ZIP-format pass adds its recorded overhead.
Exact image/calibration member inventory and unpacked size remain pending archive
inspection. Do not claim the four scenes are already runnable.

Use `/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1/` for the
future download; this path is not yet populated by this task. Initial staging
should reserve at least 10 GiB for retained archive, selected extraction and
inputs, check member sizes before extraction, and keep a separate output budget.
The machine had about 92 GiB free on `/srv` at preparation; recheck before transfer.
Do not fetch the entire 6.3 GB Points or 6.3 GB SampleSet archives for four members.

## Provenance and limitations

- Initial meshes: public [GeoSVR model files](https://huggingface.co/Fictionary/GeoSVR/tree/main/meshes_complete/DTU).
  API sizes and LFS SHA-256 identifiers are recorded in `DATA_READINESS.json`.
- Photos/calibration: exact GeoSVR-linked historical archive URL recorded in JSON,
  shared with the old scan24/37 loader. Same nominal DTU name does not establish
  crop, resolution, camera convention or scale compatibility; scan40 must verify it.
- Evaluation: [official DTU MVS 2014 data](https://roboimagedata.compute.dtu.dk/?page_id=36).
  Selected `Points/stl/stlNNN_total.ply` plus ObsMask and Plane names/sizes/CRC are
  in JSON. CRC is metadata, not a substitute for post-download SHA-256.

Input manifests and evaluation-reference manifests must remain separate.
Do not package/rehost third-party data until their distribution terms are
checked. Public download access is not a redistribution license. The publication
should ship retrieval instructions and hashes, not silently include these archives.
