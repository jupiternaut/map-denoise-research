# Reproduction and deployment scope

Host: liekkas. Workspace: `/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z`.
Read-only origin: `/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z`.
Interpreter used: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`, Python 3.12.13.
No installation, environment modification, GPU execution or new data download.

## Single deployment configuration

The frozen main arm is post_A_keep. post_AB_keep is secondary; pre_A_keep is an
archived mechanism control. Three HistGradientBoostingRegressor models were fit
once each with unchanged V28 hyperparameters to 79,594 archived rows: 39,996 from
scene 24 and 39,598 from scene 37, spanning all 24 existing ROI/condition cases.
The feature dimensions are 8 (pre_A), 64 (post_A) and 64 (post_B).
Fitting took 2.7053 seconds in a one-thread CPU context; this is fitting time only,
not end-to-end inference, image decoding, feature construction or benchmark cost.

Only saved training-label row IDs/gains and saved features were opened for the
fit. No GT was reopened. Native and +/-3 mm rows retain their original labels,
sampling and order within each scene; scenes are concatenated 24 then 37.
TRAINING_LOCK.json was written before fitting and fixes every training sample
array hash, full input-file hashes, source/schema hashes, dependency versions,
parameters and the primary decision rule. Model joblib and JSON specifications
have additional hashes in package/v28_closeout/models/MODEL_LOCK.json.

This fit is prepared for future confirmation. It cannot inherit V28 fold scores,
and there is no independent performance claim for the combined-scene models.

## Commands actually used

From the workspace above:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B prepare_release.py \
  --v28 /home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  V28_SOURCE=/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B \
  -m unittest discover -s tests -v
```

The prepare command refuses to overwrite an existing lock/model directory.
For a separate reproduction, copy the package sources, preparation script and
tests into a new empty destination and omit existing models/TRAINING_LOCK.json.
The destination is derived from the preparation script, while origin is an
explicit argument; no archive path is used by the deployable package.

## Checks and remaining boundary

Tests verify strict zero/tie behavior; model hash/load and deterministic routing;
count-matched random control; exact KEEP coordinates; invalid feature rejection;
array/camera construction without a scene loader; and no old-workspace imports.
Archive compatibility checks replay all 24 old fold predictions and decisions,
including seeded random decisions, with their archived fold models. Two cases
also replay cached photometric evidence to all four raw offsets and post features.
All seven checks passed in 11.785 seconds. The archived fold replay covered
1,181,301 point rows with bitwise-identical predictions and decisions. See
`tests/RESULTS.json`. These checks validate packaging compatibility, not
new-scene effectiveness.

`package/README.md` specifies the complete array/camera contract, feature order,
preprocessing, units and view ordering. New-scene dataset loading, coordinate
scale, ROI definition and metadata-based source selection remain adapter work.
No wheel install or clean-environment installation was exercised; the packaged
API was imported directly with the existing pinned environment.
