# View provenance audit

Audited on host `liekkas`. Prior artifacts and datasets were read-only. This audit
read source code, construction metadata and image filename inventories; it did
not open evaluator reference geometry or archived evaluation results.

## What was already used

The fixed candidate is `A_all`, not `A_fit`. In the archived V28 `surfacelet.py`,
line 258 uses source indices `(0,1,2,3)` for the `all` branches and `(0,1)` for the
`fit` branches. Lines 299–302 nevertheless use the names `heldout_incumbent_*` and
`heldout_proposal_*` for indices 2/3 in every branch. These columns are held out
only for `A_fit`/`B_fit`, never for `A_all`/`B_all`.

The archived `run_real.py:116` and each `FEATURE_SCHEMA.json:12` explicitly retain
this distinction. The preceding relative-gain laboratory `prepare_data.py:69–72`
loads `A_all_post` as its 64-dimensional input. Its zero-based columns 30/31 are
the incumbent costs from original sources 2/3, and columns 34/35 are their proposal
costs. Renaming or recombining those columns would not introduce held-out images.

Source roots:

- `/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z`
- `/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z`
- `/home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z`

## Exact image exclusions

Every ROI uses reference image `0022.png`. In the table, integer IDs mean their
four-digit `.png` filenames. Exclude the reference plus all four listed source
IDs, not just the two columns historically called held out.

| ROI | Original source IDs, ordered |
|---|---|
| scan24_gable_center, scan24_turret_join | 48, 47, 46, 45 |
| scan24_wall_control | 38, 37, 36, 35 |
| scan24_window_left | 42, 41, 40, 39 |
| scan37_clamp_jaw | 48, 47, 46, 45 |
| scan37_driver_handle | 46, 45, 44, 43 |
| scan37_scissor_cross | 18, 8, 7, 6 |
| scan37_stone_control | 41, 40, 39, 38 |
| scan55_roi0, scan55_roi2, scan55_roi3 | 48, 47, 46, 45 |
| scan55_roi1 | 21, 20, 19, 18 |
| scan65_roi0, scan65_roi1, scan65_roi3 | 48, 47, 46, 45 |
| scan65_roi2 | 40, 39, 38, 37 |
| scan69_roi0 | 45, 44, 43, 42 |
| scan69_roi1, scan69_roi3 | 48, 47, 46, 45 |
| scan69_roi2 | 44, 43, 42, 41 |

All condition variants of an ROI have the same archived view list. There are 24
development cases over 8 ROIs and 60 relevant replay cases over 12 ROIs. The extra
scan40 adaptation archive was inventoried separately (20 cases, 4 ROIs) and is
not part of this experiment's development or replay queue.

All five relevant scene image folders contain exactly `0000.png`–`0048.png`, 49
files. Thus 44 images per ROI are disjoint from its five original correction
images before applying any input-only support requirement. This does not say all
44 provide usable evidence.

Image folders:

- scan24/37: `/srv/slam-research/grf/map-denoise/datasets/loss-alignment-v23/scan{sid}/image`
- scan55/65/69: `/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1/inputs/scan{sid}/images`

## Meaning and limits of the comparison

New images can be correction-source-disjoint. They are not statistically
independent observations: the reference photograph, scene, calibration, initial
mesh and mesh-derived normals remain shared. No assertion was established that
these new source images were held out from the initial MVS reconstruction.

The old post features were scored on anchors and then interpolated to output
points (`run_real.py:118–123`; packaged `runtime.py:137–142`). Directly rescoring
the final saved incumbent/A pair also changes the measurement location. The new
protocol's matched old-source direct-scoring arm is therefore necessary to
attribute a difference to new view information. Both cohorts must use the same
point pair, normal, reference patch and photometric implementation.

Positive projection depth and an in-image footprint do not establish visibility
of a surface. A finite photometric patch can belong to an occluder or a different
physical layer. Missing-data flags prevent unsupported patches from posing as
good evidence, but cannot certify correspondence or true geometric visibility.

## Executable evidence audit

`audit_evidence.py` completed successfully and wrote `AUDIT_EVIDENCE.json`.
It verified all five evidence seals, independently recomputed the native-input
view ranking for all 20 ROIs, checked disjointness and original image hashes, and
checked exact archived row order across all 84 cases. Independent recomputation
of all 32 paired features from raw scores had maximum absolute discrepancy
`5.960464477539063e-08`, within float32 rounding tolerance. The maximum A ray
reconstruction residual was `2.842170943040401e-14 mm`.

For the lexicographically first case in each of five scenes, the audit selected
nine evenly spaced rows, recomputed full-input k24 normals, and reloaded original
pixels to recompute eight-source, two-position scores. Missingness masks matched
exactly and the maximum score discrepancy was zero. This is a 45-row direct-pixel
replay check, not a claim to have rerun raw photometry on every row. Full-array
summary, row-order, source-disjointness and ray-position checks covered every case.

Training/selection and evaluator audits remain separate. The evidence audit did
not open evaluator reference geometry or replay geometric-error arrays.
