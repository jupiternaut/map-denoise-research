# V5 real-input transfer: frozen V4 interventions

Freeze this protocol and all invoked modules before the formal run. No new
intervention, input repair, download, exact-partition search, GPU or multi-host
execution is part of this experiment.

## Exact targets

- Host: liekkas.
- Original run (read-only):
  `/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-components-v4-h3dnauhf`.
- Reuse its exact 24 input NPZ files: cy/da junction/thin/wall, each zero,
  normal_translation, tangent_translation and small_rotation. Copy input and
  evaluator files byte-for-byte into a unique `real-transfer-v5-` run directory.
- All methods receive only the same current XYZ in world metres, integer scan ID,
  and sigma=2 mm. Source point IDs are preserved by the runner, not used to infer
  truth. The existing three nonzero interventions have 3.5 mm all-point XYZ RMS.
- References, injected displacement, original PCA construction axes and source
  poses are evaluator-only. In the zero arm the original measurement is itself
  the legal current input; no extra reference is supplied to the estimator.

## Fixed eight pipelines

1. identity.
2. pool_independent: unchanged V4 surface_pooling independent arm.
3. pool_compatible: V5 slope_pooling v4_compatible exact-output reproduction,
   including its instrumentation cost and recorded support mask.
4. shared_group_slope: new shared-slope arm, fixed upstream state.
5. node_intercepts: shared-slope/local-node-intercept control.
6. graph_shared: unchanged V3 graph estimator.
7. measure_unbalanced: unchanged V3 three-outer-step raw source correction.
8. official_gicp: frozen official Open3D GICP pairwise-to-largest-input-anchor
   wrapper, without a point-level postfilter. It uses only an input-defined
   common translation gauge on the linked component; no reference alignment.

The transport and GICP references only correct scan motion; pooling and graph
also change point-level geometry. Do not treat their functional scopes as equal.
All invoked algorithms are rerun on these inputs for comparable current cost.
One warmup on the first legal zero input per pipeline is recorded separately.
An exception stays in the denominator and its artifacts are retained; no overwrite,
GT-directed fallback or automatic parameter retry is allowed.

## Outputs and diagnostics

Save world-coordinate output, scan IDs, source point IDs, actual moved-point mask,
and true reported applicability mask where the API exposes one. A movement mask
is NOT an acceptance mask. Older V4 independent/V3 graph report only a support
fraction: retain that value and mark their per-point acceptance mask unavailable.
New slope variants expose their actual unsupported row indices and mask digest;
transport/GICP expose whole-scan applicability. Report all points, not just support.

Reuse V4 recovery XYZ/construction-normal/tangent, zero-input edit, own zero-output
response and input-edit metrics in absolute world coordinates. The measured
reference is NOT clean or independent geometric truth. Low edit or low response
alone is not proof of successful denoising; constant/identity outputs can be stable.

After estimation, compare each reported candidate normal to CURRENT input PCA
and to the frozen construction axis. Never reset a method's normal using either
reference axis. Report candidate-axis changes from its own zero run. The existing
best per-scan constant-axis correction of the known injection is evaluator-only;
it is not a bound on full pooling/GICP pipelines, which permit richer edits.

Keep raw GICP stage transforms, input origin and common gauge from its metadata.
No scan/layer/reference alignment is applied during scoring. Input and estimator
source hashes are captured before/after; complete old V4 run remains unchanged.

## Verification and interpretation

Require 24 distinct inputs and 192 case/method outcomes from six patches, two
scenes, not 24 independent scenes. Independently reload all saved arrays, check
point IDs, input/output hashes, unchanged unsupported points when masks are known,
and recompute recovery/decomposition/response scores without calling estimators.
Record full method wall time, import/warmup time, total run time, and process peak
RSS with their scope; one single-thread CPU run under possible task contention
does not establish a controlled speed ranking.

Report new slope versus compatible and node-intercept control per mode and patch,
including regressions, zero edits and axis disagreements. A result closer to the
original measurement supports intervention recovery only, not real denoising
accuracy, thin-layer truth, general SLAM quality or a new independent scene gain.
