# V6 A: input-only direction replacement, frozen before formal execution

Host: liekkas. Date: 2026-09-12. No GPU, download, new dependency, partition
search, association repair, or edit to V3/V4/V5 is part of this experiment.

## Fixed construction

Compare V4 compatible with its original within-scan difference direction
(`difference`) against the smallest covariance eigenvector of the CURRENT
observed XYZ (`pooled_pca`). Both use the unchanged V4 budgets and thresholds.
Load a private V5 v4_compatible instrumentation instance, its private V4, and
its private V3 for each call. Only replace the PCA instance's `_basis` helper.
V5 instrumentation returns the exact V4 output and reconstructs its actual
support mask; its new parameter-sharing predictions are not used.

The replacement acts before graph bias fitting. Bias, local basis, cells,
associations, weights, grouping and support are all recomputed in that direction.
They are not frozen to the old direction. No projected output is reused as data.
Instrumentation and normal preparation count toward both arms' total cost.

PCA uses the centered current point covariance in mm, with its smallest
eigenvector as normal and largest eigenvector as first tangent. Set the largest
absolute component of each of these vectors positive; the second tangent is
normal cross first tangent, giving a right-handed orthonormal basis. Record the
spectrum and basis. Fewer than three points or numerical rank below two is
unsupported. Repeated eigenvalues have no unique physical direction; no new
quality threshold, normal selector or outcome-dependent fallback is introduced.

## Fixed inputs and estimator boundary

Real: copy the same 24 input/evaluation files byte-for-byte from
`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-transfer-v5-zrtui_f5`.
Six existing cy/da junction/thin/wall patches, each zero/normal_translation/
tangent_translation/small_rotation. Both arms receive only current XYZ in world
metres, scan IDs, and sigma=2 mm. Retain original point source IDs and order.
All reference measured clouds, original PCA injection axes, and perturbations
are evaluation-only. The original measured cloud is not clean truth.

Predeclared public synthetic diagnostic: the 24 full identifiable cases in
`exploration-v5-development-vdbkdouk`, seeds 912101/912113/912127, gap 0/2/4/8,
bias RMS 0/4 mm, sigma=1 mm. Reuse saved input bytes and the existing evaluator
payload, never regenerate a seed. No ambiguity, stress, or confirmation cases
are selected. These are exposed development diagnostics, not new confirmation.

Expected formal outputs: 48 real + 48 synthetic = 96. One warmup per arm on the
first legal real zero input, recorded separately. No quality-driven retries.
Every run creates a unique `direction-v6-` directory; keep failures and outputs.

## Measurements and verification

Save same-order XYZ, scan/source IDs, actual support mask and distinct moved mask,
candidate normal/basis trace, upstream fingerprints, input/source/output hashes,
per-method wall time, process RSS, and all estimator metadata. No diagnostic
changes output or gates acceptance.

Real: use each method's own zero output for self response; report XYZ/injection
normal/tangent recovery, zero edit, input edit, and support over ALL points.
Record candidate versus current PCA and injection-axis angles. The general
single-axis XYZ recovery lower bound is RMS of the injection perpendicular to
the candidate axis. With the actual support mask, unsupported points retain the
whole injected displacement in the bound. For zero or missing normal and no
support the bound is the input intervention RMS. This pointwise projection is
valid also for rotation; only a parallel translation can simplify to
3.5*sin(theta). It is not a bound on the normal-only score, surface accuracy,
self response, or real denoising error. Do not apply it to GICP.

Synthetic: save existing finite-surface accuracy/coverage, corresponding-point
and layer-gap diagnostics. GT is used only after estimation. Report paired
regressions, especially thin layers, rather than promoting normal recovery alone.

Before the formal run, tests cover exact old-arm replay, read-only legal inputs,
private upstream replacement, basis consistency, coordinate/point-order behavior,
normal-only changes, and absence of evaluation inputs. Afterward independently
reload all arrays, verify IDs/masks/hashes, recompute real decompositions and
pointwise bounds, synthetic finite-surface scores, and aggregates. Original
algorithms and both source runs must hash identically before/after. One CPU run
under possible concurrent contention is not a controlled speed comparison.

The injection axis was constructed from the original measured PCA. Therefore a
PCA-arm recovery improvement supports an allowed-direction limitation, not proof
that PCA is closer to physical surface normals or improves true geometry.
