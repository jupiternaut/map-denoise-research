# Independent candidate provenance audit

Verified on host `liekkas`, 2026-09-26. All archived inputs were read only; no replay
reference geometry, reference distances, or outcome tables were read for this audit.

## Available immutable candidates

All 84 declared cases have finite, same-shaped `identity.ply`, `A_all.ply`, and
`B_all.ply`, with finite `[N,64]` A/B post-feature arrays. 468 relevant archived
files matched their original scene/global seals. Case counts are 12/12/20/20/20 for
scans 24/37/55/65/69. The two feature schemas are identical in names and order;
the first eight input-only columns are exactly identical per row. The remaining
proposal-specific columns must be fetched with the matching candidate key.

Development paths: `v28-surface-experts-20260922T125505Z/real_results/<case>/`
under `/home/grf/Documents/Codex/2026-09-22/`, using `features.npz`,
`FEATURE_SCHEMA.json`, `state.npz`, and `construction.json`.
Replay paths: `closeout-confirmation-20260922T133739Z/confirmation/scan<S>/<case>/`
under the same date directory, using `FEATURES.npz` and `CONSTRUCTION.json`.
The shared replay feature schema is `package/v28_closeout/FEATURE_SCHEMA.json`.

PLY files contain coordinates in original row order, not a separate point-ID
property. Construction preserves rows by `q = p + displacement * ray`, with no
resampling or filtering. The audit additionally checked every A/B displacement
against its original row's reference ray; the maximum component residual was
1.90e-13 mm. For development, archived `state.input` equals the identity PLY exactly
and archived B offsets reconstruct B within 1e-12 mm. Replay native mesh-row ID
arrays have the correct length and no duplicates. Previous paired-evidence row
IDs remain valid and are complete `arange(N)` on replay.

## What historical B actually does

The old `surfacelet.py:18-20,81-87,224-264` defines four input-derived normal
hypotheses (PCA k8/k24/k64 and camera-fronto) crossed with five overlapping
full/left/right/top/bottom 7x7 supports. Half supports contain the center row or
column, so each has 28 pixels. A uses fixed hypothesis 5 (PCA-k24/full). B searches
all 20 hypotheses over the same 49 ray offsets [-6,+6] mm at 0.25 mm spacing.
Both average the best three of the original four source correlations, requiring
at least two finite scores; ties prefer smaller displacement and then earlier
hypothesis. All-invalid anchors use zero displacement.

`run_real.py:97-123` and packaged `runtime.py:106-142` use the same at-most-1000
anchors, same input PCA-k24 interpolation field, and same reference rays for A/B.
`graph_field.py:120-133` interpolates from four nearby anchors, weighted by normal
alignment and inverse distance. It does not segment visibility/layers, and its
positive 0.01 normal-similarity floor does not guarantee separation across edges.
The packaged surfacelet code differs from the historical file only by making its
`direct_evidence` import relative; no solver change is present.

Permitted claim: B supplies multiple local orientations and one-sided support
options which may reduce patch mixing at a boundary; top-three aggregation may
tolerate one disagreeing source. This is a historical candidate capability.
Unsupported claims: explicit occlusion reasoning, visibility correctness,
layer/edge preservation, topology recovery, missing-point generation, or novelty
of B in this new experiment. The scorer checks projection, finite samples and
texture, not z-buffer visibility or correspondence ownership.

## Evidence and leakage boundaries

Every case's saved five original view IDs matches the previous reserved-evidence
lab record, including condition-invariant view order. A and B were constructed in
the same score call and therefore share exactly the same four original sources.
Reuse the previous `evidence/scan<S>/VIEWS.json` ordered reserved IDs verbatim.

The 64-column B post features are GT-free but contain optimized, in-sample costs.
`heldout_*` is a misleading name for A_all/B_all: original source slots 2/3 were
used to choose their proposals (`surfacelet.py:258-302`). They are genuinely
reserved only for the old *_fit arms. These columns do not establish independent
verification. Features also interpolate anchor costs, flags and hypothesis IDs;
they are not a direct photometric measurement of every final PLY point.

The minimum-confounding cross reuses exact sealed A/B PLYs and matching cached64
features, then scores identity versus the exact final candidate coordinates using
the shared input PCA-k24 normal, full 7x7 support, fixed reference and frozen F/R
source IDs. The previous `extract_evidence.py:86-101` provides that measurement
contract for A; apply it to B without using B's winning half-support or normal.
The shared full-patch verifier can itself be limited at boundaries, but changing
it together with B would confound this declared cross. R is disjoint only from
local construction sources; it shares reference, cameras, scene and initial MVS.

## Remaining independent audit queue

1. Verify B development row IDs, support, case weights and normalized B-gain
   target against sealed B labels; reject accidental use of A gains/features.
2. Recompute paired32 summaries from saved B score tensors and verify source
   disjointness, same ordered IDs and incumbent scoring against old A evidence.
3. Verify frozen B models use cached64_B+paired32, A policy reuse remains exact,
   thresholds/calibration use only development scenes, and all 60 replay cases
   and all declared failures remain present.
4. Recompute replay masks and exported PLYs; verify B selected rows use B, KEEP
   rows use identity, and all archived inputs still match their recorded hashes.
5. After evaluator outputs are sealed, independently recompute key native-support
   MSE, oracle and interaction summaries, and inspect the reported uncertainty
   scope. Do not use those outcomes to alter the constructor or trained policy.
