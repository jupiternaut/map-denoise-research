# V7 finite-dictionary reassociation prototype

Run only on the verified host `liekkas`. The implementation is a conditional
operator using V4/V5 legal upstream state. It does not consume truth arrays or
call the V6 oracle. Earlier source files are imported read-only.

## API

```python
state = reassociation.freeze(xyz_world_m, scan_id, sigma_mm)
xyz_out, info, artifacts = reassociation.fit_frozen(
    state, variant="reassociate", budget=3, sharing="independent")
```

Variants are `original`, `local_multistart`, `reassociate`. Final fits are
`independent` or `shared`. `estimate` takes the three input arrays/scalars and
the same options, and includes freezing. Output preserves point order/count;
unsupported rows remain byte-identical. `fit_seconds` excludes reusable
upstream computation; `total_seconds` adds it back.

Artifacts expose `support_mask`, `active_mask`, `group_ids`, `original_group_ids`,
`responsibility`, `candidate_mask`, `entropy`, `margin`, `coefficients` and
`objective_trace`. All per-point arrays use original input order. Inactive rows
have group -1 and zero responsibility; active responsibility rows sum to one.
Support and active fit rows are deliberately different quantities.

## Implemented mathematical construction

The empirical observation mass is split among a finite dictionary with unit
responsibility row sums. There is **no prescribed component mass**, no equal
layer-sampling assumption and no balanced optimal-transport claim. Candidate
planes start as original V4 compatible groups. Each point can see candidates
originating in its cell or one Chebyshev grid ring, fixed across iterations.
The candidate remains a plane fitted from its full original source group, so
source-group spatial range and candidate eligibility are distinct.

E steps use the uniform prior over each row's candidates. M steps fit the
bias-corrected original measurements with the fixed weights times responsibility.
Both half-step free energies are recorded. The helper accepts an explicit
prior for a clone-invariance test, but production arms retain the frozen uniform
prior. Duplicating an identical candidate and then making all candidates uniform
changes the prior; this dictionary multiplicity sensitivity is a limitation.

Final action is hard MAP, with deterministic original-assignment tie retention
when available. Predictions are then refitted with independent/shared slopes.
This avoids directly averaging two candidate surfaces into their empty gap,
but does **not** prove that hard selection preserves true gaps or minimizes
pointwise squared error. Reported entropy/margin describe the soft search before
final hard refit, not calibrated correctness confidence for final geometry.

## Compute control and its limits

`local_multistart` uses no neighboring planes during association estimation.
Each cell compares K=1 WLS and K=2 input-residual quantile starts, including the
original two-node start when available. Frozen quantiles are (0.2,0.8),
(0.1,0.9), (0.35,0.65). Local EM work is distributed in deterministic rounds;
the final input BIC-like score selects K/start per cell. Local nodes subsequently
use unchanged V4 complete-link pooling, so both pipelines can pool surfaces.

This control gives each K=2 component independent slopes, unlike the older V3
shared-local-slope model; it changes local model capacity as well as computation.
It is not simply an extra-iteration replay, and neither it nor V7 is an external
academic strong baseline. Every selected K, start, iteration, rank and group
membership is recorded.

The target work is the planned reassociation pair-evaluation plus fitted-row
count. Initialization and final local sufficient fits can exceed it. Underflow
or a numerical step rejection can also lower reassociation's actual count.
Compare **actual counters and wall time** across arms: `compute_target_met` alone
does not certify paired equal cost. Dense final least-squares design sizes and
actual fit times are separately exposed. Pooling compatibility/matrix work is
not fully represented by the pair/row proxy, but is included in wall time.

## Validation

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/algorithm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -m unittest -v test_reassociation.py
```

These are implementation checks, not research-performance evidence. The parent
run script handles the three public smoke cases and complete 24-case evaluation;
this module has not independently duplicated those runs.
