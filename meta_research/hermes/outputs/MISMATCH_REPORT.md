# Mismatch diagnosis by query selection

Question held fixed: with the same libraries, same fitter, same initial observations, does disagreement-driven querying find model mismatch faster than covering or random sampling?

This is not a claim about open-ended science. H0/H1 are apparatus.

## Locked after 30-task development

- Domain: 64 positions, noiseless integer labels.
- H0: 51 lines `ax+b`, `a∈{-1,0,1}`, `b∈[-8,8]`.
- H1: H0 plus 12,750 one-break piecewise lines, `t∈{16,24,32,40,48}`.
- Shared initial queries: covering prefix `{0,32}` (counted in budget).
- Shared update: `Cj = {h∈Hj : h agrees on all queried points}`.
- Shared delivery: first remaining H0, else first remaining H1, else nearest observed label.
- Disagreement: among live C0 if nonempty else live C1, query unqueried `x` with most distinct predictions; ties → smallest `x`; zero disagreement → random.
- Random: 3 seeds per task, not counted as extra tasks.
- Eval: 180 new tasks, 60 per type, seeds 10000+.

## Isolation (must-pass)

36 trials (3 policies × 3 types × 4 serials): keep queried labels and RNG fixed, add 13 to every unqueried answer. Next action unchanged. `n_fail=0`.

## Eval @ budgets 4 / 8 / 12 / 16

Discovery:

- type 0: C0 still nonempty (did not over-refute).
- type 1: C0 empty and C1 nonempty.
- type 2: C1 empty.

Means ± SE, n=60. Random is the within-task mean of 3 repeats.

### Type 0 — H0 already enough

All three policies: discovery 1.00, L1=0 at every checkpoint. No unnecessary jump to H1 in delivery (H0 preferred while alive). Disagreement does not complexify here.

### Type 1 — H0 wrong, H1 enough

| budget | disc fixed | disc random | disc disagree | L1 fixed | L1 random | L1 disagree | cost-to-refute H0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.000 | 0.850 | 0.983 | 151.7 | 303.7 | **8.3** | D 2.48 / F 2.88 / R 2.99 |
| 8 | 1.000 | 0.994 | 1.000 | 0 | 123.0 | 0 | same |
| 12 | 1.000 | 1.000 | 1.000 | 0 | 101.9 | 0 | same |
| 16 | 1.000 | 1.000 | 1.000 | 0 | 5.7 | 0 | same |

Paired discovery D−F at 4: −0.017 (1 task, D slower than covering). D−R at 4: **+0.133** (26 vs 1).

So: disagreement is **not** a better mismatch detector than covering. It is a better *sampler for remaining H1* once H0 is almost gone, which turns into large L1 gains at budget 4, and it refutes H0 about 0.4 queries earlier.

### Type 2 — even H1 is wrong

| budget | disc fixed | disc random | disc disagree | L1 fixed | L1 random | L1 disagree |
|---|---:|---:|---:|---:|---:|---:|
| 4 | **0.583** | 0.483 | **0.300** | **68.2** | 108.8 | 117.3 |
| 8 | 1.000 | 0.794 | 0.900 | **25.3** | 57.5 | 37.7 |
| 12 | 1.000 | 0.861 | 0.917 | **19.3** | 39.9 | 32.3 |
| 16 | 1.000 | 0.950 | 1.000 | **12.1** | 26.0 | 20.1 |

Paired discovery D−F at 4: **−0.283** (9 better, 26 worse). Covering finds library mismatch earlier because extra breaks sit at unsampled regions; disagreement spends queries distinguishing still-alive H1 members.

## Decision (predeclared)

- Disagreement is **not** stably better than covering at mismatch discovery. Stop promoting it as a research-diagnosis policy.
- Type-1 L1 drop at budget 4 is **active sampling inside an adequate class**, not evidence that the policy discovered a new representation.
- Type-2 is the actual “current explanations are exhausted” case, and covering wins.
- No perfect external validator was used in the ranking.

Next work, if any: a *repair* action after `C1=∅` (new family), compared under the same query rules. Do not enlarge disagreement selection itself.

## Files

- `outputs/mismatch_experiment.py`
- `outputs/MISMATCH_RESULTS.json` (isolation, dev, eval, timing 10.9 s)
