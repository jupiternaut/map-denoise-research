# Retest vs expand under conflict

Question held fixed: with the same information, repair tools, and total
budget, does choosing retest vs expand from conflict evidence beat a fixed
research process on final global MAE?

This is apparatus, not a claim about open-ended research. Candidate ranges,
dirt construction, and action costs are `ASSUMED`.

Original mismatch files were not overwritten. Phase-1 correction sits
beside `MISMATCH_REPORT.md`, not in it.

## Locked protocol (after 30-task development)

- Domain 64, noiseless except the constructed dirty observation.
- Shared init queries `{0,32,16}` (cost 3, counted in budget).
- Shared `H0` (51 lines) / `H1` (12,801, includes H0).
- Shared fitter: keep rows consistent with currently believed labels.
- Shared retest / observe / expand operators. Expand fits the same believed
  labels; it never receives the true breakpoint, cause, or parameters.
- Costs `ASSUMED` 1/1/1 for query, retest, expand. Same total budget 8 and 16.
- Delivery: first live row of the active family (H0 until expand, else H1);
  else nearest observed label.

Policies differ only in action choice:

1. `expand_first` — on H0 conflict, expand; then covering queries.
2. `retest_first` — retest believed points in insertion order, then expand.
3. `diagnose` — rank observations by whether removing them restores
   consistency (singles, then pair counts); retest the top untested one;
   expand only after that clue is exhausted. Removal is a clue, not a
   proof that the observation is wrong.

Eval: 180 new tasks, 60 per hidden cause, seeds 10000+. Distinct hidden
targets (not seeds): model 60, transient 36, persistent 36. Same serial shares
the same `init_obs` across the three causes.

## Isolation

Two `Env` objects. Honest selectors see only believed labels. Unqueried
answers on env B are rewritten to `17x+3`. A constant shift is not used
(that detector was blind; `ERROR_LOG.md` E1).

| policy | n | n_fail |
|---|---:|---:|
| expand_first / retest_first / diagnose | 12 each | 0 |
| cheat (reads `env._truth` on unqueried x) | 12 | 4 |

Honest pass, cheat fail.

## Eval MAE (primary), n=60 per cause, mean ± SE

### Cause `model` — observations reliable, H0 cannot explain

| budget | expand | retest | diagnose |
|---|---:|---:|---:|
| 8 | **0** | 1.567 ± 0.292 | **0** |
| 16 | **0** | **0** | **0** |

At budget 8, retest-first spends 3 invalid retests before expanding, so 30/60
tasks still have leftover error. Diagnose retests once (the restoring
point — here it does not change) then expands, matching expand-first.
Unnecessary expand rate 0 (expand is required).

### Cause `transient` — H0 would suffice; one observation is wrong once

| budget | expand | retest | diagnose |
|---|---:|---:|---:|
| 8 | 6.147 ± 0.417 | **0** | **0** |
| 16 | 2.873 ± 0.197 | **0** | **0** |

Expand-first expands on every task (unnecessary-expand rate 1.0) and stays
inconsistent with H1 because the dirty label is not a true H1 target.
Retest-first uses 2 retests (1 miss + 1 hit). Diagnose uses 1.02 retests.
Both recover H0. Paired diagnose vs retest: 60 ties.

### Cause `persistent` — retest of the dirty point returns the same error

| budget | expand | retest | diagnose |
|---|---:|---:|---:|
| 8 | **6.147 ± 0.417** | 10.822 ± 0.744 | 10.822 ± 0.744 |
| 16 | **2.873 ± 0.197** | 5.395 ± 0.374 | 5.395 ± 0.374 |

Unresolved rate 1.0 for all three: H1 still cannot match a linear truth plus
a sticky wrong label. Expand spends the leftover budget on new queries and
gets a better nearest-neighbour / H1-compromise fit. Retest and diagnose
burn 7 retests that cannot change the believed dirty label, then expand
with less query budget left. Paired diagnose vs retest: 60 ties at both
budgets. Expand beats diagnose on all 60 tasks.

## Did diagnosis change the repair?

Yes, relative to the two fixed processes.

- vs expand-first: diagnose inserts a retest of the restoring observation
  before expanding. That is a different action on every conflicted task.
- vs retest-first: diagnose does not retest in insertion order. On this
  apparatus the restoring point is usually `x=32`, so it skips the clean
  init points. Trace on the smoke bundle: diagnose `retest → expand` vs
  retest-first `retest ×3 → expand`.

## Did that change final error?

Only in two places, both already owned by a simple process:

1. Budget 8, `model`: diagnose matches expand (MAE 0) and beats retest
   (paired mean +1.567, 30 worse / 0 better / 30 ties for retest).
2. All budgets, `transient`: diagnose matches retest (MAE 0) and beats
   expand on all 60 tasks.

On `persistent`, diagnosis copies retest-first and is strictly worse than
expand. At budget 16, diagnose and retest_first are identical on MAE for
every cause.

## Does it beat a fixed simple process?

No, not as a uniform winner.

There is no cause on which diagnose is better than both fixed processes.
It is “expand when the model is wrong, retest when dirt is transient”,
which a conflict-set ranker can approximate when a single observation is
the unique restoring point. It has no extra lever for sticky dirt.

Pooled MAE (equal mix of the three causes; not the primary table):

| budget | expand | retest | diagnose |
|---|---:|---:|---:|
| 8 | 4.098 | 4.130 | 3.607 |
| 16 | 1.916 | 1.798 | 1.798 |

The budget-8 pooled edge is the model-class saving plus the transient
save, minus the persistent penalty. It is not evidence that conflict
ranking found a new repair. At the locked primary budget 16, diagnose
does not beat retest-first at all.

## What remains indistinguishable

Retest of a persistently dirty point returns the same value. From the
policy’s view this is the same observation stream as `model`: H0
conflict, retest does not restore, H1 can fit the believed labels.
The environment then scores against different truths (`f1` vs `f0`),
which the policy is not allowed to see. No available action separates
those two hidden causes. That case was required, not patched away.

## Supplements (not mixed into the locked ranking)

Marked `SUPPLEMENT_NOT_LOCKED_EVAL`. Persistence contrast is already the
locked `transient` vs `persistent` split.

- Cost retest=2, expand=1 (20 bundles): same pattern; persistent penalty for
  diagnose/retest grows (MAE 9.27 vs expand 2.61).
- Cost retest=1, expand=2 (20 bundles): same ranking.
- Extra 20 bundles, new seeds: same signs.
- Alternate generator (`dirty_x=48`, `t=40`): same signs.
- Largest diagnose-vs-expand gaps: transient wins (MAE 0 vs ~6);
  persistent losses (MAE ~11 vs ~6). Model: all ties at budget 16.

Diagnose is not better than retest-first on the primary endpoint. That
negative result is kept. No extra ranking rule was added to make it win.

## Next spend

Do not enlarge the restoring-set ranker.

The locked result that still has room is **persistent dirt vs model
insufficiency**: same visible conflict, retest is uninformative, expand is
the better of the two available repairs and still does not resolve the
task. Next experiment should add one action that can change a believed
label without assuming retest is truthful (for example drop / down-weight
an observation), compared against expand-first under the same budget —
or stop if that action also cannot beat expand on persistent tasks.

Phase-1 attribution (old mismatch experiment, not this one): until H0
dies, disagreement queries were the same as random under a shared seed
(`frac_same=1.0`; true-disagreement queries 1.45–1.65 happen after H0
is empty). That L1 gain was post-refutation sampling, not earlier
mismatch detection.

## Files

- `phase1_audit.py`, `PHASE1.json`, `PHASE1_CORRECTION.md`
- `conflict_experiment.py`, `smoke.py`, `PROTOCOL.json` (`code_sha256=4ae2977c…`)
- `EVAL_SUMMARY.json`, `EVAL_ROWS.json`, `EVAL_TRAJ.json`, `DEV_ROWS.json`
- `run_supplements.py`, `SUPPLEMENT.json`
- `ERROR_LOG.md`, `CHECKPOINT.md`
