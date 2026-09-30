# Competing explanations: fitter first, then query choice

Previous checkpoint `20260914T051316Z` is read-only. Claims corrected in
`CORRECTION.md`: same-point retest does not split `model` vs `persistent`;
a later query can; expand’s MAE drop was a better nearest-neighbour
fallback, not a diagnosis.

Question this round: when feedback may be wrong, keep both “model is
insufficient” and “an observation is dirty”, allow a reversible isolate
in the fitter, then ask whether choosing queries that split remaining
repairs beats a fixed process that uses the **same** fitter.

48 is not a policy constant. dirty_x and t are sampled; init `{0,16,32}`
is rejected (eval: shortcut fraction 0, 30 distinct dirty_x, 5 distinct t).

## Contracts (TLC)

`RepairContract.tla`: append-only records, billed query/retest, free
reversible isolate, `PolicyPick` reads `rec` only. TLC: 139 generated,
51 distinct, 0 on queue, depth 11, no error. Not a model of the numeric
fitter.

Isolation (two `Env` objects, unqueried labels rewritten): honest 27/27
pass; cheat 4/9 fail.

## Experiment A — freeze the history, change only the fitter

Old: every believed label is a hard constraint; else nearest neighbour.
New: enumerate H0/H1 × isolate-cap-1 on the trusted set (`|trusted|≥2`);
rank fewer isolates, then H0, then index. Records are not deleted.
Isolate is not billed and is not treated as new evidence.

Eval n=60/cause, MAE mean±SE.

### `model` (truth in H1; labels reliable)

| hist | budget | old | new | isolate |
|---|---:|---:|---:|---:|
| Q | 8 | 0.397±0.154 | 0.397±0.154 | 0 |
| Q | 16 | 0 | 0 | 0 |
| QR | 8 | 0.477±0.156 | 0.477±0.156 | 0 |
| QR | 16 | 0 | 0 | 0 |

New never isolates a true model change (`wrong_isolate` is 0 in B as well).
Old and new coincide: H1 already fits the full believed set.

### `transient` (H0 would suffice; one first-look error)

| hist | budget | old | new | isolate | new family |
|---|---:|---:|---:|---:|---|
| Q | 8 | 8.334±0.908 | **0** | 1.00 | H0 |
| Q | 16 | 3.397±0.366 | **0** | 1.00 | H0 |
| QR | 8/16 | 0 | 0 | 0 | H0 |

Without isolate, covering histories fall to nearest neighbour (old_nn=1).
With isolate, H0 returns. After a successful retest (QR), isolate is
empty: the corrected label is trusted again.

### `persistent` (retest of the dirty point repeats the error)

| hist | budget | old | new | isolate |
|---|---:|---:|---:|---:|
| Q | 8 | 8.334±0.908 | **0** | 1.00 |
| Q | 16 | 3.397±0.366 | **0** | 1.00 |
| QR | 8 | 9.005±1.008 | 0.866±0.662 | 0.97 |
| QR | 16 | 3.636±0.380 | **0** | 1.00 |

Retest does not restore the label. Isolate of that point still recovers
H0. That is the previous round’s missing repair: the split at a later
query was already in the trace; the fitter would not drop the dirty
constraint.

**A answers the prior failure:** most of the persistent MAE was fitter
incapacity, not “no distinguishing experiment”. Distinct truths: model
59, transient 34, persistent 34 (seeds ≠ targets).

## Experiment B — share the new fitter, change only the next experiment

Policies: covering queries; short retest-then-cover; 1-step lookahead
over remaining repairs (uniform over those repairs, `ASSUMED`), picking
the query/retest with largest expected MAE drop per cost. Lookahead
uses candidate predictions, never hidden labels.

Primary endpoint: MAE at budget 16.

| cause | cover | retest-then-cover | compete |
|---|---:|---:|---:|
| model | 0 | 0 | 0 |
| transient | 0 | 0 | 0 |
| persistent | 0 | 0 | 0 |

All 60-way ties at budget 16. Compete **does** change the query sequence
(60/60 tasks, every cause) and is slower (~0.26–0.39 s vs ~0.07 s;
n_score on the order of 10^7–10^8). It does not change final MAE once
the new fitter is shared.

Budget 8 (secondary):

| cause | cover | retest-then-cover | compete | cover−compete |
|---|---:|---:|---:|---|
| model | 0.397±0.154 | 0.477±0.156 | **0.150±0.064** | +0.247; 9 / 4 / 47 |
| transient | 0 | 0 | 0 | 60 ties |
| persistent | 0 | 0.866±0.662 | 0 | 60 ties vs cover; retest worse on 2 |

On `model` at budget 8, lookahead sometimes hits an H1-splitting x
before covering does. That is a real but small, budget-sensitive edge.
On dirt, isolate already zeros the error under covering, so scheduling
cannot show a repair gain.

Wrong isolate on `model`: 0 for all three policies.

## Decision (predeclared)

- New fitter works; active query choice has **no extra primary-endpoint
  gain**. Adopt the simple fixed process with the new fitter (covering,
  or retest-then-cover if a transient should be written back rather than
  left isolated). Stop this strategy search.
- Do not promote 1-step lookahead as a research-diagnosis method. Do not
  claim a result about general science.
- Residual: isolate-cap-1 cannot express two dirty points; H1-without-
  isolate vs H0-with-isolate remains a ranking assumption (`ASSUMED`).

Golovin et al. (NeurIPS 2010) on equivalence-class determination is a
useful analogy for “enough information to decide”, not a transferred
guarantee. No theorem from that paper is claimed here.

## Files

`compete_experiment.py`, `PROTOCOL.json` (`de8c7225…`), `EVAL_SUMMARY.json`,
`A_ROWS.json`, `B_ROWS.json`, `B_TRAJ.json`, `RepairContract.tla` / `.cfg`,
`CORRECTION.md`, `CHECKPOINT.md`.
