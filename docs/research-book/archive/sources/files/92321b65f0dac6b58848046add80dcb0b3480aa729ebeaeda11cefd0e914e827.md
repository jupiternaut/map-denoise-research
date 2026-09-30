# Exact decision reference after a frozen isolate fitter

Prior directories `20260914T084840Z` and `20260914T051316Z` were not
modified. Soft weights were not introduced.

Question: after the fitter can reversibly isolate one observation, is
there still scheduling value worth paying for?

This is a finite apparatus (N=8, H0=6, H1=66), not a theory of fuzzy
sets or of research in general. Catalog prior, costs, and isolate-cap-1
are `ASSUMED`.

## 1. Correctness cleanup

Against `20260914T084840Z/compete_experiment.py`:

| defect | change here |
|---|---|
| Retest overwrote the only tape | `orig` first-write; `retest` edits `cur` only. Measured: `orig_stable_after_retest`. |
| Each (family, isolate) kept `table[live][0]` | Distinct live rows kept (`competing_preds`, cap 8 `ASSUMED`). Delivery still ranks fewer isolates, then H0, then index, then the tuple. |
| Lookahead weights not a probability (`w=0` rewritten to `1/nlab`) | Lookahead is a **supplement**. Query branch: counts of remaining repair predictions, renormalized. Retest branch: `P_TRANSIENT_RETEST=0.5` `ASSUMED` mixture over alternate labels. Not used as the primary comparator. |
| TLA claimed append-only / hidden unused | Split `orig`/`cur`. Claims listed in `TLA_CLAIMS.md`. |

TLC (`RepairContract`): 139 generated, 51 distinct, 0 on queue, depth 11,
no error. Checked: `TypeOK`, isolate ⊆ orig, orig never rewritten by
retest, `cur` only where `orig` is defined. Not checked: cheating
policies, numeric MAE, liveness, `MaxClock` (artifact).

Isolation (two `TapeEnv` objects; unqueried labels rewritten): honest
27/27 pass; cheat 1/9 fail.

## 2. Exact adaptive reference

Frozen fitter: isolate-cap-1, competing live rows, same ranking.
Catalog: 6 bundles × 3 causes = 18 worlds (uniform prior `ASSUMED`).
Holdout: 6 new bundles, same generator, seeds 10000+.
DP knows the catalog, libraries, and observation/retest mechanism.
It does **not** receive the current world’s hidden cause or truth.
Covering and short-retest use the same fitter and do not see the catalog.

Primary policies: `cover`, `retest_then_cover`, `optimal` (DP).
Lookahead is tagged `NOT_PRIMARY`.

Budgets 3–6 (init uses 2). Distinct dirty_x in catalog: 4. Distinct t: 2.
Shortcut init `{0,4}` rejected.

DP: 4703 memo states, 0.42 s.

## 3. Catalog (in prior) MAE mean±SE, n=6/cause

### `model`

| b | cover | retest | optimal |
|---:|---:|---:|---:|
| 3 | 0.292 | 0.292 | 0.292 |
| 4 | **0** | 0.292 | 0.292 |
| 5 | **0** | 0.292 | **0** |
| 6 | **0** | **0** | **0** |

Covering already zeros at budget 4. Optimal matches covering at 5–6;
at 4 it is worse on 2/6 tasks (still querying a catalog-value point).
Wrong isolate: 0.

### `transient`

| b | cover | retest | optimal |
|---:|---:|---:|---:|
| 3 | 2.833 | 2.833 | 2.833 |
| 4 | 1.375 | **0** | 2.833 |
| 5 | 0.438 | **0** | 1.375 |
| 6 | 0.438 | **0** | **0** |

Retest-then-cover zeros at budget 4 by writing the true label back
(isolate rate 0). Covering isolates 4/6 at budget 6 and leaves 0.438.
Optimal isolates all 6 at budget 6 (MAE 0) without retesting.

### `persistent`

| b | cover | retest | optimal |
|---:|---:|---:|---:|
| 3 | 2.833 | 2.833 | 2.833 |
| 4 | **1.375** | 2.833 | 2.833 |
| 5 | 0.438 | 2.833 | 1.375 |
| 6 | 0.438 | 1.375 | **0** |

Retest cannot fix sticky dirt; it delays isolate. Covering isolates
4/6. Optimal isolates all 6 at budget 6.

Mid-budget, covering beats optimal (paired cover−opt at 4: −1.46,
5 worse / 1 tie). Terminal budget: optimal’s extra isolates are the
only catalog gain versus covering.

## 4. Holdout (same public info, new worlds)

The DP still conditions only on catalog worlds compatible with the
observed tape. Holdout truths are not in the catalog.

| cause @6 | cover | retest | optimal | cover−opt |
|---|---:|---:|---:|---|
| model | **0** | **0** | 0.292 | −0.292 (3 worse / 3 tie) |
| transient | 0.625 | **0** | 1.312 | −0.688 (3 worse / 3 tie) |
| persistent | **0.625** | 0.812 | 1.312 | −0.688 (3 worse / 3 tie) |

Catalog-optimal actions do not transfer. On holdout, covering or
retest-then-cover (by cause) dominate the DP policy.

## 5. Lookahead supplement (catalog only, not ranked)

Normalized 1-step, `ASSUMED` retest mixture 0.5. At budget 6 it matches
optimal on dirt (MAE 0 by isolating) and covering on model (MAE 0).
It is not a new primary policy. Costs of scoring were not the endpoint.

## 6. Decision (predeclared)

- There **is** a catalog-internal scheduling gap at low-to-mid budget
  (retest vs cover vs wait-to-isolate), and a small terminal gap on
  persistent dirt (6/6 isolates vs covering’s 4/6).
- The exact adaptive policy that closes the catalog terminal gap
  **fails on holdout**. That is not “optimal is worse than possible”;
  it is “Bayes for this 18-world prior is not a research strategy”.
- Existing fixed processes already use the frozen fitter. Do not
  construct a more elaborate decision algorithm on this generator.
- Soft weights are not justified: hard isolate already zeros catalog
  dirt when the dirty point is isolated, and holdout error is prior
  mismatch, not a missing fractional membership.

Next spend, if any: a **different catalog** (larger, or a stated
family of dirt mechanisms) before any scheduler, or stop. Not fuzzy
`w_i∈[0,1]` until a task exists where isolate-cap-1 is the binding
constraint and two causes still need different actions after that
fitter.

Zadeh (1965), Javdani et al. (AISTATS 2014), Reiter (1987) remain
citations for the distinction among membership / probability /
decision value. No theorem from those papers is claimed here.

## Files

`exact_experiment.py`, `PROTOCOL.json` (`792dd99c…`), `EVAL_SUMMARY.json`,
`CATALOG_ROWS.json`, `HOLDOUT_ROWS.json`, `CATALOG_TRAJ.json`,
`RepairContract.tla` / `.cfg`, `TLA_CLAIMS.md`, `CHECKPOINT.md`.
