# Horizon-correct DP, history-consistent belief, empty-catalog fallback

Prior `20260914T111013Z` was not modified. Soft weights were not added.
The inspected holdout of that round is **not** used as confirmation.

The previous report’s two comparison errors were reproduced on the frozen
code (`reproduce_old_bugs.py`):

| check | frozen code |
|---|---|
| DP solved for H=6, sliced at step 4 | catalog @4: cover 0 / 1.375 / 1.375 vs slice 0.292 / 2.833 / 2.833 |
| DP solved for H=4 | **18/18 MAE 0** |
| Holdout empty catalog | 15 empty at cost 2 (init), 3 empty at 5, 0 never |
| Empty then stop vs empty then cover | model 0.292→0; dirt 1.3125→0.625 |
| Cur-only vs orig+retest tape | `{2:2,7:-1}` then retest 7→7: 5 worlds vs 1 |

## Acceptance (this round)

1. Belief uses first-look `orig` and the retest outcome. `cur` is fitter
   input only. Unit: retest **changes** the label (`-1` → `7`), `orig`
   stays `-1`, tape support 1 < cur-only 5.
2. Empty catalog is `support_failed` + covering fallback. Not zero loss.
3. Separate solvers for horizons 4 and 6. A checkpoint is the terminal
   of that horizon’s policy, not a slice of a longer one.
4. On the catalog, expected MAE of `optimal_h` ≤ cover and ≤
   retest_then_cover at the same h. (Expectation over the 18 worlds,
   not a per-world dominance claim.)

Isolation: honest 27/27; cheat 3/9 fail. DP states 4933.

## Catalog (same 18 worlds the DP is Bayes for)

Uniform mix, n=18.

| H | cover | retest | optimal | acc ≤cover / ≤retest |
|---:|---:|---:|---:|---|
| 4 | 0.917 | 1.042 | **0** | yes / yes |
| 6 | 0.292 | 0.458 | **0** | yes / yes |

By cause, H=4: model 0 / 0.292 / **0**; transient 1.375 / **0** / **0**;
persistent 1.375 / 2.833 / **0**. Empty-catalog rate 0. Wrong isolate 0.

So the in-catalog decision space is larger than the previous report said:
there is a horizon-4 policy that recovers all 18 tasks, and it is not
the covering order. Retest-then-cover already zeros transient; it is
the wrong action for persistent (2.833). The DP switches.

This still uses the catalog as the prior. It is not a general scheduler.

## Confirmation (new seeds 20000+, same generator, 18 worlds)

Not the inspected holdout.

| H | cover | retest | optimal | empty rate (opt) |
|---:|---:|---:|---:|---|
| 4 | 0.792 | 0.681 | **0.583** | model 1.00, dirt 0.83 |
| 6 | 0.208 | 0.396 | **0** | model 1.00, dirt 0.83 |

Acceptance on the mix still holds. Per-cause at H=4, retest_then_cover
remains **0** on transient while optimal is 0.875: after the tape leaves
the catalog, fallback covering is the wrong repair for transient dirt
in the remaining two steps. At H=6 the leftover covering budget is
enough for optimal to reach 0 on all three causes (fallback steps
~3.3–3.5).

That is **not** proof that catalog-Bayes transfers. Most confirm
trajectories spend the second half in declared covering fallback.
It does show that the previous holdout collapse was the stop-on-empty
path, not an independent test of “scheduling fails after a mechanism
change”.

## What this does and does not license

- In-catalog, horizon-matched DP has real value versus both fixed
  processes. The earlier “covering beats DP at budget 4” number was
  a slice of the H=6 policy.
- Out of catalog, empty support must be a named fallback. Stopping
  is not a risk-minimizing action.
- Do not promote this DP as a research strategy. Do not introduce
  `w∈[0,1]` on the strength of these numbers: isolate-cap-1 already
  zeros catalog dirt when the dirty point is isolated; confirm error
  at H=4 is leftover covering after prior mismatch.
- Strategy search is **not** closed. The next honest test is a
  **locked confirmation catalog** large enough that init does not
  empty it, or a stated second generator — still without soft weights.

Cassandra, Kaelbling, Littman (AIJ 1998) is cited only for the
requirement that the decision state keep history constraints. No
algorithm from that paper is implemented here.

## Files

`reproduce_old_bugs.py`, `horizon_experiment.py`, `PROTOCOL.json`
(`code_sha256` in that file), `EVAL_SUMMARY.json`, `CATALOG_ROWS.json`,
`CONFIRM_ROWS.json`, `CHECKPOINT.md`.
