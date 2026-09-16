# Implementation errors

## E1. Unexpanded H1 fallback in `deliver()`

**When found:** after first eval (`EVAL_*`) and the first cost ablation (`SUPPLEMENT.json`).

**What:** `deliver()` used `first_fit(H1, obs)` whenever H0 was inconsistent, even if the policy had not paid `expand`. That is a free model-class gift.

**Effect:** a policy that never expanded could still emit an H1 hypothesis. Visible in the first cost ablation: `retest_first` on `model` at budget 8 with `COST_RETEST=2` had `exp=0` and L1=0.

**Repair:** `deliver()` now fits only `agent.library()` (H0 until expand, H1 after). Else nearest-neighbour on observed labels.

**Rerun scope:** original `EVAL_*`, `DEV_ROWS.json`, and `SUPPLEMENT.json` are **pre-repair** and must not be used as confirmation. Confirmation used new seeds 40000+ (`CONFIRM_*`). Cost / generator supplements after this log use the repaired `deliver()`.

**Did not:** peek at eval labels, relax L1, or change the primary endpoint.
