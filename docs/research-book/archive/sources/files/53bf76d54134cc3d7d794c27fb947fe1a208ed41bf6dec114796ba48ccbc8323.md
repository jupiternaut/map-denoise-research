# Correction to 20260914T051316Z/REPORT.md

The previous checkpoint is not overwritten. These three points replace
the corresponding claims in that report. Numbers below are from the
existing `EVAL_TRAJ.json` / `conflict_experiment.py`, not a new eval.

## 1. Same-point retest does not distinguish; a new location can

Paired tasks share `init_obs`. On all 60 eval pairs, first label at x=32
is identical for `model` vs `persistent`. Retesting 32 returns that same
value in both causes.

On all 60 pairs, both causes later query x=48 under `expand_first`, and
the returned labels differ (example `eval-b000`: model −1, persistent −48).
Covering order after `{0,32,16}` is `48`. This is **post-hoc evidence**
from existing traces. It is not a license to hard-code 48 in a new policy.

So “no available action can distinguish” is false. The true failure is:
the system already obtained a splitting observation, then the fitter still
treated the dirty label as an irreversible hard constraint.

## 2. Expand did not diagnose; it bought a better nearest-neighbour fallback

`deliver` (conflict_experiment.py:268–278): if the active family has no
live row, fill every x with the label of the nearest queried point. There
is no “H1 compromise fit” branch.

Under persistent dirt, all three policies end unresolved. Expand spends
zero retests and more queries, so the fallback interpolates more points.
That is a real MAE drop, not a correct cause identification.

## 3. “Beat both specialized baselines” is not required

A policy that does not know the cause may still be useful if it selects
the right existing repair more often than a fixed process. Diagnose at
budget 8, equal mix of the three causes: MAE 3.607 vs expand 4.098 vs
retest 4.130. Keep that secondary number. At budget 16 (primary) diagnose
ties retest and does not beat it. Keep that too.

Do not continue ranking conflict points. Next: competing explanations
that can change the repair, after a fitter that is allowed to isolate.
