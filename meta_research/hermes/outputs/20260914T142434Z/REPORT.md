# Finite instance: H=4 strict adaptivity, H=6 open-loop zero

Frozen checkpoint `20260914T132630Z` was not modified. This directory
is an independent audit plus a keep/stop write-up. H=6 was not re-run.
No scheduler, fitter, or generator upgrade. No soft weights.

The locked comparison already separated two questions:

| contrast | what it tests |
|---|---|
| cover / short-retest → catalog-optimal open-loop | using the catalog *and* solving a plan |
| catalog-optimal open-loop → ExactDP adaptive | **value of revising after later outcomes** |

Open-loop is itself a decision policy: it may use the catalog and the
init tape, then executes a frozen sequence. “Adaptive extra gain is
zero at H=6” is not “decision science unused.” It is “after the init
tape, further revision has no remaining terminal MAE.”

Buying an experiment is worth it only if it changes the delivered
repair, not merely if it shrinks uncertainty.[1] Equivalence-class
determination draws the same cut between identifying every remaining
hypothesis and gathering enough to decide; analogy only, no transferred
guarantee.[2]

## Locked numbers (MEASURED, not re-solved here)

Catalog = exhaustive enum of the public generator: 780 instances,
2340 worlds. Confirm = 30 independent samples, seed 40_000; same
generation family, **not** a new mechanism.

| H | cover | retest | openloop | adaptive | Δ_adapt |
|---|---:|---:|---:|---:|---:|
| 4 | 0.8788 | 0.7534 | 0.0102564 | **0** | 0.0102564 |
| 6 | 0.2769 | 0.4394 | **0** | **0** | **0** |

H=6: 2340/2340 worlds have open-loop MAE = adaptive MAE = 0. MAE cannot
go below zero, so on this catalog, fitter, action menu, and terminal
MAE, a more elaborate scheduler has no remaining endpoint precision.

H=4: adaptive MAE = 0 on 2340/2340. Open-loop residual is 72 worlds,
all `model` (48 at 0.375, 24 at 0.25). Adaptive is strictly better on
those 72 and tied on the rest; never worse.

Confirm (same family, independent draws): H=6 Δ_adapt = 0; H=4
Δ_adapt = 0.0264 from 7/30 model worlds. Call this **independent
sampling of the enumerated family**, not transfer.

## Mechanism at H=4 (re-solved)

Named init `{2: 1, 5: −1}`:

| object | MEASURED |
|---|---|
| value-only posterior | 79 worlds |
| generator-location posterior | 12 worlds |
| open-loop sequence (both filters) | query 3, then query 6 |
| adaptive, value posterior | first query 3; if y=−1 then query 0, else query 6 |
| adaptive Bayes risk | 0 |

The second action **changes with the new label**, not only with the
action name in a log. Remaining budget cannot buy both the left-side
and right-side last query; wait-then-choose has terminal value.

Cover already has MAE 0 on every `model` world at H=4. That does not
cancel the increment: the policy does not know the hidden cause when
it acts. A cause-unknown policy is not assembled by picking the best
specialized baseline after the fact.

Open-loop already conditions on the init tape. The H=6 stop statement
is “after init, do not keep revising,” not “observation is unused.”
Only H=4 and H=6 were solved; 6 is not claimed to be the smallest
budget that admits a zero-error open-loop.

## Init-design observation model (audit, not a retraction)

The generator chooses init locations from the hidden world (`c1` on
the left of the breakpoint, `dirty_x` on the far side). The locked
filter keeps worlds that match the **values** at those locations, not
worlds that would have **generated the same location pair**.

Independent re-solve of H=4 open-loop on all 438 unique init histories:

| | value filter (locked) | generator-location filter |
|---|---:|---:|
| unique inits | 438 | 438 |
| posterior size min / median / mean / max | 11 / 35 / 40.8 / 90 | 3 / 3 / 5.34 / 12 |
| inits whose open-loop plan changes | — | **36** |
| catalog worlds whose init is in those 36 | — | 108 |
| catalog H=4 open-loop MAE | **0.0102564** | **0.0102564** |
| residual worlds | 72, all model | 72, all model |
| residual worlds among the 36 changed inits | — | **0 / 72** |
| true world always remains | yes | yes |

The 36 changed plans are a different slice of the catalog
(`query 1,5` vs `query 0,5` and `query 1,6` vs `query 0,6`). They do
not contain the 72 H=4 residuals. Executed catalog MAE is therefore
identical. Planned Bayes risk of a given init can still move (example:
0.0665 vs 0.0313) because the posterior is smaller.

ASSUMED: locked policies treat init locations as given and condition
only on labels. MEASURED: tightening that model changes some plans and
does not change the catalog H=4 MAE or the 72-world residual set.

Document the observation model. Do not retract Δ_adapt.

## Correspondence (not a solution of open-ended science)

| correspondence | strength | keep | drop |
|---|---|---|---|
| finite catalog + ExactDP | restricted instantiation | known worlds, unknown which; nested terminal MAE | unknown families, unknown representation, unknown proxy |
| Δ_adapt vs open-loop | restricted instantiation | revision value after init, per solved horizon | “science must always replan” |
| Javdani / ECD | useful analogy | information for a decision, not for identification | their approximation guarantees; our generator |

This apparatus studies strategy **inside a known, enumerable set of
worlds**. That is part of the original meta-research question, not a
substitute for “the correct model, representation, and goal proxy are
themselves unknown.”

## Keep / stop

On this generator, this fitter, this action menu, this terminal MAE:

1. **Stop the scheduler contest.** H=6 open-loop already zeros every
   enumerated world. More sequential machinery cannot improve the
   endpoint.
2. **Keep the H=4 fact.** There exist inits where competing repairs
   need different last queries, the remaining budget cannot buy both,
   and the new label changes the useful action. Strict adaptive gain
   is real, horizon-specific, and not a general research strategy.
3. **Do not call confirm a transfer result.** It is a new sample from
   the same enumerated family.
4. **Do not introduce soft weights** here: isolate-cap-1 is not the
   binding constraint once a long-enough catalog plan exists.
5. **Do not start a second generator to rescue the scheduler.**

If meta-research continues, the next question is a new lock:

> When the candidate explanations are incomplete, what evidence is
> enough to expand the explanation set, rather than to reorder
> experiments inside an already-complete catalog?

Zero error on these 2340 known worlds does not answer that.

## Sources

[1] https://arxiv.org/abs/1402.5886 — Near Optimal Bayesian Active Learning for Decision Making (Javdani et al., AISTATS 2014)
[2] https://papers.nips.cc/paper_files/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf — Near-Optimal Bayesian Active Learning via Adaptive Submodularity (Golovin, Krause, Ray, NeurIPS 2010)
