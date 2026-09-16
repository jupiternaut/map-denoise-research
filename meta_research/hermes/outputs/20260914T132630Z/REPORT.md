# Support-complete catalog: open-loop plan vs adaptive DP

Prior `20260914T124618Z` was not modified. Soft weights were not added.
No second generator.

Question: after locking a catalog that enumerates the public generator,
does changing the next experiment from new outcomes beat the best plan
that may use the catalog and the init tape but cannot revise afterwards?

\[
\Delta_{\mathrm{adapt}}
=
R_{\text{open-loop}}
-
R_{\text{adaptive}}.
\]

That split is decision-oriented active learning: information is worth
buying only if it changes the delivered repair, not merely because it
shrinks uncertainty.[1] Equivalence-class determination makes the same
cut between identifying every remaining hypothesis and gathering enough
to decide; it is an analogy, not a transferred guarantee.[2]

## Setup (locked)

| object | value |
|---|---|
| fitter | isolate-cap-1 from `20260914T111013Z` (imported, not edited) |
| catalog | exhaustive enum of the public generator: 780 instances, 2340 worlds |
| prior | ASSUMED uniform on enumerated worlds |
| horizons | 4 and 6, each solved separately |
| confirm | 30 new bundles, seed 40_000, 90 worlds; not used to pick the catalog |
| empty support | `support_failed` + covering fallback |

Policies share the fitter, action menu, init observations, and delivery.

| policy | catalog + init | revises after new outcomes |
|---|---|---|
| cover | no | no |
| retest_then_cover | conflict rule only | partially |
| openloop | yes | no |
| adaptive | yes | yes |

## Support (MEASURED)

On the enumerated catalog, every true world remains in the posterior
after init and after covering to budget 6:

- missing true world after init: 0 / 2340
- empty after init: 0
- empty after cover @6: 0
- support size after init: min 11, mean 52.1, max 90

Independent confirm samples (seed 40_000) also have nonempty init
support (20–90). Empty-catalog fallback was never entered.

Isolation: honest next action invariant 36/36; cheat 9/9 caught.
History unit: retest changes `cur` (−1 vs 0), `orig` unchanged; tape
filter is stricter than `cur`-only. Open-loop sequence stays frozen
after the first executed action (smoke). Adaptive Bayes risk on the
largest init posterior (90 worlds, rem=4) is 0 and ≤ open-loop.

Code hash: `8f0e71efcf3681f51804f2eda84fc02517c2db819b73075cea0d492d28f47920`.
Distinct hidden targets: catalog 336, confirm 81 (not the seed count).

## Catalog (2340 worlds)

| H | cover | retest | openloop | adaptive | Δ_adapt |
|---|---:|---:|---:|---:|---:|
| 4 | 0.8788 | 0.7534 | 0.0103 | **0** | 0.0103 |
| 6 | 0.2769 | 0.4394 | **0** | **0** | **0** |

By cause, H=4:

| cause | n | cover | retest | openloop | adaptive | seq differs | openloop residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| model | 780 | 0 | 0.3119 | 0.0308 | 0 | 234 | 72 |
| transient | 780 | 1.3183 | 0 | 0 | 0 | 0 | 0 |
| persistent | 780 | 1.3183 | 1.9484 | 0 | 0 | 0 | 0 |

By cause, H=6: openloop = adaptive = 0 on every cause. Cover still has
dirt residual 0.4154; retest remains 1.3183 on persistent.

The 72 catalog worlds with open-loop residual at H=4 are all `model`.
Adaptive is strictly better on those 72 and tied on the other 2268.
Adaptive never worse. Sequences: open-loop locks `query 3` then a
right-side point (`5` or `6`); adaptive also starts at `3`, then
switches to a left-side point (`0` or `1`). Cover already has MAE 0
on every model world at H=4, so this increment is **not** a win against
covering on model — it is a win against a catalog-optimal hedge that
spent the last slot on the dirt side.

## Confirm (new seeds, 90 worlds)

| H | cover | retest | openloop | adaptive | Δ_adapt |
|---|---:|---:|---:|---:|---:|
| 4 | 0.8472 | 0.7917 | 0.0264 | **0** | 0.0264 |
| 6 | 0.2250 | 0.4236 | **0** | **0** | **0** |

Confirm H=4 model: 7/30 open-loop residuals (MAE 0.25 or 0.375),
adaptive 0. Same pattern as the catalog. Dirt: openloop = adaptive = 0.
Empty support: 0.

## What this attributes

1. **Catalog knowledge, not mid-course feedback, is the large term.**
   Open-loop already zeros H=6 on the full generator and on the confirm
   sample. Cover and the short-retest rule do not.

2. **Δ_adapt is near zero at the longer horizon.** At budget 6 a single
   precommitted sequence is enough. Replanning still changes 234/780
   model traces, but terminal MAE does not.

3. **A real, narrow adaptive increment exists at budget 4.** It is
   confined to model-mismatch worlds where the open-loop hedge spends
   the last query on the breakpoint’s right side. After seeing `x=3`,
   adaptive moves left. 72/2340 catalog worlds, 7/90 confirm worlds.

4. Retest_then_cover is already a feedback rule. It zeros transient and
   is the worst policy on persistent. “Fixed process” is not the same
   as “non-adaptive.”

## Keep / stop

Predeclared:

- Δ_adapt ≈ 0 → value is better experimental design, not smarter sequential
  revision.
- Stable Δ_adapt > 0 → then a sequential decision algorithm is justified.
- Same increment after a new dirt family → then, and only then, a
  transfer test.

On this generator, with this fitter:

- **H=6 (enough budget to finish a hedge): stop sequential search.**
  Adopt a catalog-using open-loop plan. Adaptive adds no terminal MAE.
- **H=4: keep the increment as a horizon-specific fact**, not as a
  general research-strategy result. It does not beat covering on model;
  it beats a dirt-hedging open loop that is short one query.

Soft weights remain deferred: isolate-cap-1 is not the binding
constraint once the catalog is complete, and dirt vs model already share
a working repair (isolate or expand) under a long enough open-loop.

Second generator still deferred. The next honest test, if any, is
whether the H=4 residual survives a different observation mechanism —
not another selector on this one.

## Sources

[1] https://arxiv.org/abs/1402.5886 — Near Optimal Bayesian Active Learning for Decision Making (Javdani et al., AISTATS 2014)
[2] https://papers.nips.cc/paper_files/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf — Near-Optimal Bayesian Active Learning via Adaptive Submodularity (Golovin, Krause, Ray, NeurIPS 2010)
