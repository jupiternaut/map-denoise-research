# Phase-1 correction (does not replace MISMATCH_REPORT.md)

Original report remains at
`/home/grf/.hermes/attachments/outputs/MISMATCH_REPORT.md`.
This note only records the two audits required before the new experiment.

## Randomness attribution

Same per-task seed `rng_seed=7000` for random and disagreement, 20 eval-style
tasks per hidden type from the frozen mismatch generator.

| type | same queries until H0 refuted | mean true-disagreement queries | mean random-fallback queries | H0 never refuted |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 0.00 | 14.00 | 20/20 |
| 1 | 1.00 | 1.45 | 12.55 | 0/20 |
| 2 | 1.00 | 1.65 | 12.35 | 0/20 |

Until H0 dies, disagreement is not a different query policy under this seed:
live C0 has no residual disagreement, so it uses the same RNG fallback as
`random`. True disagreement queries (1.45–1.65 of 14 post-init steps) occur
after H0 is already empty, when the selector moves to live C1.

The original Type-1 L1 gain at budget 4 is therefore not “random and
disagreement explored differently before the old model died”. It is
post-refutation sampling inside H1. Random-fallback counts and true
disagreement counts are stored in `PHASE1_ALIGNMENT_ROWS.json`.

## Real dual-environment isolation

Two `Env` objects. Honest selectors receive only the queried dict / cover
index / RNG — not hidden labels. Unqueried answers on env B are rewritten
to `17*x+3` (a constant shift does **not** change `argmax` of hidden labels;
that was the first detector bug, recorded in `ERROR_LOG.md`).

| policy | n | n_fail | contract |
|---|---:|---:|---|
| fixed / random / disagreement | 12 each | 0 | must pass |
| cheat (peeks `env.label` on unqueried x) | 12 | 6 | must fail |

Honest pass, cheat fail: `PHASE1.json` `pass=true`.

The original mismatch isolation stored the mutated array as `_unused` and
never handed a second environment to a selector. That test is weaker than
this one; it is not re-run as a substitute.
