# CHECKPOINT

Host: `liekkas`
Workdir: `/home/grf/.hermes/attachments/outputs/20260914T124618Z`
Start: 2026-09-14T12:46:18Z

`20260914T111013Z` was not overwritten.

## Done

1. Reproduced both comparison bugs on frozen code:
   `reproduce_old_bugs.py` (DP@4 zeros 18/18; holdout 15 empty at
   init + 3 at 5; cur-only 5 vs tape 1).
2. History-consistent filter; retest-changes-label unit.
3. Empty catalog → support_failed + covering fallback.
4. Separate H=4 and H=6 policies.
5. Catalog E[MAE] DP ≤ cover and ≤ retest at both horizons.
6. Confirmation on **new** seeds 20000+, not the inspected holdout.
7. Isolation: honest 27/27, cheat 3/9 fail.

## Not done

- Soft weights.
- Larger catalog so init does not empty support.
- Second generator.

## Commands

```
python3 .../reproduce_old_bugs.py
python3 .../smoke.py
python3 .../horizon_experiment.py
```

Protocol `code_sha256=fa627c6ac2b3b4b9e5279f2a4fcfbe226dbbd3cb69c2d7269572cce98672286f`.
