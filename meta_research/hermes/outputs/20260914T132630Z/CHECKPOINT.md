# CHECKPOINT

Host: `liekkas`
Workdir: `/home/grf/.hermes/attachments/outputs/20260914T132630Z`
Start: 2026-09-14T13:26:30Z

`20260914T124618Z` and `20260914T111013Z` were not overwritten.
No soft weights. No second generator.

## Done

1. Enumerated the public generator: 780 instances, 2340 worlds.
2. Support-complete: true world always remains after init and after
   cover @6; init support 11–90, never empty.
3. Compared four policies on the same fitter: cover, retest_then_cover,
   catalog-optimal open-loop, ExactDP adaptive. Horizons 4 and 6 solved
   separately.
4. Confirm: 30 new bundles, seed 40_000, not used to pick the catalog.
5. Isolation: honest 36/36 invariant; cheat 9/9 caught.
6. History unit: retest changes `cur`, `orig` stays.
7. Open-loop sequence frozen after first executed action (smoke).

## Results (MEASURED)

- Catalog H=6: openloop = adaptive = 0. Δ_adapt = 0.
- Catalog H=4: Δ_adapt = 0.0103, all 72 residuals are `model`.
- Confirm H=6: Δ_adapt = 0.
- Confirm H=4: Δ_adapt = 0.0264, 7/30 model residuals.

## Keep / stop

H=6: stop sequential search on this generator; adopt catalog-using
open-loop. H=4: keep as a horizon-specific residual, not a general
strategy result. Soft weights and a second generator remain deferred.

## Commands

```
python3 enum_probe.py
python3 support_probe.py
python3 dp_scale.py
python3 smoke.py
python3 adapt_experiment.py
```

## Restore

`PROTOCOL.json` code_sha256 =
`8f0e71efcf3681f51804f2eda84fc02517c2db819b73075cea0d492d28f47920`

Files: `adapt_experiment.py`, `EVAL_SUMMARY.json`, `CATALOG_ROWS.json`,
`CONFIRM_ROWS.json`, `REPORT.md`.
