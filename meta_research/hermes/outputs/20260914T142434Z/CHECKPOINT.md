# CHECKPOINT

Host: `liekkas`
Workdir: `/home/grf/.hermes/attachments/outputs/20260914T142434Z`
Start: 2026-09-14T14:24:34Z

Frozen `20260914T132630Z` was not overwritten (code sha256
`8f0e71efcf3681f51804f2eda84fc02517c2db819b73075cea0d492d28f47920`).
H=6 was not re-run. No scheduler / fitter / generator upgrade.

## Done

1. Independent H=4 open-loop re-solve on all 438 unique init histories
   under value-only vs generator-location posteriors.
2. Example `{2:1, 5:−1}`: 79 vs 12 remaining worlds; open-loop stays
   query 3 then 6; adaptive branches to 0 iff y=−1 else 6.
3. Catalog H=4 MAE remains 0.0102564 under both filters. 36 inits
   change plan; those 108 worlds are disjoint from the 72 residuals.
4. Finite-instance keep/stop written. Scheduler contest on this
   generator is closed.

## Not done (by design)

- H=6 re-execution (locked 2340/2340 zeros stand as recorded)
- smallest budget that admits a zero-error open-loop
- second generator
- soft weights
- incomplete-catalog expansion (next lock, if any)

## Commands

```
python3 init_design_audit.py
```

## Restore

`INIT_DESIGN_AUDIT.json`, `INIT_DESIGN_ROWS.json`, `REPORT.md`.
