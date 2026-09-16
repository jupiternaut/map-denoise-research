# E0 COMMANDS LOG

Working directory: `/home/grf/Documents/Codex/2026-09-15/e0/`
Interpreter: `PY=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python` (numpy 2.2.6, scipy 1.15.3, matplotlib 3.11.1; read-only use, nothing installed)
Environment for all runs: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg`
Timing: bash `time` builtin with `TIMEFORMAT='wall %3R s user %3U s sys %3S s'`; the wall/user/sys line and `exit=<code>` of every run are stored in `logs/<run>.time`, stdout+stderr in `logs/<run>.log`.

## 0. Protocol registration (before any code was run)

```
$ cd /home/grf/Documents/Codex/2026-09-15/e0 && sha256sum PROTOCOL.md && date -Iseconds
19112a92ffaeaa4dfce2560b3fdce4d412ddae37c3f6f2c643000aadd1057aed  PROTOCOL.md
2026-09-15T14:17:20+08:00
```
exit code: 0

`PROTOCOL.md` was not modified afterwards (sha256 re-checked at 14:41 and identical).

## 1. Runs (chronological)

All runs were launched from the working directory with the environment above, in the form
`{ time $PY e0_mechanism_test.py <args> > logs/<run>.log 2>&1; echo "exit=$?"; } 2>> logs/<run>.time`
(the launcher line itself was not captured verbatim for runs 1–4; the arguments, exit codes and wall times below are
taken from the `logs/*.log` headers and `logs/*.time` files, which were written by the runs themselves).

### Run 1 — smoke test (small grid), `logs/smoke_run1.log`, `logs/smoke_run1.time`
```
$PY e0_mechanism_test.py --smoke        # preset: nx=40 ny=20 (N=800), n_cal=200, seeds=1,2 → smoke/results, smoke/figures
```
exit code: 1; wall 1.595 s (user 2.051 s, sys 0.064 s). Finished 14:32.
All 16 smoke runs (decisions + evaluation) completed; the crash was in the report writer:
`TypeError: unsupported operand type(s) for -: 'int' and 'dict'` at `write_results_md` (coverage table), because after
seed-aggregation `aggs[key]['scenario']['alpha_tgt']` is a `{mean, sd, values}` dict, not a scalar.

**Bug 1 (report writer only, no effect on decisions/metrics):** fixed by using `['alpha_tgt']['mean']`.
Two further edits made at the same time before run 2, both outside decision/metric code: `write_decisions_csv` now takes
the already computed `t_argmin` as an argument instead of recomputing it (identical values); `ax.legend()` calls are
skipped when a panel has no labelled series (matplotlib warning only).

### Run 2 — smoke test rerun, `logs/smoke_run2.log`, `logs/smoke_run2.time`
```
$PY e0_mechanism_test.py --smoke
```
exit code: 0; wall 2.649 s (user 3.031 s, sys 0.105 s). Finished 14:33.
Wrote `smoke/RESULTS_RAW_smoke.md`, `smoke/results/`, `smoke/figures/`. One residual matplotlib warning
("No artists with labels found to put in legend", `ax.legend(fontsize=8)` at line 1604 of the then-current script): a
legend call on a figure panel with no labelled series (the smoke exch scenario has no CPR2 moves).

Edit before run 3 (figure code only): fig4 changed from one panel (exch scenario) to two panels (exch and sfm scenario,
f_big=0.10, α_tgt=0.5, seed 1) so that the panel with zero moves does not leave the figure empty; all `legend()` calls
guarded by "panel has labelled series".

### Run 3 — smoke report regeneration from saved JSON, `logs/smoke_run3_report_only.log`, `.time`
```
$PY e0_mechanism_test.py --smoke --report-only
```
exit code: 0; wall 1.789 s (user 2.223 s, sys 0.104 s). Finished 14:35. No warnings.

### Run 4 — FULL pre-registered grid (run exactly once), `logs/full_run1.log`, `logs/full_run1.time`
```
sha256sum e0_mechanism_test.py > logs/script_sha256_before_full_run1.txt
   → 658b31e8ed3d57821b65d25b560fe83e34861b2e26348a51aca53f7d693c8b7a  e0_mechanism_test.py
date -Iseconds > logs/full_run1.start          → 2026-09-15T14:36:13+08:00
$PY e0_mechanism_test.py                       # defaults: nx=200 ny=100 (N=20000), n_cal=2000, seeds=1,2,3 → results/, figures/
```
exit code: 0; wall 26.666 s (user 26.804 s, sys 0.396 s). Finished 14:36.
Outputs: `results/per_seed/*.json` (24 runs), `results/decisions/<scenario>_seed1.csv.gz` (8 files),
`results/aggregate.json`, `RESULTS_RAW.md`, `figures/fig1..fig4*.png`. No warnings, no tracebacks.
No bug was found in the decision/metric code after the full run; the grid was NOT rerun. No parameter or threshold was changed at any point.

### Run 5 — report regeneration from the saved full-run JSON (no recomputation), `logs/report_regen.log`, `logs/report_regen.time`
Edit before run 5 (report writer only): added section "0. Sanity counters" to `RESULTS_RAW.md` (0∈L anomaly total,
guard fire counts and min NN distance among moved points, conformal p-value floor per scenario read from the saved
seed-1 `decisions.csv.gz`). Copies of the run-4 report and aggregate were kept first:
`logs/RESULTS_RAW_full_run1_before_report_regen.md`, `logs/aggregate_full_run1_before_report_regen.json`.
```
$ cp -p RESULTS_RAW.md logs/RESULTS_RAW_full_run1_before_report_regen.md && cp -p results/aggregate.json logs/aggregate_full_run1_before_report_regen.json
$ sha256sum e0_mechanism_test.py > logs/script_sha256_before_report_regen.txt
   → b3d69fe9be611fb872182f8af7287c5e4ac27b5bf415b9cbd0802cc9e2537fac  e0_mechanism_test.py
$ date -Iseconds > logs/report_regen.start     → 2026-09-15T14:42:50+08:00
$ TIMEFORMAT='wall %3R s user %3U s sys %3S s'; { time $PY e0_mechanism_test.py --report-only > logs/report_regen.log 2>&1; echo "exit=$?" >> logs/report_regen.exit; } 2> logs/report_regen.time; cat logs/report_regen.exit >> logs/report_regen.time; rm logs/report_regen.exit
```
exit code: 0; wall 2.546 s (user 3.004 s, sys 0.080 s).
Verification: `cmp logs/aggregate_full_run1_before_report_regen.json results/aggregate.json` → identical;
`diff logs/RESULTS_RAW_full_run1_before_report_regen.md RESULTS_RAW.md` → only (a) the new section 0 and (b) a different
ordering of per-run rows in the per-seed tables (run 4 wrote rows in execution order seed→scenario; `--report-only`
loads the JSON files in scenario→seed order); every numeric row is unchanged. Figures were re-rendered from the identical aggregate.

## 2. Ad-hoc read-only checks (python one-liners on the saved `results/decisions/*.csv.gz`; no files written)

- 14:41 — p-value distribution in `fbig0.10_exch_atgt0.2_seed1` vs `fbig0.10_sfm_atgt0.2_seed1` (min p, counts at p ≤ 0.002/0.01/0.05, BH rejection count recomputed from the stored p column: 0 and 6347, matching the run). exit 0.
- 14:41 — `fbig0.10_sfm_atgt0.5_seed1`: among CPR2 MOVED & coverage-satisfied components (n=808), max of e_after/(w/2) = 1.975, none > 2 (bound e_after ≤ w holds); median 0.740 matches RESULTS_RAW.md. exit 0.

## 3. Final file hashes

```
19112a92ffaeaa4dfce2560b3fdce4d412ddae37c3f6f2c643000aadd1057aed  PROTOCOL.md
b3d69fe9be611fb872182f8af7287c5e4ac27b5bf415b9cbd0802cc9e2537fac  e0_mechanism_test.py   (after run 5)
658b31e8ed3d57821b65d25b560fe83e34861b2e26348a51aca53f7d693c8b7a  e0_mechanism_test.py   (at run 4, from logs/script_sha256_before_full_run1.txt)
```
The only edits between run 4 and run 5 are the ones listed under run 5 (report-writer section 0). The run-4 file itself
was not archived, so this statement rests on the edit log above, not on a byte-level diff; the run-4 numbers are
archived in `results/per_seed/*.json` and `logs/RESULTS_RAW_full_run1_before_report_regen.md`, and run 5 reproduced
them exactly from those files.
