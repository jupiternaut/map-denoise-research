# Detector execution and reporting notes

Host verified as `liekkas`; working directory was
`/home/grf/Documents/Codex/2026-09-15/e0-diagnostics-20260915T072044Z`.

Commands executed from that exact directory:

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s tests -p test_detector.py -v
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B detector.py
```

Seven unit tests passed. The 18-cell experiment completed in 9.11 seconds.
Existing result directories cause the runner to stop rather than overwrite data.

Additional read-only replay checks, using the same environment and no new scene
seeds, verified all saved prefix hashes, observation identity across budgets,
p-value hashes, every saved sorted-rank BH margin, and equality to the frozen BH
implementation. Regenerating the same six old-model observations also reproduced
the full observation hashes and the old `decide` p-values/BH masks at 2k exactly.
Results of these checks are in `verification.json` and
`old_replay_verification.json`.

## Reporting corrections after the run

The ambiguous summary field `more_calibration_fixes_all_zero_bh` was renamed
`any_larger_budget_has_nonzero_bh`, and `all_32000_cells_zero_bh` was added. The
earlier summary is retained as `summary_before_reporting_rename.json`.
No raw per-case metrics or arrays changed, and the experiment was not rerun.
Both new flags are true: one 8k cell rejects 76 points, and every 32k cell rejects
zero. This is not a consistent repair of the all-zero behavior.

The old generator's `mm` flag means multimodality. Initial output names
`wrong_association` and `calibration_wrong_association` refer only to this `mm`
flag and do not measure known wrong associations. They remain in the raw outputs
to preserve the original record; `reporting_revision.json` defines their precise
aliases as `multimodal` and `calibration_multimodal`. Future source uses those
precise names. Source hashes at the run and after reporting corrections are
recorded in `run_definition.json` and `reporting_revision.json`.

A pre-run RNG multiplier typo was corrected to `round(f_big * 100)`, matching
the old E0 runner, before any detector experimental outputs existed.

## Independent projection audit

A read-only check of the root agent's 24 projection NPZ cases independently
recomputed 1,296 support/arm/subset blocks. Saved MAE and RMSE matched exactly
(maximum difference 0.0); movement counts, eligibility and covered counts also
matched. All 24 cases shared the exact detector evaluation truths, subsets and
observed strata; all 12 exch p-value arrays matched detector 2k p-values exactly.
This audit changed no projection files.

## Limits

Six independent development populations are reused across calibration budgets;
18 cells are not 18 independent scenes. More calibration changes sampled tails
as well as attainable p-value resolution. Pointwise selection is diagnostic and
does not establish FDR. Rejection among points within tolerance is an evaluation
quantity, not a validated conformal null. These detector checks do not establish
movement quality, full CPR-2 success, real-scene validity, or generalization.
