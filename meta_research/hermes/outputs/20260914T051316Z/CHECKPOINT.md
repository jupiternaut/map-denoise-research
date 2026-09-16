# CHECKPOINT

Host: `liekkas`. Workdir:
`/home/grf/.hermes/attachments/outputs/20260914T051316Z`

Window: start `2026-09-14T05:13:16Z` (unix 1789362796), 210 min, deadline
unix 1789375396. This file written after eval + supplements.

Prior directory `20260914T022716Z` was not used as input.
Original mismatch files were not overwritten.

## Done

1. Phase 1 (old mismatch experiment, read-only import)
   - Shared-seed alignment: until H0 refuted, random and disagreement
     execute the same queries (`frac_same=1.0` on 20 tasks/type).
     True-disagreement vs random-fallback counts in
     `PHASE1_ALIGNMENT_ROWS.json`.
   - Dual-env isolation: honest 36/36 pass; cheat 6/12 fail after E1 fix.
   - Correction: `PHASE1_CORRECTION.md` (original report kept).

2. Phase 2–4 (new conflict experiment)
   - Three hidden causes, paired same `init_obs`.
   - Three policies, shared tools, shared budget.
   - Isolation: honest 36/36, cheat 4/12 fail.
   - Dev 30 tasks → `PROTOCOL.json` (`code_sha256=4ae2977cb07299aa…`).
   - Eval 180 tasks, budgets 8/16 → `EVAL_SUMMARY.json`, `EVAL_ROWS.json`,
     `EVAL_TRAJ.json`. Distinct truths: model 60, transient 36, persistent 36.

3. Supplements (tagged, not mixed into locked ranking)
   - Worst/best cases, cost 2:1 and 1:2, extra 20 bundles, alt generator
     (`dirty_x=48`, `t=40`). `SUPPLEMENT.json`.

4. Report `REPORT.md`. Errors `ERROR_LOG.md` (E1 detector, E2 write_file,
   E3 empty sampler).

## Not done / leftover

- No TLA+ model check.
- No drop/down-weight action (named as next spend, not implemented).
- No second confirmation split after a post-eval code fix: none was needed
  after lock (sampler fix was pre-dev).

## Commands that produced the artifacts

```
python3 .../phase1_audit.py
python3 .../smoke.py
python3 .../conflict_experiment.py
python3 .../run_supplements.py
```

Resume: re-run those four files in this directory. Eval seeds `EVAL_SEED0=10000`,
dev `DEV_SEED0=1`. Do not edit original
`/home/grf/.hermes/attachments/outputs/mismatch_experiment.py`.

## Recoverable paths

- Code: `conflict_experiment.py`, `phase1_audit.py`, `run_supplements.py`, `smoke.py`
- Locked: `PROTOCOL.json`
- Results: `EVAL_SUMMARY.json`, `EVAL_ROWS.json`, `EVAL_TRAJ.json`, `DEV_ROWS.json`
- Phase 1: `PHASE1.json`, `PHASE1_ALIGNMENT_ROWS.json`, `PHASE1_CORRECTION.md`
- Supplements: `SUPPLEMENT.json`
- Narrative: `REPORT.md`, `ERROR_LOG.md`, `SESSION.json`
