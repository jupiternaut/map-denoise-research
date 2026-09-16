# CHECKPOINT — construction v0

Host: `liekkas`
Workdir: `/home/grf/Documents/meta-research-open-world-challenge`
Start: 2026-09-14T15:14:51Z

Old catalog checkpoints were not modified. This is a new lock.

## Done

1. Simulator: linear MSD, cubic damping, hidden memory, sensor drift,
   combined stress. Named hard case: same visible `(x,v)`, different
   drive history / hidden memory.
2. Public API: `run_experiment`, `repeat_experiment`, `calibrate`,
   `fit`, `submit_prediction`, `submit_final`. Commit-before-labels
   enforced.
3. Isolation tests pass: honest invariant; cheat caught; fake
   prediction detected; memory branches differ only on the memory plant.
4. Policies A/B = 2-D library; C/D = open constructors; ridge =
   non-symbolic residual. Construction 6 systems run (30 policy
   sessions), 24.7 s.
5. LLM-ACES / AutoSciLab official stacks not runnable (no PySR, no LLM
   API). Recorded, not renamed.

## Not done (TASK.md remaining)

- 12-system development lock
- 24-system frozen blind eval
- OS sandbox / closed-book LLM participant
- Official LLM-ACES / PySINDy baselines

## Keep / stop

Construction: adding `memory3` beats 2-D library on C2 (C−A = −0.49)
and fails the same plant at another init memory (C4). D−C ≈ 0. Ridge
wins observation drift. 20% challenge **not met**. Do not retune the
scorer on these six.

## Restore

```
PYTHONPATH=. python3 tests/test_host.py
PYTHONPATH=. python3 builder/run_construction.py
```

`outputs/construction/{ROWS,SUMMARY,REPORT}.json/md`
