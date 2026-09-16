# E0-D checkpoint

Status: bounded implementation and development matrix COMPLETE; not deployed.

Host/path: liekkas:/home/grf/Documents/Codex/2026-09-15/e0-diagnostics-20260915T072044Z

Read REPORT.md first, then PROTOCOL.md. Old ../e0 is frozen and its 87 files
are unchanged; SOURCE_LOCK.json and VERIFICATION.json supply evidence.

## Implemented

- detector.py: calibration-budget diagnostic, exact full-rank BH checks.
- projection.py: matched-support original projection/ARGMIN comparison and old replay.
- projection_boundary.py: piecewise-linear boundary interpolation repair.
- guards.py: initial-gap-contraction candidate plus identical-input two-world tests.
- verify.py: stored-array metrics, tests and source hash check.

## Results

- A: 18 cells, BH zero in17; an8k exception with76rejections disappears at32k.
- B: 24 configurations, projection better than ARGMIN in24, identity in3 only.
- Boundary: covered damage events348→0, same24exposed configurations.
- C: new guard blocks3600badmerges but also3600correctghostrepairs; normal10800moves retained.
- 24/24 unit tests pass. Source and stored-array verification pass.

## Next authorized boundary

No E1, new dataset, install, GPU job or publish has started. No task background
process remains. The bounded three-module diagnosis is finished; further work
would be a new evidence/score or reference-observation experiment, not retuning
this matrix. Do not treat this synthetic result as real-map confirmation.

## Preservation

Do not overwrite results. Follow-up numerical results are separately stored;
DISCRETIZATION_DIAGNOSTIC.md explains why they were added after the first run.
Reporting-only corrections are documented under detector and guards outputs.
