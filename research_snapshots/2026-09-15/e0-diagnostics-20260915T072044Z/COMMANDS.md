# Execution record

Host: liekkas. Directory: `/home/grf/Documents/Codex/2026-09-15/e0-diagnostics-20260915T072044Z`.

Before new experiments: old E0 file hashes captured in SOURCE_LOCK.json.
New PROTOCOL.md SHA256: `5931225783037bf06d747b2b28619b7e6bc1b6db1433c554af2de63c70f8334f`.
Old E0 source SHA256: `b3d69fe9be611fb872182f8af7287c5e4ac27b5bf415b9cbd0802cc9e2537fac`.

Interpreter: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`.
All workers set `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`.
No GPU, installation or external write requested or performed.

Actual test/run commands and outcomes will be appended after execution. Programs
write experiment artifacts only under this directory. This directory is not a
git worktree, and no commit or upload is part of the task.

## Actual completed runs

Prefix every Python command below with the environment and interpreter above;
the working directory was always this exact directory.

1. `python -B -m unittest discover -s tests -p test_projection.py -v`: exit0,
   original six tests passed before the full projection run.
2. `python -B projection.py`: exit0;8oldseed-1comparisons and24newconfigurations;
   script-internal wall11.9655s. No earlier projection-result file overwritten.
3. Detector worker: `python -B -m unittest discover -s tests -p test_detector.py -v`
   exit0,7tests; `python -B detector.py` exit0,18cells, internal wall9.1083s.
   Full command notes under results/detector/COMMAND_NOTES.md.
4. Guard worker: `python -B -m unittest discover -s tests -p test_guards.py -v`
   exit0,10tests; `python -B guards.py` exit0,54cases, internal wall.39585s.
   Full command notes under results/guards/RUN_NOTES.md.
5. After projection-grid discrepancy was observed, wrote the separate
   DISCRETIZATION_DIAGNOSTIC.md and projection_boundary.py, added one targeted
   test. `python -B -m unittest discover -s tests -p test_projection.py -v`:
   exit0,7tests. `python -B projection_boundary.py`: exit0,24samecases,
   internal wall7.0338s; separately stored results/boundary.
6. `python -B -m unittest discover -s tests -v`: exit0,24tests.
7. `python -B verify.py`: exit0;87oldfiles unchanged/noadditions;
   144projectionoverallblocks,216guardblocks,24boundarycases checked.

All three experiment branches ran concurrently with disjoint output ownership.
No full experiment crashed or needed a results-driven parameter change. A
detector RNG multiplier typo and a guard test-example arithmetic issue were
corrected before their first experimental runs; worker notes preserve these.
Later label/summary wording corrections did not change raw numeric results.

Internal times exclude portions of interpreter startup and are not end-to-end
development time, combined wall-clock speedup or peak-memory measurements.
