# CHECKPOINT

Host: `liekkas`
Workdir: `/home/grf/.hermes/attachments/outputs/20260914T111013Z`
Start: 2026-09-14T11:10:13Z

Prior `20260914T084840Z` and `20260914T051316Z` were not overwritten.
No soft weights.

## Done

1. TLA+ split `orig`/`cur`. Claims in `TLA_CLAIMS.md`.
   TLC: 139 generated, 51 distinct, 0 on queue, depth 11, no error.
2. Numeric cleanup: orig vs cur, competing live rows, normalized
   lookahead as supplement only.
3. Frozen isolate-cap-1 fitter. Mini domain N=8.
4. Exact DP over 18 catalog worlds (4703 states, 0.42 s).
5. Cover / retest-then-cover / optimal on catalog and holdout.
6. Isolation: honest 27/27, cheat 1/9 fail.
7. `REPORT.md`.

## Not done

- Soft weights (intentionally deferred).
- Larger catalog / other dirt families.
- 2-step lookahead as a primary policy.

## Commands

```
python3 .../smoke.py
python3 .../exact_experiment.py
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -config RepairContract.cfg -workers 4 RepairContract
```

Protocol `code_sha256=792dd99cb0405c505bd62b421dc9ebd6890c1a8578a403b3b250c3527627d6bb`.
