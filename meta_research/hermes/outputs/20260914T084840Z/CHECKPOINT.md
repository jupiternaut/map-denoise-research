# CHECKPOINT

Host: `liekkas`
Workdir: `/home/grf/.hermes/attachments/outputs/20260914T084840Z`
Start: 2026-09-14T08:48:40Z

Prior checkpoint `20260914T051316Z` and original mismatch files were not
overwritten.

## Done

1. Read-only audit of previous traces: 60/60 pairs same first label at
   x=32, different at x=48; `deliver` is nearest-neighbour only.
2. `CORRECTION.md` for the three mis-statements.
3. TLA+ `RepairContract` TLC-checked: 51 distinct, empty queue, no error.
4. Experiment A: freeze history, old vs isolate-capable fitter.
5. Experiment B: share new fitter; cover / retest-then-cover / 1-step
   compete. Isolation: honest pass, cheat fail. Shortcut init fraction 0.
6. Dev 30 then eval 180 (60/cause), seeds 20000+. Protocol
   `code_sha256=de8c7225bc4e7f4295d90a226ad99129f596a63616e419606efa002ac299f558`.
7. `REPORT.md`.

## Not done

- No 2-step lookahead (1-step already no primary gain).
- No second dirt mechanism beyond the locked generator (decision was to
  stop the strategy search).
- TLC models contracts, not the numeric fitter.

## Commands

```
python3 .../smoke.py
python3 .../compete_experiment.py
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -config RepairContract.cfg -workers 4 RepairContract
```

Resume: re-run those in this directory. Do not edit
`20260914T051316Z/*` or `outputs/mismatch_experiment.py`.
