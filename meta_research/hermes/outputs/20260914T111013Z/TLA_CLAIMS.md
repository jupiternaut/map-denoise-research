# TLA+ claims (checked vs not)

TLC run: `RepairContract.tla` with `RepairContract.cfg` on a 3-point tape,
`MaxC=2`, `MaxClock=8`. Report the footer (generated / distinct / queue)
in CHECKPOINT. This is not a proof about the numeric fitter, N=8
libraries, or research strategies.

## Checked (INVARIANT Safety)

- `TypeOK`
- `IsolateSubsetOrig`: isolate ⊆ originally recorded points
- `OrigIntegrity`: `orig[x]` is None or the first hidden write (no
  retest rewrite of `orig`)
- `CurOnlyIfOrig`: `cur` is defined only where `orig` is defined

## Not checked, do not claim

- “Records are append-only” in the previous spec was false: `Retest`
  overwrote the only tape. The new spec splits `orig` / `cur`.
- Policy cannot read hidden labels: `PolicyPick` is a `CHOOSE` on
  `orig` slots; Hidden is a constant. TLC does not quantify over
  cheating policies.
- Isolate is free: no cost variable change. Not a claim that isolate
  is free in the numeric experiment (there it is also unbilled,
  `ASSUMED`).
- `MaxClock` is an artifact so isolate/unisolate cannot stutter forever.
- No liveness. No numeric MAE. No “optimal diagnosis”.
