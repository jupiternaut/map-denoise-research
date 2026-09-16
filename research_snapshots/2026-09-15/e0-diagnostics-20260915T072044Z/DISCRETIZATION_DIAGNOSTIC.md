# Post-E0-D projection follow-up (not a changed primary protocol)

The first projection run is complete and preserved in results/projection. Its
eligible-no-BH arm has covered-but-harmed cases, despite the exact continuous
interval projection theorem. The source evaluates coverage using linear
interpolation of scores but projects to the nearest accepted GRID SAMPLE. These
are different feasible sets. The discrepancy was anticipated as possible in
PROTOCOL.md; it was not a reason to overwrite the original run.

Before running the follow-up, fix this comparison: reconstruct endpoints of the
same piecewise-linear score sublevel interval; project 0 to those interpolated
endpoints, leaving the admission mask, calibration, curves and score threshold
unchanged. Use exactly the 24 existing development cases. This tests a numerical
implementation issue, not extra geometric information or a new statistical law.

Predictions: covered harmful moves should vanish up to 1e-9 numerical tolerance;
grid-to-interpolated movement differences should not exceed .05 mm. The total
error may improve or worsen outside coverage; report both and do not change
thresholds to improve the aggregate. Save follow-up arrays separately under
results/boundary. Do not replace projection.py output or call this independent
confirmation. Add a deterministic non-grid-aligned endpoint test.
