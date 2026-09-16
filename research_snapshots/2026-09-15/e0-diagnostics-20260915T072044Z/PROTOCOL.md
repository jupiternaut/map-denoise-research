# E0-D: diagnose approval, projection and structural guard separately

Written before this directory's new runs. Old E0 is exposed development evidence,
not erased or reclassified as a successful experiment. This is a bounded CPU
development study, not a blind confirmation or an end-to-end CPR-2 upgrade.

## Frozen source and execution

Source: `../e0/e0_mechanism_test.py`; old artifacts remain read-only. SHA256
manifest before/after checks all old E0 files. Python is the existing open3d-019
environment; no packages or hardware changes. CPU thread cap 2 per worker.

## A. Detector: calibration budget versus evidence discrimination

Use old generative curves, N=20000, f_big in {0.01,0.10}, new development seeds
101,102,103. Preserve each test population while using nested prefixes of one
matched (`exch`) calibration draw at n=2000,8000,32000. Generate the calibration
draw in fixed 2000-row chunks so prefixes have identical entries. Calibration
contains the same declared mixture, including wrong-association/misfit examples;
do not remove them after seeing results. Stratification follows old contrast
tertiles. Test selectors receive only s(0), observed stratum, calibration scores.

Compare unchanged BH q=.1 against pointwise p<=.1 (diagnostic, NO FDR claim).
Record all sorted-rank BH margins, rejection counts, p floors, big-error detection,
and rejection among |t*|<=.5 (evaluation ONLY), plus latency. Increasing calibration
size changes both resolution and sampled tails; do not attribute all differences
exclusively to p floors. Include deterministic small-p positive/negative controls
and the example where a later rank causes rejection although few points hit floor.

## B. Projection: separate admission from where to move

Use the same old curve family, N=20000, seeds 101,102,103, f_big .01/.10, original
2000-row calibration, both exch/sfm and alpha_tgt .2/.5. Regenerate old seed-1
cases and compare stored decision arrays before interpreting new results.

For each fixed observation, compare identity, original grid-endpoint projection,
and grid ARGMIN on EXACTLY the same eligible mask: absolute-score pass, one
sublevel interval, and incumbent 0 outside it. This bypasses BH only to test the
projection module, not as an approved full algorithm. Also report original-BH-mask
results separately. No mask or surface choice may use test truth.

Report movement count, harmful frequency over all/moved, harmful magnitude mean
and tail, coordinate MAE/RMSE, big-error repair fraction, and covered/uncovered
move errors. Original projection is discretized; finite-grid discrepancies are
not silently promoted into exact continuous projection guarantees. Report all
cells; no winner-based hyperparameter selection. Counterexample unit tests cover
projection helping, argmin repairing more, and incorrect-set harmful movement.

## C. Guard: physical reachability and a constructive alternative

First confirm old guard NN<.4 can never fire with unique XY grid spacing .8 and
z-only motion. Positive control uses coincident XY, so old guard can really fire.

Construct a local initial-gap-contraction guard using observed geometry only:
XY neighbor radius 1.5*h, initial vertical gap>=2 mm, candidate gap less than half
initial gap. Reject moved endpoints of flagged pairs; one pass, no iterative
feedback. This is an incumbent-relative heuristic, not truth recognition.

Use independently generated fixtures with h=.4,.8,1.6 and seeds 101,102,103:
real separated checkerboard layers incorrectly merged; small within-layer noise
repair; uniform rigid translation; smooth tilted plane; and an observationally
identical two-height input whose true target is a single plane (correct ghost
merge). The two latter-world labels MUST NOT enter the guard. The exact same
input/proposal for true-layer versus ghost-merge worlds must produce the same
decision; record any consequent false veto, rather than hiding the ambiguity.

Report alarms, accepted movement, source-sheet geometry error before/after,
repair retained or blocked, and output arrays. This is a guard module study with
prescribed proposals, not an autonomous surface reconstruction benchmark.

## Decision and outputs

Do not relax q or guard thresholds to obtain a win. Finish the bounded matrix
and report: which failure is reproduced; which candidate demonstrates useful
mechanism; where that same mechanism blocks legitimate correction. Old negative
system results remain valid. Do not start E1 or claim deployment readiness.

Deliver runnable modules, tests, raw per-case JSON/CSV/NPZ, a short REPORT.md,
COMMANDS.md, SOURCE_LOCK.json and CHECKPOINT.md. Distinguish development runs,
deterministic controls, reused observations and independent scenes. Test execution
success is separate from scientific success. Any post-freeze code fix must be
recorded with its cause; preserve earlier output files if they exist.
