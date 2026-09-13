# Transport outer-budget diagnostic V3

Frozen before executing this diagnostic. This uses exposed development cases,
not reserved seeds or independent confirmation. No frozen estimator source is edited.

## Question and scope

The source-translation update permits at most `2 sigma * 0.8 = 1.6 mm`
per outer iteration at sigma 1 mm. With unchanged supported components and their
weighted zero-mean gauge, three iterations from zero therefore permit at most
4.8 mm per scan while injected bias RMS 4 mm can have larger extrema. Component
splits/merges can change the centering gauge, so this is not an unconditional
whole-program bound. Save component histories and actual bias increments.
This bound is not a proof that every error is caused by insufficient iterations.
Test whether more outer iterations improve output, and whether UOT remains better
or worse than balanced at the SAME outer/Sinkhorn budget.

## Frozen inputs and configurations

- Exact host: liekkas. Base input root:
  `/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/repair-v2-oyuie4pl/synthetics/identifiable`.
- Seed 912101, bias RMS 4 mm: ghost, dual gap 2/4/8 mm.
- Each base case: full and the existing `metrics.sampling_indices(..., 'partial_overlap')`
  observed-coordinate subset. Eight input variants from four original worlds,
  not eight independent scenes. Do not regenerate noise or tune sampling.
- sigma 1 mm; balanced/unbalanced; outer iterations 3, 6, 12; Sinkhorn iterations
  24 throughout. All other parameters and graph/local_only remain unchanged.
- Forty-eight correction calls, each followed by the identical graph/local_only
  filter; save both stages, giving 96 outputs. Do not select configurations by GT.
- Six sequential independent configuration worker processes. A worker reads legal XYZ/scan
  arrays only and receives sigma/configuration, not evaluation metadata, normals,
  layer labels or injected offsets. Change only its in-memory outer-iteration
  parameter, restoring the original dictionary in `finally`.

## Evidence and scoring

- Parent process alone uses the existing `evaluate_v2.synthetic_geometry` and
  `metrics.structure_metrics`, preserving all metrics and failures.
- Save original-world raw-correction and postfilter outputs without alignment;
  existing global-translation-only aligned diagnostic remains secondary.
- Save supplied parameters, per-scan biases, mass, marginal discrepancies,
  outer history and source/point counts in output JSON; report bias RMS and maxima.
- No deletion or rearrangement of points; check shape/finiteness/input immutability.
- Transport-only wall time and postfilter time are separately recorded, plus
  their full-pipeline sum. Worker import/startup, total wall time and parent versus
  child peak RSS are separately labelled. Single-thread CPU, no bytecode writes.
- One fixed tiny observed-coordinate warmup per worker, excluded from timed method
  calls and reported separately. It does not select parameters or use these cases' GT.
- Save protocol/source snapshots and SHA256 manifests before the first worker;
  verify frozen source and original input/evaluation files after all workers.
- Each run uses a new `transport-budget-v3-` temporary directory under the existing
  runs root. Never overwrite exploration-v3-lnx049yd or any historical run.

## Interpretation fixed in advance

Report 3 -> 6 -> 12 changes per input as well as paired aggregates, including
regressions and layer/tilt diagnostics. Do not treat tiny ordering differences as
general superiority. Additional iteration budget is additional compute: no naked
comparison against original fast or other baseline timing/quality is a fair claim.
Sinkhorn stays at 24, so nonzero balanced row residuals remain an inner-solver
limitation. UOT marginal deviation is permitted by its objective and is not a
convergence residual. More outer iterations do not establish either OT objective
is solved exactly. No conclusions about independent real geometry are possible.
