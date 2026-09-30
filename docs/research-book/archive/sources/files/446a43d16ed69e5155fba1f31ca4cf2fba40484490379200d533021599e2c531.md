# V24: repair fields and allocation at equal displacement budgets

## Question and scope, fixed before this run

Do the existing local-plane, single-scale quadratic and multiscale repair fields
have different accuracy/coverage behaviour at the same total displacement? Does
the existing alpha assign movement more usefully than uniform or shuffled weights?

Reuse exactly V22's 18 scan24 and 24 scan37 patches, cached input arrays, outputs,
alpha and evaluation support. Both scenes are now **exposed development evidence**.
No new fit, ROI, noise estimator, sensor observation, download or hidden confirmation.
No claim of a pure direction intervention, new MLS principle, physical micrometre
accuracy, topology preservation, or global optimum.

## Input-only construction

For each field D, RMS(D)=sqrt(mean_i ||D_i||^2). Three field sources:
local_plane64, quadratic64, multiscale_full. They retain differences in directions,
relative per-point amplitudes and edited support; call them repair fields.

Requested budgets are 0.025, 0.05, 0.10 mm. The actual common budget in each patch
is min(requested budget, native RMS of each of the three fields). Thus none of the
three base fields is enlarged. Output is X+B*D/RMS(D).

On the multiscale field only, also normalize alpha*D and five alpha permutations
to that same B. Permutations use fixed seeds 92401..92405 and are controls, not
five independent scenes. Their normalization CAN enlarge weighted fields; log
the factors and largest movements without claiming otherwise. These are not
the original V22 outputs. A zero/near-zero weighted field at positive B is
INFEASIBLE, never epsilon-amplified. If the base budget is zero, identity carries
no direction evidence. Repeated capped budgets are explicitly marked.

Retain all nine native cached V22 outputs unchanged, including identity, V18,
APSS2, RIMLS2 and the original two damping controls. External baselines are cached
native references, not falsely labelled budget-matched comparisons.

Probe alpha separately using fixed t=0.5 and 1.0 on the original multiscale field.
Signed gain is distance(X, reference)-distance(X+t*D, reference). No alpha in this
probe action; no absolute-gain correlation. Use rank correlation and tie-aware
alpha bins as descriptive diagnostics, not probabilistic calibration or point-
level independent significance. A ranking signal need not improve set coverage.

## Execution and evaluation

All candidate arrays and their metadata are saved and hashed before opening the
independent reference for evaluation. Raw/cached inputs and prior project files
are hashed before and after. Source snapshots have explicit scope; publication-
modified old documentation is not reinterpreted as a new historical failure.

Reuse V22's fixed scored input rows and reference IDs. Accuracy: scored output
points to the full independent DTU laser reference. Coverage: fixed local
reference points to all output points. Record MAE, P95, completeness, 1mm recall,
F-score, movement RMS/P95/max and edited point fraction. These are local metrics,
not official full-scene DTU leaderboard scores or layer-identity labels.

Summaries are patch-equal, scene-separated. Paired bootstrap intervals (2,000
resamples, seed 92424) describe within-scene patch variation only; patches may
share reference geometry. No point-level sample-count inflation. Average five
shuffle controls within each patch before comparing alpha to them.

No winner is selected per test metric; show all budgets with their corresponding
coverage. No automatic promotion of V22 or a new default. Determine whether to
retain a simpler field, pursue a demonstrated field-specific effect, or stop
the current alpha hypothesis. This is a bounded mechanism experiment, not an
obligation to complete a full model x domain x solver factorial study.

## Verification and outputs

Tests cover budget algebra, zero fields, deterministic permutations and signed
confidence diagnostics. A separate verifier reconstructs every output from the
frozen cache and re-evaluates geometry without importing the construction module.
Report any discrepancy. Deliver code, arrays, tables, compact comparison figures
and a research report; no routine advisor brief and no automatic GitHub push.
