# Visibility evidence and development search

2026-09-26, liekkas. Old sources remain frozen. Code lives here; artifacts in OUT
declared by common.py because the home partition has only about2GB free.

## Question and construction

Can current-input geometric visibility proxies explain which reserved source views
support a useful A/B correction, and improve the attainable KEEP/A/B choice?
Freeze incumbent/A/B coordinates and camera/source IDs. Do not introduce new geometry.
Rasterize CURRENT case input points in source cameras. Never use the native parent or
unperturbed scene mesh for perturbed cases. Convert projective depth to millimetres.
Distinguish behind-front support, in-front disagreement, and unsupported projection.
Use raw geometric features plus interaction with existing reserved paired photometry.
This is an occlusion proxy from the reconstruction, not an independent depth sensor;
incumbent self-support and ROI-external occluders remain limitations.

Per-candidate feature groups: existing R96; R96+raw geometry prefix; R96+all new features
including weighted paired photometry. A and B use the same formulas. Exact names and
dimensions are declared in visibility_features.py and frozen before training. Inputs
contain no scene ID, injection condition, evaluator geometry, error, or native-parent
depth. Missing geometric evidence is explicitly encoded, never certified visible.

## Bounded development search

Six learned arms: three feature sets (base, geometry, interaction) × two capacities.
Each has two separate A/B HistGradientBoosting square-error gain regressors, target
gA=e0−eA,gB=e0−eB in mm². Shallow:80iterations,7leaves,depth3,minleaf80.
Rich:160iterations,31leaves,depth6,minleaf40. Both lr=.08,l2=1,seed20260926,
early_stopping=False. Same78,598 support rows, labels and case-relative sample weights
as the previous run. All79,594 sampled rows receive OOF predictions. Two scene folds:
train24/predict37 and train37/predict24. Development labels never enter inference.

Per arm, use100 OOF score quantiles plus0 andKEEP to calibrate four policies:
- balanced: minimum equally weighted mean case-relative MSE.
- native_priority: same objective subject to mean native MSE<=identity and injected
  relativeMSE<=.95; if infeasible returnKEEP.
- recovery: minimum mean injected relativeMSE, without a native-protection constraint.
- natural: threshold0.
Tie-break by smaller moved/action fraction, then larger threshold. For each policy,
choose its development winner by that policy's objective; preserve all six arms.
Primary is development-selected balanced, not the best replay model. Export the
development-selected recovery policy too, to show best recovery under a different
declared purpose. Freeze every model/threshold/winner before replay inference.
No additional arms or threshold changes after replay outcomes.

## Measurement

Old exposed replay55/65/69,4fixedROIs,5conditions(native,±1,±3mm),60cases.
Same fixed native evaluation support, source-MSE/MAE/P95, harm/benefit>.1mm,
actual edits and KEEP/A/B counts. ROI then scene equal means; no point-pooling switch.
Compare previous independent_absolute balanced, previous joint_support balanced/
native_priority, previousA/Breserved, identity and evaluator-only A/B/AB oracles.
Ten randomized KEEP/A/B-count controls within support/non-support for primary.
They do not match actual moved counts or displacement norms.

Report actual gain and evaluator-only candidate ceiling on the same points. Net
headroom captured=(identityMSE−methodMSE)/(identityMSE−oracleMSE); can be negative.
Best replay arm per condition is a descriptive hindsight statistic, not a new algorithm.
B gross complement capture is separate from signed incremental value and overall gain.
No deployment or Git changes; old scenes are not independent confirmation.

## Verification and completion

Toy occlusion/slanted-plane/missing-support tests, row identities, candidate symmetry,
per-source scale, no clean-parent evidence, fold separation, exported coordinates and
independent reference-distance samples. CPU only, no installs or downloads. Record
wallclock and ownRSS for actual scoped stages. End with actual pointcloud outputs,
all arm results, comparison figure and concise report, whether improvement is positive
or not. Model search is intentionally concentrated on development; evaluation reports
both protection and recovery instead of maximizing one column after seeing it.
