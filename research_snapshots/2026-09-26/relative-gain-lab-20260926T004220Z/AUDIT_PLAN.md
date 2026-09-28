# Independent audit plan

Scope: this bounded same-candidate experiment on host `liekkas`; no new benchmark,
candidate, hyperparameter search, or replay-guided tuning is requested by this audit.
Old V28 and closeout files remain read-only. The cached scan55/65/69 results are
exposed replay, never new independent confirmation.

## Findings before execution

- The sealed closeout evaluator fixes source support using native identity rows,
  the ROI bounds, and the official observation mask. Each condition and arm uses
  those same row IDs for the primary MSE. Output-dependent crop membership does
  not enter that primary metric.
- Archived V28 training rows were sampled from each condition's source support,
  not native support. Retaining these rows preserves the frozen training data;
  threshold validation should explicitly identify which sampled rows also belong
  to native support. Reported calibration results then refer to this sample.
- Old `distance_*` diagnostic arrays are float32 distances in mm, while archived
  `A_gain` is float64 squared-error reduction in mm². Reconstructing absolute
  errors from float32 arrays can introduce rounding mismatch; recomputation from
  the saved PLY and same reference is preferable when exact gain labels matter.
- The 64 A columns are observable construction features. Names such as
  `reference_std` describe the reference camera image, not evaluator ground truth.
  Names containing `heldout` do not establish independent observations for A_all:
  the old schema states all proposal arms already use those source cameras.
- Two scene-held-out predictions can calibrate a common threshold, but the same
  predictions after threshold search are tuned development results. With only
  two scenes, numeric score calibration can also shift after the final joint fit;
  that is a limitation to report, not a reason to expand this experiment.
- Count and displacement-bin random controls must match within and outside
  native support separately. Otherwise the control can have a different number
  of evaluated edits despite matching the full-row acceptance count. They remain
  evaluator-only diagnostics because the official support is not deployable.

## Minimal independent checks

1. Check 24 development cases, scenes 24/37 only, archived row identity, finite
   64-column X, case/scene/condition weights, and gain=e0−e1 in squared-mm units.
   Verify fixed support and candidate identity for replay against sealed sources.
2. Review policy inference signatures and code: only X reaches score; same X and
   training rows reach all learned comparators; labels and replay IDs cannot enter
   inference. Verify documented additional model heads rather than claiming equal
   compute when budgets differ.
3. Independently enumerate the frozen strict-`>` threshold set from OOF scores,
   evaluate equal-case relative-MSE objective, native absolute-MSE constraint,
   +/-3 relative-MSE constraint, movement tie-break, and infeasible KEEP behavior.
4. Recompute all saved mask metrics from independent NumPy formulas. For a small
   predetermined set of replay cases, independently load saved PLY/reference and
   calculate nearest distances, verifying fixed native support and old CSV rows.
5. Check every random-control seed for support-stratified selected count and bin
   count equality; record actual movement counts/RMS because bin counts do not
   imply identical displacement. Check the fixed-A oracle is evaluation-only.
6. Verify model/threshold/source seals before versus after replay and saved output
   geometry for the predeclared exported arms.

The audit will report concrete failures that could change conclusions. Different
training support, sampled calibration, two-scene score transfer, and exposed replay
are limitations, not automatic failures when accurately described.
