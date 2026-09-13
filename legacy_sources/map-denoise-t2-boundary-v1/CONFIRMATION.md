# Confirmation choice — frozen before new seeds

Specialized deliverables, selected using the development set only:

1. Fixed-normal mode: `framewise_proposal_hard`, with per-frame mixture proportions.
2. Approximate-normal mode: `plane_profile_map`, jointly estimating common plane slope.
3. Low-cost reference: `scalar_profile_hard`; not called the same optimizer as the plane mode.

Modes correspond to supplied normal confidence; do not choose a mode using ground truth or
select the best result separately on each case. Confirmation reports each frozen method separately.

The already written `suite('replication')` has eleven off-grid configurations, including
finite-plate ray visibility, a single plane, approximate normal, and frame-dependent slope.
Use new seeds 73013, 73019, 73033 (33 independent generated inputs). No method changes after
their output is inspected. This is synthetic replication, not a real-data accuracy validation.

Main metrics: paired normal MAE, full 3D point RMSE, layer-gap error, confusion and time.
Do not use raw `bias_rmse_mm` to compare tilted methods: their returned bias parameter directions
differ. Saved final geometry is transformed back into the common reference coordinates correctly.

Cheap staged normal estimation is included as a same-information baseline. Official Open3D ICP
is included. The new profile and joint implementations are in-house references, not claimed to be
official JRMPC/BALM or to exhaust current literature. Formal verification is not a completion gate.
