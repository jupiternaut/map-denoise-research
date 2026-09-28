# Proposal expansion versus new surface-field coordinates

2026-09-26, before any new replay geometry evaluation. Actual inference is already
running under the unchanged PROTOCOL/source locks; this addendum changes no
constructor, score, threshold, output choice, or inference input.

Add `oracle_proposals`: the evaluator-only pointwise minimum over the initial
nine shared positions, in the frozen POOL order KEEP, A, B, half-A, half-B,
−6, −3, +3, +6 mm. It excludes K1/K2 field predictions and all selections from
them. The current input is also retained to make the tie preference explicit.

Retain `oracle_all_layers` over that initial pool, all three fitted field
positions, archived A/B and actual outputs. Also retain the old discrete and
continuous reference-oracle baselines and `oracle_expanded` (new pool plus old
continuous oracle coordinate). These are labelled GT diagnostics, not deployable
selectors. All use the same finite reference and fixed source-row support.

Two differences answer different questions:

- discrete → proposals: benefit available from searching additional ±3/6 mm and
  half-step positions, without shared surface fitting;
- proposals → all_layers: additional source-MSE potential from fitted field
  positions and their output choices, beyond the broad initial search pool.

Do not attribute the combined discrete → all_layers improvement entirely to the
multi-surface field. Actual point_wta → single_field → multi_field → visibility
→ graph comparisons remain separate and all previously specified outcomes stay.
Primary actual method remains `multi_field`. Evaluation now has 12 arms across
60 cases, yielding 720 metric rows. No new evaluation reference has been opened
to select or design this diagnostic.
