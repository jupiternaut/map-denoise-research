# Post hoc oracle diagnostic: ties among model-optimal adaptive policies

This analysis was added after Experiment C showed instances in which the chosen
adaptive policy had higher actual error than the chosen fixed sensing policy.
It is **post hoc**, was not in the frozen research plan, and is evaluation-only.
The original planning and composition code and their results remain unchanged.

For each distinct `(pred_mask, target_mask, budget)` appearing in such a reversal,
we solve two exact adaptive-policy problems with lexicographic objectives:

1. Minimize learned-model error first; among all model-optimal policies, minimize
   true error.
2. Minimize learned-model error first; among all model-optimal policies, maximize
   true error.

Both terminal guesses are retained as candidates even when their model risks tie.
Stopping and querying are both considered, and every affordable query is evaluated.
Costs are one, support is all 16 worlds uniformly, and each query reveals one bit.
Dynamic programming adds model and true mistake counts across disjoint branches.
Lexicographic optimization therefore preserves the primary model optimum exactly.
The observed original adaptive error must lie inside the resulting true-error
envelope. Envelope policies are replayed independently to check both errors and
their path budgets.

This diagnostic uses true labels to select tied policies. Its best endpoint is
therefore an evaluation oracle, not a deployable improvement and not a policy
available to the original learner or planner. A tied fixed sensing policy is in
the adaptive policy class, so an actual-error optimum no worse than that fixed
policy must exist whenever primary model risks tie. Canonical reversals in that
situation cannot establish that adaptivity intrinsically damages actual outcomes.

Output files record all observed reversal pairs, distinct-case envelopes, both
endpoint policies, and summary counts. `test_tie_audit.py` independently enumerates
achievable prediction functions for every two-bit model/true-goal pair at budgets
0–2, in addition to checking original-policy inclusion and terminal ties.

Commands:

```text
python3 -m unittest -v test_tie_audit
python3 tie_audit.py --source results/composition_rows.csv --output results
```

An initial read attempted the nonexistent name `results/combined_rows.csv`, which
returned file-not-found; directory inspection identified the actual source as
`results/composition_rows.csv`. No dataset was changed by this read failure.

## Results

Both commands passed on host `liekkas`, 2026-09-16. All five tests passed in
0.063 seconds; no test or solver failure occurred. The source contains 41,472
adaptive/fixed pairings. The 131 canonical reversals reduce to 59 distinct
model/target/budget cases. All 131 have equal model error for the two chosen
policies; none pairs strictly improved model error with worse actual error.

All 131 canonical adaptive errors lie inside their exact envelope. Every case
has a model-optimal adaptive policy with actual error no greater than the chosen
fixed policy. Across these selected reversal pairs, with their original duplicate
weights retained, mean errors are:

| Policy or diagnostic endpoint | Actual error |
| --- | ---: |
| Original adaptive choice | 0.38501908396946566 |
| Original fixed-subset choice | 0.25333969465648853 |
| Best true error among model-optimal adaptive policies | 0.21421755725190839 |
| Worst true error among model-optimal adaptive policies | 0.4098282442748092 |

The best and worst endpoints are oracle quantities conditioned on the selected
reversal subset, not unbiased overall performance estimates or deployable policies.
They diagnose objective ambiguity under model error. The original deterministic
tie rule remains intact; no method was replaced or retuned using these results.

The envelope solvers visited 4,982 states across the 59 unique cases and took
0.007163 seconds; the output-producing audit took 0.258543 seconds. Raw timings
and pair/unique-case data are saved in `results/tie_audit*`.
