# Independent protocol and implementation audit

Audit target: host `liekkas`, exact workspace
`/home/grf/Documents/Codex/2026-09-16/pnp-oracle-lab-20260916T112018Z`.
Initial review: `AGENTS.md` and `research_plan.md`, before primary results.

## Interpretation and attribution

- This finite exhaustive benchmark neither tests P=NP nor implements an assumed
  polynomial SAT oracle. Measured wall time is a property of this implementation
  and bounded problem sizes.
- Broad-grammar exact search versus its 32-candidate prefix changes search extent
  with the grammar, data, objective, and tie rule fixed. This is the designated
  comparison for attributing improvement to search. Broad versus affine search
  additionally changes representation and cannot isolate search computation.
- Exhaustive evaluation over 16 deployment inputs is an exact finite-domain risk
  for that fitted model. It includes seen inputs. Report unseen-only error with
  the number of unseen inputs; when that number is zero the error is undefined,
  not zero. Uniform and shifted training/deployment regimes are separate
  estimands.
- Repeated target draws, shared training prefixes, paired methods, and repeated
  seeds do not create independent observations for rowwise confidence intervals.
  Target duplicates and unique target counts must be disclosed. Any descriptive
  dispersion should state its unit and avoid implying population independence.
- A sample-consistent hypothesis need not equal the full target. An empty
  consistent set establishes incompatibility with the current data and class;
  it does not identify model misspecification versus label corruption on its own.
- Adaptive versus fixed sensing must use the same worlds, sensor availability,
  query budget, terminal decision rule, and objective. Fixed sensing may use all
  its observations in its final decision. With nonzero query costs, dominance is
  in the optimized objective, not necessarily terminal error alone.
- Experiment C's oracle model is an evaluation comparator. Actual target labels
  must not be supplied to fitting or to planning from a fitted model. Exact
  planning cannot recover target distinctions absent from observations/model.

## Concrete checks required

1. Re-evaluate every emitted grammar expression independently on all 16 inputs;
   verify its truth table and permitted gate count. Confirm constants and all
   affine functions are represented as intended; disclose unique library counts.
2. Independently compute training error, minimum error, sample consistency, and
   deterministic tied selection. Verify prefix search inspects only its prefix.
3. Fix training inputs and labels, mutate target labels only at unseen inputs,
   and compare fitted output. The legitimate learner must remain invariant. A
   deliberately cheating learner receiving hidden labels must change, so the
   same check detects that positive control. Passing is behavioral evidence,
   not complete process/security isolation.
4. Independently replay each policy against all deployment worlds and real goal
   labels. Every query must be available, budget-feasible, and charged exactly
   once. Verify leaf values and outcome-dependent fixed-sensing decisions.
5. Check budget-zero, no-sensor, constant-goal, XOR/parity, and multiplexer cases.
   With fixed nonnegative query costs, optimized risk cannot increase as the budget grows; adaptive
   risk cannot exceed optimal fixed risk with the same constraints.
6. For missing sensors, construct pairs of indistinguishable worlds with opposite
   labels to verify the information floor. Test an empty public world set using
   the implementation's documented behavior rather than silently normalizing it.
7. For C, hold training samples and fitted truth tables fixed, mutate unseen true
   labels, and confirm the fitted policies do not change while their evaluated
   errors can. The oracle policy is expected to depend on true labels.
8. Keep all paired comparisons paired by target, training sample, regime, and
   budget. Disclose any missing rows, failed runs, or undefined unseen errors.

## Implementation status

Initial protocol audit completed before the learning/planning modules were
present. The implementation and raw-artifact audit subsequently passed all ten
check groups. No correctness defect or hidden-target access was found in the
reviewed learning, planning, and composition call paths.

## Analytic fixtures fixed before seeing code/results

All fixtures use uniform worlds, unit query costs, and query budgets 0 through 4.

| Goal and sensing condition | Adaptive error by budget | Optimal fixed error by budget |
| --- | --- | --- |
| Constant zero | 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0 |
| Four-bit parity | 1/2, 1/2, 1/2, 1/2, 0 | 1/2, 1/2, 1/2, 1/2, 0 |
| If x0 then x1 else x2 | 1/2, 1/4, 0, 0, 0 | 1/2, 1/4, 1/4, 0, 0 |
| x3, with sensor x3 unavailable | 1/2, 1/2, 1/2, 1/2, 1/2 | 1/2, 1/2, 1/2, 1/2, 1/2 |

The multiplexer fixture specifically detects a fixed-sensing baseline that
ignores its observations or an adaptive solver that chooses the wrong first
query for its remaining horizon. The parity fixture detects accidental
communication of an unqueried bit.

## Executed checks and verified findings

Command: `python3 audit_checks.py` from the exact workspace above.
Result: **10 checks passed**, no failures, errors, or skips; 1.387 seconds on this
run. Machine-readable record: `results/audit_results.json`.

The independent checker regenerated the entire grammar through three gates using
exact-size semantic levels that retain redundant expressions, unlike production's
minimal-size pruning. Its complete mapping of truth tables to minimum gate counts
and canonical expressions matched all **3,000 programs**. A separate string parser
also evaluated every expression on every input without using production trees.
Naive weighted empirical losses, including contradictory repeated observations,
matched the optimized fitting routine and its deterministic ties.

For planning, independent reference enumeration/recursion matched 48 seeded random
cases at six budgets each, including restricted public supports, missing sensors,
and positive integer costs. Analytic constant, parity, multiplexer and missing-bit
fixtures passed. A separate interpreter checked query availability, distinctness,
cost charging, terminal predictions, and the explicit sensor callback boundary.
All three solvers rejected empty public support. The implementation's costs are
hard worst-case path budgets; therefore terminal-error dominance remains applicable
at nonunit costs, not just when costs are one.

Every saved primary artifact row was checked:

- 27,648 learning rows: sample-stream reproduction; target labels; error metrics;
  unique/duplicate observations; undefined unseen errors; candidate counts; and
  distinct representation-versus-prefix-search failure accounting.
- 3,630 planning policies and 58,080 saved execution traces: independent policy
  replay on all worlds, terminal error, query costs, trace contents, missing-sensor
  exclusion, and outcome-independent sensor sequences for fixed sensing.
- 82,944 composition rows: actual/model/oracle excess risk arithmetic, high-bit
  error, full-budget realization, paired adaptive/fixed comparisons, and unique
  row identities. Additional generated cases verified the matching oracle bound
  against independent reference planners.

Holding training data fixed and mutating every unseen label left the legitimate
learner unchanged. The same behavioral check rejected an intentionally cheating
learner that returns hidden target labels. Holding a fitted model fixed while
changing unseen true labels also left every compiled deployed policy unchanged.
The explicitly named true-goal oracle was evaluated separately. These checks
support the intended data boundary; they do not demonstrate process isolation or
exclude every possible information channel.

## Post-result attribution correction

There are **131 paired cases where the adaptive policy has greater actual error
than the fixed-subset policy**. In every one, their predicted error under the
fitted model is equal. There are **zero cases** of a strictly better predicted
adaptive objective paired with worse actual error. Consequently these rows show
the consequences of choosing among equally model-optimal policies under a wrong
model; they do not empirically establish that strict objective improvement made
actual performance worse. These counts are descriptive checks of the frozen
suite, not preregistered scientific acceptance thresholds.

The independent tests validate this bounded implementation and its saved results.
They do not validate physical transfer, a polynomial P=NP algorithm, autonomous
language-model weight learning, calibrated uncertainty outside the finite domain,
or rowwise iid statistical inference.
