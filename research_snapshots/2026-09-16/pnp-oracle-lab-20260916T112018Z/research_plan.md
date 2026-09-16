# P=NP oracle surrogate: experience, programs and feedback

## Status and authorization

Authorized by the user's request to boldly execute the conditional construction.
Host: liekkas. New workspace: this directory. Old experiments are read-only.
The workflow skill's artifact/task-boundary tools are unavailable here; this local
plan is the equivalent artifact. No additional approval or external mutation is needed.

## Question

If finite search is solved exactly, when can an agent synthesize an unseen program
and an observation-dependent policy, and what residual errors are caused by missing
information, a restricted representation, or a tight interaction budget?

## Interpretation

This is a finite exact-optimization study, not an experimental proof of P=NP.
Search time is measured but not asserted polynomial. All library/state counts are
reported. Generalization is measured on unobserved inputs; a target-independent
grammar is distinct from a pre-enumerated list of named target programs.

## Literature takeaway

- Valiant (1984), A Theory of the Learnable:
  https://web.mit.edu/6.435/www/Valiant84.pdf
  Separate sample sufficiency from computation and representation.
- Aaronson, Computational Learning:
  https://www.scottaaronson.com/democritus/lec15.html
  Finite-class consistency yields a union-bound generalization guarantee under
  realizability and iid sampling. Our finite exhaustive learner is not that
  asymptotic theorem's efficient implementation.
- Bylander (1994), The computational complexity of propositional STRIPS planning:
  https://ai.dmi.unibas.ch/research/reading_group/bylander-aij1994.pdf
  General unbounded planning must not be silently reclassified as NP.

## Detailed Action Plan

1. Freeze this plan and protocol before running primary evaluation. Use local Python
   standard library, NumPy/Matplotlib already installed; no upstream AI4S execution.
2. Experiment A: synthesize boolean expressions on four input bits. Public grammar:
   signed input literals and constants, binary AND/OR/XOR, up to three binary gates.
   Canonicalize truth-table-equivalent formulas, retaining the smallest expression.
   Compare exact empirical-error minimization in that grammar, a restricted affine
   class, and a prefix-limited search of 32 broad-grammar candidates. If inconsistent,
   explicitly report support failure and return the lowest empirical-error candidate;
   do not report this as a consistent solution. Same deterministic tie rule throughout.
   Evaluate 48 affine, 48 nonlinear in-grammar, and 48 out-of-grammar target draws
   (duplicates allowed and unique targets disclosed), at 4/8/16/32 iid training samples,
   with eight paired sample seeds; uniform deployment over all 16 inputs. Add restricted
   training support x3=0 with deployment on x3=1, with unchanged methods. Preserve
   seen/unseen error, empirical fit, candidate checks, support status, and expressions.
3. Experiment B: exact decision-tree policies over 16 hidden four-bit worlds.
   Queries reveal one bit; final action guesses a known boolean goal. Compare optimal
   adaptive querying, optimal fixed query subset followed by the best outcome-dependent
   guess, and fixed-prefix queries with that same terminal rule. Budgets 0/1/2/3/4.
   Populations: multiplexers, affine goals, and seeded arbitrary boolean goals.
   Separately remove an informative sensor and vary query costs to test informational
   and simulated time constraints. Planner sees goal and possible worlds, not actual
   world. Exact DP must weakly dominate optimal fixed queries under the same objective.
4. Experiment C: use A's fitted programs as B's goal models. Compare exact planning
   with true goal (evaluation-only oracle), exact planning with fitted goal, and fixed
   sensing with fitted goal. This isolates what exact planning can/cannot repair after
   model estimation. Evaluate uniform and restricted-support training separately.
5. Independently test replayed policies, grammar expression truth tables, budget
   accounting, empty-support behavior, hidden-label invariance, and a deliberately
   cheating learner. A positive-control cheat must be detected. No test claims
   complete security isolation from passing behavioral checks.
6. Save raw CSV/JSON, solver counts, hashes, tests, a compact report and plots. Inspect
   every generated plot. Do not modify protocol to force positive outcomes.

## Primary estimands and decisions

- A: paired full-domain error and unobserved-input error, consistency and support
  failure; improvements attributed to search only when grammar/data are identical.
- B: uniform-world terminal error at each separately optimized budget. Define
  adaptivity gain as optimal fixed-plan error minus optimal adaptive error.
- C: actual-goal action error; separate target-model oracle from deployable methods.
- No across-domain scientific validity, physical safety, hard realtime or novel
  complexity-theory claim. Do not equate iid confirmation with mechanism transfer.
- If exact search fails with a consistent fit, inspect observational ambiguity before
  adding solver effort. If representation has no consistent fit, report that fact
  without uniquely diagnosing label corruption versus model failure.

## Skills and missing capabilities

Used: general-workflow-planner, local method-informed adaptation. The in-house
literature workflow registry is atomistic and does not supply boolean synthesis or
decision-tree solvers; those bounded modules must be implemented here. No chemical
simulation tool or pretrained language-model API is involved.

