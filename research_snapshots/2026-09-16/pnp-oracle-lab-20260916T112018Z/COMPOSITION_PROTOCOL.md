# Experiment C — composition, fixed before execution

This experiment links program synthesis to active feature acquisition. It is NOT
full robotic control, physical dynamics identification, or general autonomous science.

For each Experiment A row with m in {8,32}, all 144 target draws, eight sample seeds,
three learners and both training regimes, retain the fitted goal function g. For
each query budget in {2,3,4}, synthesize (a) exact adaptive and (b) exact fixed-subset
sensing policies. Both use the same uniform prior over all 16 input worlds. Queries
reveal bits and the final action predicts the goal's boolean label. The planner
receives g and the public input world set only. It never receives true target f.

The evaluator replays the policy in all 16 worlds and scores its predictions against
f. It additionally scores the high-bit region x3=1, but the primary distribution is
uniform over all 16 for BOTH training regimes. Thus no prior change is secretly
introduced into the planner when training support is restricted. Experiment A's
shift test error, by contrast, is evaluated solely on x3=1.

Evaluation-only oracle: run the same planner with f. Do not count this as a deployable
algorithm. This oracle establishes the finite policy-class floor at that budget.

Primary outputs:

- actual full-world action error;
- planner's predicted action error under g;
- oracle action error under f with identical planner class/budget;
- excess action error relative to that oracle;
- error specifically on x3=1;
- explicit policy examples with target/fitted expressions and query traces.

By exhaustive optimality, each fitted policy's actual risk must be >= the matching
true-goal oracle's risk (up to exact arithmetic). At budget4 all sensors can be read;
full-domain risk then equals program prediction error. Conditional on a wrong g,
adaptive planning need NOT improve actual risk relative to nonadaptive planning.
Report this possibility without changing the populations if the sign is surprising.

Select display examples only after computation, clearly labeled as illustrative:
largest fitted-vs-oracle gap, largest actual adaptive disadvantage, and one correct
novel-expression recovery. They are not independent confirmatory samples.

No hypothesis test or confidence interval treating correlated rows as independent.
Means describe this finite test suite; target uniqueness and paired sample seeds
are disclosed separately. Raw files retain all included rows.

