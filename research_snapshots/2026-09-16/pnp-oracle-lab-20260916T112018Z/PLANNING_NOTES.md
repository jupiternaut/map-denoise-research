# Experiment B: exact finite feedback planning

## Frozen definitions before execution

The experiment has 16 possible hidden worlds, the integers 0 through 15. Sensor
`i` reveals `(world >> i) & 1`; bit `world` of the 16-bit goal mask is the desired
terminal guess. The public prior is uniform over all 16 worlds. A planner receives
the goal mask and public world support, never an actual deployment world.

The primary comparison independently solves each budget 0 through 4 for each goal:

- `adaptive`: exact memoized dynamic programming over public beliefs, remaining
  sensors and remaining budget. It can choose its next sensor from observations.
- `open_loop`: enumerate every affordable fixed sensor subset and choose the one
  minimizing terminal error. The terminal guess still depends on observations.
- `fixed_prefix`: query the affordable prefix of sensors in order 0, 1, 2, 3;
  use the same optimal response-dependent terminal majority guess.

Terminal ties favor guess zero. Adaptive stopping wins an error tie against
querying, and query ties favor the provided sensor order. Open-loop subset ties
favor fewer sensors, then combination order. All chosen open-loop sensors execute
on every supported branch. Query cost is a hard simulated path budget, not a wall
clock guarantee. Every budget receives a fresh optimization, with no cross-budget
policy reuse or assumption about the first action remaining the same.

Goal populations are all unique multiplexer functions with three distinct selector
and payload bits, with optional overall complement; all nonconstant affine Boolean
functions on four inputs including optional constant term; and 32 arbitrary
16-bit goal-mask draws from `random.Random(2026091602).randrange(65536)`, retaining
duplicates. Family counts and unique-mask counts are saved, not inferred afterward.

Two diagnostics are separate from the primary unit-cost results:

1. Remove the numerically first informative sensor for each goal, then solve at
   budget 4. Informative means flipping that bit changes the goal in at least one
   world. Constant arbitrary goals, if any, have no such diagnostic row. This rule
   is deterministic and uses the publicly known goal, not a realized world.
2. Assign sensor costs `(1, 2, 3, 4)` and independently solve budgets 0 through 4.
   Fixed prefix stops before the first unaffordable next sensor; it does not skip
   that sensor. These budgets deliberately may be less than full sensing cost 10.

## Interface and replay

Each solver has signature
`solve_METHOD(goal_mask, budget, allowed_bits=(0,1,2,3), costs=(1,1,1,1), *, support=tuple(range(16)))`.
Results include `error`, integer `error_count`, `support_size`, `policy`, operation
`counts`, and `runtime_seconds`. The optional support is uniform and must contain
distinct valid worlds; empty support raises `ValueError`, never reports zero risk.
Budgets are nonnegative integers and query costs are strictly positive integers.

Policies are JSON-compatible trees: leaves `{kind: "guess", value: 0 or 1}`;
queries `{kind: "query", bit: i, cost: c, zero: ..., one: ...}`. Public supports
smaller than the full domain can yield unreachable `{kind: "unsupported"}` leaves
in fixed policies; attempting to execute such a leaf raises `ValueError`.
`replay(policy, world)` returns a terminal guess. `replay_trace(policy, world)`
returns the guess and query records. `replay_with_sensor(policy, read_bit)` has no
world argument and executes only explicit sensor reads. No replay function accepts
a goal mask; evaluation of the true goal happens outside policy execution.

## Outputs and limitations

`results/planning_rows.csv` stores one solver result per goal/case/budget/method.
`planning_goals.json` stores every goal draw and generation metadata.
`planning_policies.jsonl` stores every full policy.
`planning_traces.csv` stores all 16 evaluation-world executions per result.
`planning_summary.json` records counts, elapsed times, and family means.
Operation counts have method-specific meanings and are not comparable units of
work. Runtime is descriptive Python execution time on this host, not evidence of
polynomial complexity or real-world realtime control. The finite optimum certifies
this explicit prior, goal, action set and budget only; it is not a P=NP algorithm.

Run from this directory:

```text
python3 -m unittest -v test_planning
python3 planning.py --output results
```

## Execution record and findings

Both commands above completed successfully on host `liekkas` on 2026-09-16.
The test suite passed all 11 tests in 0.227 seconds, with no observed failures.
It independently enumerated every achievable adaptive prediction function for
all 256 targets on three input bits at budgets 0–3, and enumerated fixed sensing
subsets and terminal assignments for all 16 two-bit targets under nonunit costs.
Other checks covered all catalog goals, replayed errors, allowed sensors, path
costs, response-dependent fixed-policy decisions, empty support, and equal action
and trace for worlds indistinguishable under the actual queries. These behavioral
checks do not constitute complete security isolation.

The run produced 3,630 policy-result rows and 58,080 world traces. All 110 goal
entries are unique within their families: 48 multiplexers, 30 affine functions,
32 arbitrary functions. All replayed errors match solver error counts; adaptive
error never exceeds optimal fixed-subset error, and optimal fixed-subset error
never exceeds fixed-prefix error under the same support, costs, and budget.

At primary budget 2, mean errors (adaptive / optimal fixed / fixed prefix) are:

| Family | Adaptive | Optimal fixed subset | Fixed prefix |
| --- | ---: | ---: | ---: |
| Multiplexer | 0 | 0.25 | 0.291667 |
| Affine | 0.166667 | 0.166667 | 0.4 |
| Arbitrary | 0.1953125 | 0.2265625 | 0.30859375 |

For affine targets adaptive and optimal fixed errors coincide at every primary
budget. For arbitrary targets their mean adaptivity gain is 0.03125 at budget 2
and 0.05078125 at budget 3. All methods reach zero error at unit-cost budget 4.

With the designated informative sensor removed, even budget 4 leaves errors of
0.25 for multiplexers, 0.5 for affine functions, and 0.2578125 for arbitrary
functions; all three methods reach that same information-limited floor. With
costs `(1,2,3,4)` at budget 4, adaptive error is respectively 0.229167, 0.3,
and 0.265625, so the original numeric budget no longer buys full observation.

The run's summed solver times were 0.048129 seconds adaptive, 0.023339 seconds
optimal fixed subsets, and 0.008001 seconds fixed prefix; full output-producing
runtime was 0.344218 seconds. Adaptive optimization visited 27,830 states and
evaluated 29,150 query candidates across fresh solves; optimal fixed sensing
evaluated 8,140 subsets. These tiny finite-problem timings support no asymptotic
complexity or hardware-deadline claim. Exact raw timings and counts are in the
summary JSON.
