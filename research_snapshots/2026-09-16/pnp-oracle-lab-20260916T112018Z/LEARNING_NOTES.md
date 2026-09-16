# Experiment A: exact finite expression synthesis

Executed on host `liekkas` in this new workspace. Protocol SHA256 remained
`c06d34af5ba8637f19565d46cd6d91b36376b608697870f3a5b46516a97842f0`.
No thresholds, populations, grammar, or seeds were revised after evaluation.
This is an exact finite search experiment; its exhaustive enumeration is not a
polynomial-time SAT oracle and supplies no evidence that P=NP.

## Reproduction and API

Run `python -m unittest test_learning -v`, then `python learning.py` from this
directory. Python standard library only; no network, paid API, dependencies, or GPU.
The first command passed 9 tests, including independent expression replay,
independent exhaustive canonicality through 2 gates, all affine programs,
known-expression synthesis, iid multiplicity, naive weighted ERM agreement,
candidate-order-invariant ties, immutability, and hidden-label invariance with a
deliberately cheating positive control that the same audit rejects.

`build_library()` returns an immutable tuple of immutable `Program` objects, with
`mask`, `gates`, `expression`, and recursive executable `tree` fields.
`fit(samples, candidates)` accepts only `(input, label)` pairs and public programs.
It returns immutable `FitResult`, including `.program`, `.train_errors`,
`.sample_count`, `.consistent`, `.candidate_checks`, and `.train_error`.
The fitter has no target argument. `Expr.evaluate(x)` evaluates the syntax tree
independently of its cached semantic mask.

Inputs are 0..15; `xj=(input>>j)&1`. A truth mask's bit at position `input` is its
output. Expressions are fully parenthesized, with `!xj` signed literals and
`&`, `|`, `^` binary operators. The gate bound counts nodes in an expression tree,
including repeated subexpressions; this is not a DAG-size circuit bound.

Canonicalization minimizes gate count first, then expression string, sorting the
two children lexicographically because each operation is commutative. Maintaining
only minimal semantic representatives in each size level is complete: a larger
child could always be replaced by an equivalent smaller child. All methods use
this same tie rule. Prefix32 is the first 32 candidates in this canonical order;
its restricted depth/composition is part of the declared finite search budget.

## Scale and seeds

The exact grammar has **3,000 distinct truth tables**. Minimum gate counts 0/1/2/3
contain 10/60/456/2,474 programs. All 32 affine functions are included.
The constructor made 10/165/1,800/19,170 attempts by gate count, with
0/105/1,344/16,696 attempts duplicating a new or previously represented semantic
function. These count attempts after semantic child pruning and commutative
symmetry reduction, not every unreduced syntactic expression.

The affine/nonlinear/out-of-grammar populations contain 32/2,968/62,536 distinct
functions. Target seed `20260916` draws 48 times per family with replacement;
unique observed targets are **27/47/48** (21/1/0 duplicate draws). Targets retain
their original draw weights in summaries; no deduplication changes the sample.

Sample base seeds are `9100..9107`. Per target/regime/rep, the seed is the first
8 SHA256 bytes, big endian, of `learning-v1|{base_seed}|{target_id}|{regime}`.
Python `Random.choice` draws 32 iid inputs with replacement; sizes 4/8/16/32 use
nested prefixes. Each method gets exactly the same observations at each size.
Uniform draws use inputs 0..15; shifted draws use x3=0, inputs 0..7. Seeds and all
32 labeled observations are saved for each of the 2,304 sampling streams.

The run produced **27,648 method rows** from 9,216 training sets and checked
**28,237,824 candidates**. The first run took 2.6844 seconds including output,
with 0.00969 seconds for library construction. These are single-run local timings,
not asymptotic or cross-machine performance claims.

## Endpoint results

Mean deployment error at m=32, 384 rows per cell (48 target draws × 8 replicates):

| Training/deployment | Target family | Exact grammar | Affine ERM | Prefix32 |
|---|---|---:|---:|---:|
| Uniform/uniform | Affine | 0.00000 | 0.00000 | 0.23893 |
| Uniform/uniform | Nonlinear in grammar | 0.01921 | 0.20605 | 0.20085 |
| Uniform/uniform | Outside grammar | 0.14323 | 0.27018 | 0.30892 |
| x3=0/x3=1 | Affine | 0.50000 | 0.50000 | 0.49089 |
| x3=0/x3=1 | Nonlinear in grammar | 0.43294 | 0.46810 | 0.43620 |
| x3=0/x3=1 | Outside grammar | 0.50326 | 0.47656 | 0.50326 |

For exact ERM with uniform training, m=32 error on inputs never observed in that
particular training set is 0.00000/0.12602/0.49155 across the three families.
These means have 354/344/353 defined rows: when all 16 inputs have appeared,
unseen error is undefined and excluded, with denominators saved explicitly.
Full-domain error includes previously seen inputs. In particular, its apparent
improvement for arbitrary out-of-grammar targets is not evidence of unseen-input
generalization: the corresponding unseen error remains near one half.

All exact fits to affine and nonlinear in-grammar targets are consistent.
Outside the grammar, exact fits at m=32 are consistent in 11.979% of uniform rows
and 86.979% of shifted rows. This is observational compatibility on the training
sample, not full-domain representability of those targets.

For uniform nonlinear targets at m=32, prefix32 is inconsistent in 99.479% of
rows while the full grammar is consistent in every row. Those failures are
classified `search_budget_exhausted`, not representation support failure. When
the full grammar itself has positive empirical error, the status is
`representation_support_failure`; affine support failure refers to its own
representation. This diagnosis identifies absence of a consistent program in
the stated class, not a unique causal distinction between label corruption and
model misspecification in an unknown real-world problem.

## Saved artifacts and interpretation

- `results/learning_rows.csv`: every fitted mask/expression, sample regime,
  training/deployment/full-domain/seen/unseen error, support status, separate
  budget/representation flags, sample multiplicity, checks and fit seconds.
- `results/learning_catalog.json`: all canonical programs, target draws and
  masks, seed rules, class sizes and construction counts.
- `results/learning_samples.json`: complete labeled sampling streams; take the
  first m observations to reconstruct each training set.
- `results/learning_summary.json`: all family/regime/size/method means and
  undefined-unseen-error denominators.

The learner can synthesize an unnamed program from a compositional grammar. For
the independently specified target `(x0 AND x1) XOR (x2 OR x3)`, full labeled input
coverage recovered truth mask 30584, with canonical equivalent expression
`((!x0|!x1)^(!x2&!x3))`. This checks actual semantic synthesis, not retrieval by
target name. It does not demonstrate general natural-language understanding or
autonomous LLM improvement.

Comparing exact with prefix32 isolates search budget under the same grammar and
data, with prefix ordering disclosed. Comparing exact with affine also changes
the representation and inductive bias. The shift condition shows that additional
examples confined to x3=0 cannot establish behavior on x3=1 without further
assumptions, even when exact empirical optimization succeeds.
