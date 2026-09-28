# Four fixed constructions over the same A candidate

This file declares the constructions before exposed replay scoring. The lead is
`normalized_gain`; there is no additional exploratory arm. These are executable
learning-objective/factorization changes to a fixed candidate selector, not a new
geometric candidate or a claim of a new theoretical guarantee.

Let `e0` be the identity squared geometric error, `e1` the frozen-A squared
geometric error, and `g = e0 - e1`, all in mm². The only inference input is the
same cached 64-dimensional observable feature array `X`. Training error labels
come only from the development scenes. Neither scene/condition IDs nor reference
geometry are policy inputs. The policy function returns larger scores for rows
more worth moving. All four natural thresholds are zero, using strict `score >
threshold`. Calibrated thresholds are root's common, predeclared scene-held-out
procedure; this module never selects a threshold.

| Name | Target / factorization | Returned score | Maximum fitted HGB heads |
| --- | --- | --- | ---: |
| `direct_gain` | Regress `g` | predicted signed gain | 1 |
| `normalized_gain` | Regress `g / (e0 + e1 + 0.01)` | predicted normalized gain | 1 |
| `benefit_harm` | Regress `max(g,0)` and `max(-g,0)` on all rows | nonnegative predicted benefit minus nonnegative predicted harm | 2 |
| `hurdle_gain` | Classify `g>0`; regress `g` on positive rows and `-g` on nonpositive rows | `p*m_plus - (1-p)*m_minus` | 3 |

## Why these might help, and why they might fail

`direct_gain` is a control for the old signed-gain objective with the same
per-head capacity and feature input. It tests whether a benefit comes from the
objective/factorization rather than a larger individual model. A zero threshold
on a well-estimated conditional mean gain is aligned with expected squared-error
reduction. Its squared training loss can nevertheless be dominated by large
gain magnitudes in displaced inputs.

`normalized_gain` is the predeclared lead. Its epsilon is fixed at **0.01 mm²**,
not estimated from either development or replay errors. For consistent squared
errors its target lies in [-1,1], suppressing the dominance of large absolute
gains and representing the proportionate tradeoff between retaining and moving a
point. This could make native-input, smaller-scale damage more visible to a
fixed-capacity regressor while retaining useful signs on injected cases. The
epsilon also reduces the importance of tiny changes when both errors are nearly
zero. The fitted regression score is not clipped: the bounded training target
does not imply a strictly bounded boosted prediction. This target estimates an
expectation of a ratio, not a ratio of expected errors and not expected absolute
gain. It may thus reduce MSE less, or lose large-error recovery. Common
development calibration and separately reported native/injected outcomes are
needed to assess that tradeoff.

`benefit_harm` makes the two nonnegative terms explicit. A shallow model may fit
rare harms differently from frequent benefits when fitted in separate heads;
this might reveal native risks hidden by a signed target. In ideal conditional
expectation estimation, their difference equals expected gain. In a finite model
it need not match direct regression. Predictions of each nonnegative magnitude
are clipped below at zero before subtraction. It receives twice the per-head
tree budget of `direct_gain`, so any empirical gain is not an equal-total-budget
proof that factorization itself is superior.

`hurdle_gain` separates how often an A move helps from how much it helps or harms.
The probability model can learn a sign boundary while conditional regressors
learn the two magnitude scales. Native damage might be captured by a low benefit
probability or a large predicted harm even when unconditional errors mix scales.
For the identity `E[g|X] = p E[g|g>0,X] - (1-p) E[-g|g<=0,X]`, zero-gain rows must
join the nonpositive branch; the implementation does so. Each magnitude is
clipped below at zero. Conditional class scarcity and extrapolation can make
this factorization unstable, and the probability estimate is not independently
calibrated. It has three heads, not the single-head control's total budget.

## Fixed capacity, weighting, and edge cases

All nonconstant heads use scikit-learn HistGradientBoosting with learning rate
0.08, 80 iterations, at most 7 leaves, depth at most 3, minimum 80 rows per leaf,
L2 regularization 1, and early stopping disabled. Regression uses squared error;
classification uses log loss. All heads receive the same supplied seed. There
is no condition-specific head or hyperparameter search. The implementation uses
the existing environment and NumPy/scikit-learn only.

Sample weights are accepted, passed without rescaling, and restricted to the
appropriate conditional subset. Rows with zero weight are excluded from fitting
each head. Root defines case/scene/condition balancing; global weight scale can
affect HGB regularization and must remain common across methods. A head with a
constant target, or fewer than 160 positive-weight rows (too few to split with
the fixed minimum leaf size), returns its weighted target mean. An absent
conditional magnitude returns zero. A single-class or unsplittable classifier
returns its weighted positive fraction. These conventions handle all-benefit,
all-harm, all-zero, tiny-class, and zero-weight-class cases without changing the
declared model family. An entirely empty or zero-weight training set is rejected.

Input labels/weights must be finite and squared errors nonnegative. Feature NaNs
are allowed through HGB's native missing-value behavior; infinite features are
rejected. Feature count must match fitting. Empty inference batches return empty
scores, and nonfinite outputs raise an explicit error instead of being silently
turned into favorable decisions. Every policy is a module-level dataclass with
serializable estimators and exposes only `score(X)` for inference.

## Verification boundary

`test_innovation_models.py` uses synthetic 64-feature arrays only. It checks the
fixed objective formulas in constant cases, weighting, sign-class edge cases,
HGB parameter consistency, deterministic fitted scores, missing features, empty
inference, and joblib round trips. These are implementation checks, not evidence
that the constructions improve maps. Native improvement, injected recovery,
movement/harm statistics, and compute cost must come from the root's sealed
development/replay experiment. Historical scenes 55/65/69 remain exposed replay;
an independent confirmation is still absent.

Synthetic verification completed on liekkas with the mandated CPU environment:
`python -B -m unittest -v test_innovation_models`, with OMP/OpenBLAS/MKL threads
all fixed to one and bytecode disabled. All four tests passed in 23.879 seconds.
The environment reported scikit-learn 1.9.0, NumPy 2.2.6, and joblib 1.6.0.
