# Independent figure and interpretation review

The predeclared B_R balanced arm does not meet the joint protection and repair
goal. It improves all four shifted conditions but increases native MSE by 4.15%.
Its native MSE is higher in all 12 ROIs and all three scenes. Retain this result
as mechanism evidence; it is not a deployment upgrade.

## Fixed candidate × evidence comparison

Values below are MSE reduction versus identity (%), with each scene the mean of
its four ROI metrics, then the three scenes equally weighted. Positive is better.
The percentage is calculated from these aggregated MSE values, not by averaging
point-level ratios. No condition or arm was selected after viewing these outcomes.

| Balanced arm | Native | −1 mm | +1 mm | −3 mm | +3 mm |
|---|---:|---:|---:|---:|---:|
| A_F | −5.8255 | −0.4253 | 7.8951 | 42.1849 | 34.7475 |
| A_R | −4.5163 | 1.3857 | 8.5459 | 46.7730 | 37.1202 |
| B_F | −5.4229 | −0.1378 | 7.9847 | 41.9418 | 34.5219 |
| **B_R (primary)** | **−4.1505** | **1.7681** | **8.3680** | **46.4489** | **36.3575** |

Reserved sources improve the balanced gain relative to fitted sources for both
candidates in every condition. Changing A to B does not systematically increase
this benefit. The interaction `(B_R−B_F)−(A_R−A_F)` is −0.0368, +0.0949,
−0.2674, −0.0810 and −0.5372 percentage points, respectively. These are descriptive
contrasts on three exposed scenes, not statistical equivalence tests or evidence
of a positive interaction. B_R versus A_R is mixed: B_R has less native damage
and more −1 mm repair, but less gain in the remaining three conditions.

Native identity MSE is 0.7531651 mm²; B_R produces 0.7844254 mm². B_R moves 3.2081%
of all input rows; 1.9276% of fixed native-support rows are harmed by more than
0.1 mm and 0.5388% benefit by more than 0.1 mm. Movement and harm have different
denominators and are not directly subtractable. Native scene gains are −2.9162%,
−3.3148% and −6.0396% for scenes 55, 65 and 69. Native is not assumed to consist
entirely of valid geometry, but the declared native error target is still missed.

## Candidate availability and selector capture

| Evaluator-only diagnostic | Native | −1 mm | +1 mm | −3 mm | +3 mm |
|---|---:|---:|---:|---:|---:|
| A / KEEP oracle gain (%) | 23.8482 | 30.6446 | 34.6009 | 63.2565 | 56.2928 |
| B / KEEP oracle gain (%) | 23.0320 | 29.5991 | 34.5098 | 64.0140 | 56.4930 |
| A / B / KEEP oracle gain (%) | 32.6020 | 38.8729 | 43.8971 | 69.9427 | 63.2670 |
| B increment beyond A / KEEP (gain pp) | 8.7537 | 8.2282 | 9.2962 | 6.6862 | 6.9741 |
| B_R capture of that increment (%) | 3.3431 | 12.3019 | 7.9104 | 28.4046 | 21.6400 |

B contributes complementary useful coordinates despite similar standalone A and B
oracle gains. The increment is the loss reduction from adding B to the pointwise
best of A and identity. The capture row is the fraction of this incremental
benefit lying on rows selected by B_R. It is a gross beneficial contribution;
it excludes harmful B selections elsewhere and is not a net deployed gain or the
performance of an A/B router. Native capture contributes only 0.2926 gain pp,
while B_R's total native gain is −4.1505%.

Thus candidate absence alone cannot explain the result: useful B coordinates
exist under the tested fixed-row loss, but the tested features, model and threshold
do not reliably exploit them while protecting native geometry. This does not
establish that the oracle is attainable by an available observation-only selector.
No A/B score-max router was deployed. The oracles are bounded diagnostics over
these frozen candidates, not physical-layer, completeness or topology guarantees.

## Retained failures and controls

Ungated B has gains of −136.0313%, −108.5614%, −56.8884%, +27.7392% and +20.7400%.
The B_R fixed paired-margin gate also damages native and both 1 mm conditions
(−45.1027%, −35.0827%, −10.1200%). Reserving evidence does not itself make a
paired-margin rule a reliable native-protection decision.

B_R balanced exceeds every count-matched and displacement-bin-matched random
replicate on all four repair conditions. On native, its −4.1505% gain is within
the count-matched range [−4.2699%, −3.9767%], while better than the bin-matched
range [−5.2668%, −4.9732%]. Random repetitions are diagnostic controls, not
additional scenes or a basis for point-count confidence intervals.

The predeclared B_R native_priority policy nearly abstains on native: 0.00542%
of input rows move, with a −0.00570% native gain. It gives +0.8266%, +0.8304%,
+29.4733% and +20.8895% repair gains. This is a protection/repair tradeoff;
the small native regression must not be rounded to exact preservation or used to
replace the predeclared primary. Natural-zero B_R instead damages native by
15.3447%, despite stronger gains for the two 3 mm shifts.

## Verification and figure scope

`plot_results.py` independently recomputed the plotted values from `METRICS.csv`:
1,495 comparisons with sealed `SUMMARY.json` passed with zero numeric difference.
It checked complete matching memberships (60 cases per arm), finite metrics,
oracle order, and all three balanced interaction terms. A further 175 independent
checks of `COMPLEMENT.json` case → scene → aggregate means and metric identities
passed with maximum absolute difference 1.60e−14. Input/output SHA-256 checks in
`figures/MANIFEST.json` were rechecked after rendering.

The single four-panel figure was visually inspected: labels, annotations and
legend are readable, and no clipping or overlap was observed. PNG is 4680×3090
at 300 dpi; PDF is vector with embedded TrueType text; SVG contains 108 editable
text elements. The scientific-figure-making skill supplied the print palette,
font hierarchy, minimal axes and vector export conventions. The map-denoise-research
skill informed separation of candidate potential, actual edits and deployable
selection. No uncertainty intervals are inferred from points or repeated ROIs.
All results remain exposed replay rather than independent confirmation.
