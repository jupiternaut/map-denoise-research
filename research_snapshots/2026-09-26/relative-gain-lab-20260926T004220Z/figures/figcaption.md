# Exposed replay: fixed-A selection study

This is historical, exposed replay of scenes 55, 65 and 69, with four ROIs per
scene and five input conditions (60 cases per arm). It is not independent
confirmation, and 12 scene–ROI pairs are not 12 independent scenes.

**A.** All 11 predeclared selectors use their development-calibrated balanced
threshold, alongside identity, moving every A candidate, and the old frozen gain
policy. Each cell is 100 × (mean identity MSE − mean method MSE) / mean identity
MSE, with a denominator floor of 1e-6 mm². MSE is averaged over ROIs within each
scene and then equally over scenes. Positive values mean improvement, negative
values mean harm. The symmetric color scale includes the full observed range;
no negative value is clipped or omitted. The outlined normalized-gain row is the
predeclared exploratory lead, not a winner chosen from replay.

**B.** Fractions of the 12 matched scene–ROI cases with strictly smaller MSE than
identity, for native input and +3 mm. Rows align with panel A. Ties, including an
identity/KEEP output, are not wins. This is a descriptive fraction, not an
independent-sample significance analysis or a point-level benefit fraction.

**C.** Native and +3 mm relative MSE reductions for the four predeclared gain
constructions and the absolute-confidence comparator. Filled symbols are balanced
operating points; open symbols are the predeclared native-priority points; arrows
connect these two settings of the same method. Coincident settings show a filled
center within the open marker. The gray diamond is the old frozen policy with
its existing threshold. **All 11 native-priority constraints were infeasible on
development data, so all returned KEEP.** Their overlapping origin points are
identity outcomes with no recovery, not successful safe policies. The locked
threshold-status JSON is checked to substantiate this annotation. There is no condition-wise model or
threshold selection. Per-head capacity is fixed, but benefit/harm uses up to two
heads and hurdle gain up to three, versus one for direct and normalized gain.

The predeclared lead's displayed values are: balanced native -1.247%
and +3 mm +32.058%; native-priority native +0.000%
and +3 mm +0.000%. These values are descriptions of exposed replay.
They do not select a deployment policy or establish an external generalization
claim. Native outcomes are never averaged into injected conditions.

`all_arm_condition_summary.csv` retains every supplied arm and condition,
including natural-threshold, random-control, and oracle rows when present.
It separately reports the mean of per-case percentage gains because that
quantity differs from the ratio of equally weighted mean MSEs used in the figure.
No raw reference geometry, learned score, or decision mask is read by this script.
The locked threshold file is read solely to verify infeasible/KEEP status.

Source: `/home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z/evaluation/METRICS.csv`

Source SHA-256: `279f12739ce809c1e07cc0116f08acf8b3a0f3855657a4368ed70e3d63c4e7f7`

Figure style follows the scientific-figure-making skill: sans-serif editable
vector text, a restrained blue/red/neutral palette, minimal spines, explicit
zero references, and PNG at 300 dpi plus PDF/SVG exports.
