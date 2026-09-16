# E0-F: incumbent-local ambiguity, locked before execution

Date 2026-09-16; exact host liekkas. Previous E0-E remains read-only at
`/home/grf/Documents/Codex/2026-09-16/e0-e-selective-revision-20260915T160016Z`.

## Question and construction

E0-E blocks a move if ANY far secondary minimum is prominent. Does restricting
that competing evidence to the incumbent's neighborhood recover useful repair?
This is a new geometric decision rule, not a trained router or GPU optimization.

Keep original noisy curves, TEST p<=.1, absolute-fit check, connected level-set
requirement, alpha_target=.2, calibration size 2000 and continuous projection.
Only replace the veto. Prominence threshold stays max(.25*observed contrast,
3*calibration roughness), and separation from the observed primary minimum stays
>=2*its estimated width, exactly as E0-E.

Primary anchored veto: qualifying secondary minima must also lie within 1 mm of
incumbent displacement zero. A predeclared 2 mm radius is sensitivity only, not a
candidate selected on confirmation results. Neither radius is a universal constant:
both encode local incumbent uncertainty; the generator often places valid incumbents
near truth, making favorable matching a serious limitation.

Arms: identity, TEST projection, E0-E global veto projection, anchor1 projection,
anchor2 projection, anchor1 argmin (same mask, different endpoint). No tuning sweep.
Anchor output uses no xyz/subset/true surface/true noise/true width input.

## Experiment and accounting

Development seeds 101,102; new same-family confirmation seeds 701,702,703,704.
Each seed at f_big=.01 and .10, each with exch and sfm calibration: 24 configurations,
12 populations of 20000. Calibrations and proposals are regenerated identically.
Main claims are exch; sfm is known distribution-mismatch stress, not coverage proof.
First exact-reproduce E0-E development outputs. No changing implementation or decision
thresholds after inspecting confirmation, except logged correctness fixes and rerun
in a fresh results directory if needed.

Record own-source coordinate MAE, weighted loss ledger, point-set NN accuracy and
completeness, 1 mm precision/recall, back harm, source-layer flip proxy, actual movement,
and fraction of TEST big positive gain retained. Record both gains and losses among
points recovered by relaxing the old global veto.

Mechanism diagnostic (development only): reconstruct noiseless curves from generator
parameters with identical surfaces and zero noise. Freeze original TEST/proposal,
strata/calibration thresholds. Recompute guard features using noiseless curves.
This oracle intervention diagnoses the effect of noise on veto decisions, not an
available new sensor or a deployable denoiser. Account by source group and observed
contrast, and save original/noiseless veto flags. It does not separately identify
primary-width versus prominence versus contrast effects.

## Stress and impossibility control

12 same-evidence pairs: weaker minimum at displacement 0,.75,1.5,2.5 mm, amplitude
ratio .15,.30,.60; primary at 6 mm, 256 curves/case, observed correlated noise .03,
seed 991. Evaluate identical outputs against two hypothetical truths: secondary
location vs primary location. Both worlds have exactly the same method inputs.
This is not a physical multiview renderer. Tests whether anchor gains disappear
when an authentic alternative lies outside its radius and when a ghost has a valley.

## Upgrade gate, unchanged in substance from E0-E

Primary anchor1, confirmation exch: mean delta MAE<=0 separately at BOTH f_big;
back positive harm <=10% of old no-BH projection; >=80% of TEST positive big gain
retained. Also show every seed and geometry metrics (no cherry-picked endpoint).
If all pass, label candidate for an independent real evidence test, not deployment.
If not, default remains identity. No automatic repeated tuning or promotion of anchor2.

## Deliverables and checks

Executable observation-only operator, frozen protocol/source hashes, tests, per-case
NPZ and PLY outputs, loss ledger, mechanism and paired stress tables, report/checkpoint.
Independently recompute MAE/ledger from saved arrays; compare old hashes before/after.
Record runtime scope. No claim of new theorem, real geometry benefit, or completed
ACQUIRE action. A bounded read-only inspection may locate existing per-view evidence
for the next real test; do not silently replace missing evidence with generated views.

