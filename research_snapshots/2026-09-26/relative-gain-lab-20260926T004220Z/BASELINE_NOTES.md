# Baseline selectors: implementation declared before replay

These are local, literature-inspired mechanism baselines, **not official
implementations or reproductions of the original authors' methods**. The
comparison asks whether scoring the absolute quality of a frozen A candidate
is an effective proxy for deciding whether to replace the incumbent point.
No result from replay scenes 55/65/69 informed these definitions.

## Shared interface and fixed definitions

`baseline_models.train_policies(X, gain, e0, e1, sample_weight, seed)` returns
the seven policies below. Every policy exposes `score(X)` and
`default_threshold`; all are joblib serializable. Inference consumes only the
finite cached 64-column `A_all_post` array, without paths, scene identifiers,
conditions, labels, geometric references, or lookup access. It accepts empty
inference batches. Training rejects empty batches, inconsistent gain, negative
squared errors, and invalid weights. Single effective target classes produce
exact constant probabilities without constructing a degenerate classifier.

| Method key | Larger-means-move score | Default threshold |
|---|---|---|
| `photo_cost` | `-X[:,12]`, negative chosen photometric cost | None |
| `mode_gap` | `X[:,14]`, competing depth-valley cost minus selected cost | None |
| `source_agreement` | `X[:,20]`, source depth agreement fraction | None |
| `small_displacement` | `-abs(X[:,8])`, negative candidate offset magnitude in mm | None |
| `absolute_confidence_hgb` | Estimated `P(e1 <= 1 mm² | X)` | 0.5 |
| `candidate_error_hgb` | Negative predicted candidate squared error `-E[e1 | X]` | -1 mm² |
| `gain_sign_hgb` | Estimated `P(e0 - e1 > 0 | X)` | 0.5 |

The error regressor uses squared-error loss without target transforms or
prediction clipping. The classifier targets are explicitly inclusive for
candidate correctness (`e1 <= 1`) and strict for improvement (`gain > 0`).
Every acceptance rule is strict `score > threshold`, including the default
error threshold, which means predicted error strictly below 1 mm². The latter
default is an interpretable auxiliary operating point, not an extra optimized
threshold. There is no probability calibration stage.

Feature names and indices were checked against the sealed
`package/v28_closeout/FEATURE_SCHEMA.json` in the 2026-09-22 closeout. Meanings
were checked by reading that package's `surfacelet.py` and `runtime.py` only:

- Photometric cost is `1 - ZNCC`; lower is better.
- The mode gap is a **cost gap**, not distance between valleys. The competitor
  is a local minimum more than 0.5 mm from the chosen depth. Larger is better.
- Source agreement is the fraction of valid fitting sources whose independently
  chosen depth lies within 0.5 mm of the proposal. Larger is better.
- Cached post features are interpolated from anchors. Offset is interpolated
  in the same way as the frozen A displacement. Scalar policies retain the
  cached values and established missing sentinels (cost 1, other numeric values
  0), with no new validity mask or invented 0.5 validity cutoff. This means a
  missing mode gap or source agreement has score zero and is tied with genuine
  zero evidence. Learned policies receive all existing validity columns.

## Model and comparison budget

All three learned arms use the same 64 columns and one shallow histogram
gradient boosting estimator: learning rate 0.08, 80 iterations, at most 7 leaves,
depth at most 3, minimum 80 samples per leaf, L2 regularization 1, no early
stopping, and the supplied common random seed. Classifiers use log loss and
the error regressor uses squared-error loss. Other sklearn parameters retain
their installed defaults. `sample_weight` is passed unchanged; root controls
the common weighting scale. Scalar arms fit no parameters.

Root performs the common two-scene LOSO threshold calibration and seals the
final full-development fit before replay. The classifier target threshold of
1 mm² was specified before any results; it is not swept. The policies do not
read files or fetch features and do not use replay-specific adaptation. The
old frozen policy and identity remain separate retained arms owned by root.

## Literature grounding and limits

[Park and Yoon, *Leveraging Stereo Matching with Learning-Based Confidence
Measures*, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)
learn match correctness from a vector of confidence measures, including cost
and disparity cues, with regression forests. This supports the general
comparison of individual observable cues against learned absolute confidence.
Our classifier replaces that paper's forest, feature selection, disparity
targets, and matching-cost modulation with a single controlled HGB selector
over the existing map-candidate features. It does not reproduce their pipeline.

[Poggi and Mattoccia, *Learning to Predict Stereo Reliability Enforcing Local
Consistency of Confidence Maps*, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Poggi_Learning_to_Predict_CVPR_2017_paper.pdf)
uses a deep network to improve confidence measures through spatial consistency.
It reinforces the distinction between confidence in an estimated assignment
and the geometric benefit of replacing a particular incumbent. Our baseline
does not implement that network or its local confidence-map refinement.

Neither paper establishes that these particular cached map features, the
1 mm² nearest-neighbor target, or this selector are its method. The candidate
error regression is a local absolute-quality control, not a claimed published
baseline. The sign classifier is a local ablation of relative gain, discarding
benefit/harm magnitudes. A positive outcome against these arms would support a
matched selector-mechanism claim, not superiority over complete modern stereo,
MVS, or confidence-estimation systems. Equal single-model tree settings reduce
capacity confounding but do not equalize objective difficulty, head counts in
other arms, or end-to-end reconstruction compute. Historical replay remains
exposed evidence, not an independent generalization test.

## Local validation

On host `liekkas`, the following command passed all six synthetic unittest
cases (5.317 seconds of unittest time). It reads no archived data. The checks
cover exact scalar directions, model budget, meaningful weighted constant-X
predictions, inclusive/exclusive label boundaries, zero-weight classes,
finite scores, empty inference batches, malformed inputs, and joblib round trips.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest -v test_baseline_models.py
```

Run from `/home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z`.
This establishes the implementation contract; it does not establish the
scientific effectiveness of a selector.
