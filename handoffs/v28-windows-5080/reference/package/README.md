# V28 closeout runtime

The primary method is `post_A_keep`; `post_AB_keep` is a fixed secondary arm.
This package combines frozen V28 geometry with one regressor per existing feature
signature, fitted to the archived development rows from both scenes 24 and 37.
It has no evaluator, laser/reference loader, scene path, dataset loader or GT input.
The unified model is a new frozen configuration: previous V28 cross-scene fold
scores do **not** measure this model's independent performance.

## Use without installation

Add this `package` directory to `PYTHONPATH` and use the recorded Python 3.12.13
environment and pinned dependencies in `requirements.txt`. No install was run.

```python
from v28_closeout import construct, apply_arrays

# points: finite N x 3, millimetres, fixed ROI, unchanged row order.
# reference, sources: camera arrays described below; exactly four ordered sources.
raw, features, state = construct(points, reference, sources)
selected, predicted_gain, masks = apply_arrays(
    points, raw['A_all'], raw['B_all'], features, seed=20260922
)
result = selected['post_A_keep']
secondary = selected['post_AB_keep']
```

`apply_arrays` also accepts archived feature arrays and raw candidate arrays. The
frozen column order is in `v28_closeout/FEATURE_SCHEMA.json`: 8 pre columns and 64
post columns. Predicted values are squared-error gains (mm squared), not calibrated
probabilities. Main acceptance is strictly `post_A > 0`. Secondary routing takes
the maximum of KEEP=0, A gain and B gain; exact ties prefer KEEP, then A. Random
A/KEEP preserves the main selected count using the explicitly supplied seed.
All input rows are returned, including rejected and unsupported points.

## Camera and adapter contract

Every camera is a mapping with `image` (2D grayscale float array), `P` (physical
3 x 4 projection) and `center` (3-vector, millimetres). `P` and `center` must agree.
Image values use V28's [0, 1] convention. For original RGB uint8 images of W x H,
the frozen preprocessing is Pillow bilinear resize to `(W//2, H//2)`, divide by
255, then grayscale dot product with `[.299, .587, .114]`. With `sx=(W//2)/W`
and `sy=(H//2)/H`, left-multiply the original projection by
`[[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]]`. The original Pillow version was 12.3.0;
it is an optional adapter dependency, not needed for already prepared arrays.

The adapter must establish scale, coordinate convention, ROI and the ordered
reference plus four source views from input metadata before outcome access.
The constructor freezes voxel size 1.5 mm, maximum 1000 anchors, RNG seed 20260922,
normal k=[8,24,64], 7 x 7 pixel patches, offsets [-6,6] mm in .25 mm steps,
20 surfacelet hypotheses and four interpolation neighbors. Photographs from
reserved sources 2/3 are used for all-source proposals; only `A_fit_reserved`
and `B_fit_reserved` reserve them after fit proposal construction.

The complete geometry/feature constructor works from arrays and cameras. A new
dataset's file loader, world-scale conversion, ROI rule and metadata-based view
selection still require a frozen adapter and validation. No new dataset adapter
or independent scene result is claimed by this release.

## Provenance and compatibility

`surfacelet.py` is copied from frozen V28 with only its package-relative import
changed. `direct_evidence.py` and `graph_field.py` are byte-identical copies.
`runtime.py` extracts the array computations from V28 `run_real.py` and selector
application, adding input shape checks and single-thread boundaries.
Training reads archived selected row IDs, archived gains and archived feature
matrices only. Regressor settings, row order, sample hashes and source hashes are
recorded in the enclosing `TRAINING_LOCK.json`; model metadata and hashes are
packaged in `v28_closeout/models/`. Per-model JSON is an auditable specification;
the actual estimator is the hashed joblib artifact. Load only trusted releases.
