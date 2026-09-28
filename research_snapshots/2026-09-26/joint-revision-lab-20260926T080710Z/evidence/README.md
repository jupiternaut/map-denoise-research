# Fixed-coordinate support evidence

The extractor reads frozen p/A/B geometry and the sealed original/reserved view
lists, calibration and images. No evaluator geometry, distance arrays or labels
enter the evidence path. R is reserved from the local support choice; it shares
the reference image and the upstream reconstruction with F, so it is not a claim
of statistical independence or visibility estimation.

For each B coordinate the original-input k8/k24/k64/fronto normal bank and five
7x7 full/half footprints give twenty hypotheses. The four original source scores
at B choose the lowest-cost hypothesis, averaging the best three finite ZNCC
scores and requiring at least two. Ties within 1e-12 use the lowest hypothesis.
The chosen h is then held fixed for both p and B on all four reserved sources.
No F choice gives h=-1 and all-NaN pairs, reduced to neutral costs, zero margins
and explicit zero-validity flags by the historical summarize_pairs function.

Each case's SUPPORT.npz stores row_ids, A[N,32], B[N,32] and signed chosen_h[N].
A is an exact copy of the sealed h5 PCA24/full R summary. B is the new R summary.
The duplicate A feature block and newly fitted B hypothesis are intentional;
this is not a symmetric increase in independent observations.

Extraction uses chunks of at most 4096 rows and two positions, p and B. The full
twenty-hypothesis score bank is reduced immediately and never written to disk.
All B h5 source scores are compared with the previous full-footprint extraction.
Sixteen deterministic rows per case independently check A h5 source scores and
original normal-bank parity. Metadata records the actual discrepancies and row
IDs, frozen geometry hashes, row counts, chosen-h counts, missingness and runtime.
Each scene is independently sealed after successful extraction.

Run from this workspace on liekkas with the existing environment, using -B and
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1:

    /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest -v test_support
    /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B extract_support.py --scene 24

The same command takes scenes 37, 55, 65 and 69. Existing output directories
cause an error; completed evidence is not overwritten. Development uses exactly
the previous 79,594 sampled row IDs; replay uses every archived row.
