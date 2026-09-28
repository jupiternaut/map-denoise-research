# Current-input visibility evidence

For each frozen p/A/B row, use the same z-buffer built only from this condition's
current input p. Injected cases never use the clean native parent as an occluder.
Camera projection is exactly the half-resolution convention used by the archived
paired photometry. Camera depth is the homogeneous denominator divided by the
norm of P's third-row spatial coefficients, giving millimetres rather than an
arbitrary projection-matrix scale.

The z-buffer covers only the current ROI point set, not the full scene. It can miss
occluders outside that support. Input points anchor their own pixel unless a
nearer input point wins it. This is explicitly retained in the self-front-owner
feature, not presented as independent depth or true visibility. It intentionally
uses the same self-supported map for the incumbent and both candidates.

Each candidate gets 96 float32 features. The first 80 are four views ×20 geometric
features; views are sorted, for each incumbent row, by source/reference ray
parallax sine. Sorting is shared by A/B. Within a view: old/new projection-valid,
center-support-known, inverse-depth-plane-fit-known, center signed depth gap,
plane signed depth gap, plane residual, 3x3 occupied-pixel fraction, log1p center
point count, self-front-owner; then parallax sine and candidate pixel movement.
Feature names are exported in `visibility_features.py`.

The plane fits inverse depth against the actual subpixel UV of nearest input
points in the 3x3 pixel neighborhood. Perspective planes are affine in inverse
depth. This detrends smooth tilted planes rather than labeling all depth span as
an occlusion edge. Fewer than three noncollinear front samples, or nonpositive
predicted inverse depth, is an unknown plane. Gaps and residuals are zero with
explicit missing flags, never treated as known free space.

The final 16 features interact old archived reserved-view ZNCC with geometric
support. For each fixed tolerance 0.5 and 2 mm, the weight is
`exp(-max(old_gap,new_gap,0)/tolerance)` on paired-valid views with geometric
support. Plane gap is used when available, otherwise center gap. Four weighted
margins, weighted mean margin, paired-valid fraction, mean weight and weighted
win fraction are returned. Missing support/photo pairs contribute no weight.
These are features, not a hard visibility veto; tolerances are not tuned on replay.

No new source views, no GT, no error labels, and no condition names enter this
module. The camera adapter opens calibration/scale files only, never native mesh.
Input p/A/B geometry is unchanged. Five tests cover front/back signs, tilted-plane
detrending, behind-camera/empty support, self anchoring/depth units, A/B exchange,
all-missing photo neutrality and nonmutation. Stage seals bind all sources.
