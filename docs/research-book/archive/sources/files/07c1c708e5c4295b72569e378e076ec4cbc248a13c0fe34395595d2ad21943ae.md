# Continuous-step oracle: mathematical object and implementation

This note specifies an evaluation-only diagnostic over already constructed
coordinates. It makes no novelty claim and does not supply a deployable rule for
choosing a step without the evaluation reference.

## 1. Fixed inputs and loss

For input row i, write its original point as p_i and its frozen candidate endpoints
as A_i and B_i. Let G={g_1,...,g_M} be the same nonempty, finite reference point
cloud used by the historical evaluator. All coordinates use millimetres.

The per-point squared loss is

    ell(q; G) = min_{g in G} ||q - g||^2.

The support I is fixed independently of each method's output. The ROI loss is

    L(Q) = (1 / |I|) sum_{i in I} ell(q_i; G).

ROI means are averaged equally within a scene, then scene means equally across
the three scenes. This linear aggregation preserves every pointwise inequality
below. Relative gain is computed after this aggregation:

    gain(Q) = 100 * (1 - L(Q) / L(identity)).

The reference cloud and the support mask are evaluator-only inputs. The actual
observation operator is defined separately and cannot read either.

## 2. From three coordinates to two continuous segments

The historical discrete candidate set is

    C_i = {p_i, A_i, B_i}.

The new set is defined as the union of two closed segments:

    S_i = {p_i + lambda * (A_i - p_i) : lambda in [0,1]}
          union
          {p_i + lambda * (B_i - p_i) : lambda in [0,1]}.

In the actual frozen data, p_i, A_i and B_i lie on the same reference-camera ray.
Consequently this union is exactly the closed interval on that ray spanning the
three positions, and equals their convex hull. The two-branch parameterization
does not prevent traversal of the interval between A_i and B_i, or of a physical
gap between two surfaces. Not explicitly computing an A-to-B average supplies
no layer-preservation guarantee.

For general noncollinear endpoints the union is smaller than their convex hull:
it contains the two p-anchored segments, not the triangle between them. That
distinction is a property of the general kernel, not an additional safeguard in
this collinear experiment. Here the representation change adds continuous depth
positions within the interval, not new correction directions. No positions
beyond the spanning interval are permitted.

For one endpoint c_i, put d_i=c_i-p_i. For each reference point g, the nearest
position on the segment has parameter

    lambda_i(g) = clip( dot(g-p_i, d_i) / dot(d_i,d_i), 0, 1 ),

with lambda_i(g)=0 when d_i=0. Its squared residual is

    e_i(g) = ||g - p_i - lambda_i(g) * d_i||^2.

Thus the exact segment minimum is

    min_{lambda in [0,1]} min_{g in G} ||p_i + lambda*d_i - g||^2
        = min_{g in G} e_i(g).

The interchange is valid because this is a joint minimum over a compact segment
and a finite, nonempty reference set. It does not assume that the nearest
reference index stays fixed as lambda changes. Projecting only the nearest
reference at p_i would not in general solve this problem.

Compute the minimum on [p_i,A_i] and on [p_i,B_i], then take the smaller loss.
This solves the union-of-segments problem exactly in real arithmetic; the code
evaluates the same formula in float64 without a sampled lambda grid.

## 3. Endpoint inclusion gives the improvement guarantee

Because lambda=0 and lambda=1 are permitted,

    C_i subset S_i.

Consequently,

    min_{q in S_i} ell(q;G) <= min_{q in C_i} ell(q;G).

After fixed-support averaging,

    L_union <= L_discrete <= L_identity.

Therefore the continuous oracle cannot have worse source MSE than the discrete
oracle. Equivalently, its achievable MSE-reduction upper bound cannot be lower.
Strict improvement is possible, not guaranteed: it requires useful reference
proximity at an interior position not attained by an old endpoint.

These inequalities describe the oracle. A reference-free method can choose an
unhelpful interior position and does not inherit them.

## 4. Selected-branch oracle separates two unresolved errors

Freeze the previous deployed-style selector's route r_i in {KEEP,A,B}. Define
its restricted set T_i as {p_i} for KEEP, [p_i,A_i] for A, or [p_i,B_i] for B.
Let L_recovery be the old selector's output loss: lambda=1 on a selected A/B
branch and no motion on KEEP. Let L_selected be the oracle minimum restricted
to T_i. Then

    L_union <= L_selected <= L_recovery.

The existing selector's excess loss above the expanded oracle decomposes as

    L_recovery - L_union
      = (L_recovery - L_selected) + (L_selected - L_union).

Both terms are nonnegative:

1. L_recovery - L_selected: recoverable by better step length while keeping the
   current branch/KEEP decision fixed.
2. L_selected - L_union: inaccessible without changing the selected branch or
   undoing KEEP. It includes interactions between branch and best step length;
   it is not a causal estimate of a trained classifier module's contribution.

For any actual output constrained to the same T_i, an analogous decomposition is

    L_actual - L_union
      = (L_actual - L_selected) + (L_selected - L_union).

The first term now measures how much continuous step-selection error remains.
There is no general ordering between L_selected and L_discrete: one has extra
positions but restricts the branch, while the other can select all endpoints.
The representation-only extra headroom is instead

    L_discrete - L_union >= 0.

## 5. Exact candidate retrieval without a dense N-by-M array

For a single segment let m=(p+c)/2 and h=||c-p||/2. Obtain any feasible upper
bound u on its distance to G. The implementation queries reference points
nearest to p, m and c, projects those three reference points onto the segment,
and uses the smallest resulting distance as u.

Suppose a reference point g can tie or improve this bound. There is a q on the
segment with ||g-q|| <= u. Every point of the segment is at most h from m, so
the triangle inequality gives

    ||g-m|| <= ||g-q|| + ||q-m|| <= u+h.

Hence every reference point capable of attaining the global minimum lies in
the closed ball centred at m with radius h+u. Query that ball with a cKDTree,
evaluate the analytic projection for every returned reference, and take its
minimum. Reference points outside the ball cannot improve the feasible bound.

The implementation pads the radius outward before querying to accommodate
floating-point rounding. Padding can add candidates, not remove them. The
guarantee above is geometric; the software is a float64 implementation, not
an interval-arithmetic certificate over exact real numbers.

Candidate retrieval is chunked. Temporary memory depends on the number of
returned segment/reference pairs in a chunk, rather than materializing all
N*M pairs. Extremely long segments or sparse references can still make the
candidate count large; no subquadratic worst-case bound is claimed.

## 6. Code interface, ties, and checks

`segment_oracle.py` exposes

    segment_nearest(p0, p1, reference=None, *, tree=None,
                    chunk_size=2048, workers=1)

The result contains `squared_distance`, `lambda_`, `reference_index`,
`candidate_evaluations`, `max_candidates_per_segment` and `seconds`.
Coordinates must be finite arrays of shape (N,3), and the finite reference must
be nonempty. Empty source arrays are supported. A zero-length segment returns
lambda=0. On exactly tied floating-point losses the smallest lambda wins, then
the smallest reference index; this includes KEEP when lambda=0 is tied.

The candidate-evaluation counter covers retrieved-ball projections, excluding
the three initial nearest-neighbour probes per row. `seconds` is the kernel's
elapsed wall time, not end-to-end pipeline time or total memory consumption.

Tests compare against exhaustive projection of every reference, verify endpoint
monotonicity, interior optima, zero-length segments, ties, chunk independence,
unchanged inputs, empty sources and translated large-coordinate cases. Main
evaluation additionally reconstructs coordinates from lambda and rechecks their
nearest-reference distances and the union/selected-branch inequalities.

## 7. What the oracle does not optimize

The reference is a finite sampled cloud after the evaluator's crop and voxel
operations, not an exact continuous physical surface. A nearest-reference
minimum can favor the wrong physical layer, exploit sampling structure, or
move independently neighbouring points in incompatible directions.

The fixed-support source MSE is separable: a point's choice does not affect any
other point's term. That is why individual minima yield the global minimum
inside the product of these allowed sets. If the objective adds coupled surface
continuity, point-to-point exclusion, layer consistency, topology constraints,
or reverse coverage, this independent construction generally ceases to be a
global optimizer. Such metrics may improve or regress and must be measured
separately. Adding constraints cannot lower the minimum of the same source MSE
over the unchanged candidate space, though it can improve physical validity.

The output oracle uses the evaluator's knowledge. Its role is to expose whether
new positions contain useful error-reduction opportunities, not to demonstrate
that the available photographs can identify those positions or that a new
algorithm has already realized that gain.
