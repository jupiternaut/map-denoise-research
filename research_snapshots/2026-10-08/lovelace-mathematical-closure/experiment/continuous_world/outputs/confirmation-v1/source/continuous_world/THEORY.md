# Continuous depth and bounded camera uncertainty: observation enclosures

This note proves conservative observation enclosures without sampling the
continuous parameters. It extends the problem contract; it does not extend the
stage-1 finite-pool certificate to continuous worlds by interpolation.

The guarantee concerns one isolated, opaque, double-sided, constant-colour
triangle and a constant background, with the fixed parallel pinhole camera and
pixel-area convention. It does not describe the whole LOVELACE character,
unknown texture, lighting, clipping at the camera plane, or arbitrary multi-mesh
occlusion. The scalar target remains the first hit of the fixed reference +Z
unit ray. The centred M4 seed facet intersects that ray at its translation z.

## 1. Positive-Z parameter cells and exact vertex intervals

Let a cell have translation depth z in [L,U] and side-camera centre c in
[c0-eta,c0+eta], with eta >= 0. The reference camera is fixed at c=0.
For a centred world vertex (x_i,y_i,d_i), projection is

\[
u_i=(x_i-c)/(z+d_i),\qquad v_i=y_i/(z+d_i).
\]

Require **L+d_i > 0 for every vertex**. This condition holds throughout the
cell, not merely at its midpoint. A cell crossing a projection pole is outside
this enclosure contract: it must not produce a finite certified interval by
ignoring the pole. All interval endpoints and polygon operations may be
represented by exact rational numbers.

Put D_i=[L+d_i,U+d_i] and N_i=[x_i-(c0+eta),x_i-(c0-eta)]. Since D_i is positive,
the extrema of N_i/D_i occur among its four endpoint quotients. This includes
negative and sign-changing numerators. The extrema of y_i/D_i occur at the two
denominator endpoints. Thus obtain a rectangle R_i enclosing the projected
vertex for **every** z,c in the cell. The vertex rectangles need not encode the
shared-parameter correlations; dropping those correlations only enlarges the
enclosure.

Use nominal parameters z0=(L+U)/2, c=c0 and nominal vertices p_i0. Define

\[
\rho=\max_i\max\{|r_{u,i}^- -p_{u,i0}|,
 |r_{u,i}^+ -p_{u,i0}|,
 |r_{v,i}^- -p_{v,i0}|,
 |r_{v,i}^+ -p_{v,i0}|\}.
\]

Then each actual vertex p_i satisfies ||p_i-p_i0||_infinity <= rho.
The nominal parameter midpoint generally is **not** the midpoint of a projected
coordinate interval; using only the coordinate interval half-width is unsafe.
Camera centres throughout the interval are admitted, not a finite grid of
camera-centre samples.

## 2. Convex-hull Hausdorff enclosure

Let K=conv{p_1,p_2,p_3}, K0=conv{p_10,p_20,p_30}, and
B=[-rho,rho]^2. For the same convex coefficients lambda_i,

\[
\left\|\sum_i\lambda_i p_i-\sum_i\lambda_i p_{i0}\right\|_\infty
\le\sum_i\lambda_i\rho=\rho.
\]

Applying this in both directions gives

\[
K\subseteq K_0\oplus B,\qquad K_0\subseteq K\oplus B.
\]

This proof uses convex hulls, not signed triangle area or a fixed orientation.
It therefore remains valid at edge-on projections, degenerate projected
triangles, and orientation flips. With all world vertices at positive Z,
pinhole projection of the physical triangle has precisely this projected
convex hull. Positive depth makes the perspective reweighting of barycentric
coordinates positive. Double-sided material avoids a discontinuous back-face
culling rule.

Define the nominal erosion and dilation

\[
I=K_0\ominus B=\{x:x+B\subseteq K_0\},\qquad O=K_0\oplus B.
\]

In addition to K subset O, we have **I subset K**. Indeed, for every normal n,
Hausdorff enclosure implies

\[
h_K(n)\ge h_{K_0}(n)-\rho\|n\|_1.
\]

For x in I, x.n+rho||n||_1 <= h_K0(n), hence x.n <= h_K(n).
Intersecting all supporting half-planes of the compact convex set K proves
x in K. This argument also handles actual degeneracy. For a degenerate nominal
triangle and rho>0 the erosion is empty; its area lower bound is zero. If
rho=0, K=K0, including degeneracy.

### Exact polygon construction

For a nondegenerate nominal triangle, orient its vertices counterclockwise.
For edge e_i=p_(i+1)0-p_i0 choose outward normal n_i=(e_iy,-e_ix) and
b_i=n_i.p_i0. Its half-plane is n_i.x <= b_i. The erosion uses the same normals
with thresholds

\[
n_i.x\le b_i-\rho(|n_{ix}|+|n_{iy}|).
\]

Clip the nominal triangle by these three shifted half-planes, allowing an empty
or zero-area result. The dilation is the convex hull of the twelve points
p_i0+(s_x rho,s_y rho), s_x,s_y in {-1,+1}. Fraction arithmetic, exact signed
cross products, and ordinary polygon clipping suffice. A clockwise nominal
triangle must be reoriented before choosing its outward normals. Zero nominal
area requires the degenerate fallback rather than division by its area.

## 3. Pixel coverage and colour enclosures

For a fixed positive-area rectangular pixel P with area A_P, its actual
foreground fraction is alpha=area(K intersect P)/A_P. The previous inclusions
give an exact conservative enclosure

\[
\alpha^-={\operatorname{area}(I\cap P)\over A_P}
\le\alpha\le
{\operatorname{area}(O\cap P)\over A_P}=\alpha^+.
\]

Both endpoint areas are computed by exact polygon clipping; 0 <= alpha^- <=
alpha^+ <= 1. Their construction does not evaluate an arbitrary selection of
z or c values. Pixel intersection makes this substantially tighter than a
single whole-triangle area error for every pixel.

For each known constant-colour channel, F=B_colour+(G_colour-B_colour)alpha.
Evaluate this affine expression at alpha^- and alpha^+ and sort the two values.
In particular foreground darker than background reverses endpoint order.
The mixture coefficient is **pixel area coverage**, not material transparency.
Predictions remain in [0,1] if both endpoint colours do. Noisy observations need
not remain in [0,1] when the noise contract specifies unclipped addition.

Unknown material/texture cannot be inserted into this same two-colour formula
without an additional radiance enclosure. Multi-layer visibility changes also
need their own enclosure. The present one-facet contract is intentionally small.

## 4. A safe global symmetric-difference fallback

Let P0_L1 be the sum of the L1 lengths of the three nominal triangle edges
(for a degenerate triangle, its closed vertex traversal gives the corresponding
degenerate convex perimeter). For convex planar K0, square dilation has area

\[
\operatorname{area}(K_0\oplus B)
=\operatorname{area}(K_0)+\rho P_{0,L1}+4\rho^2.
\]

The L1 perimeter equals twice the sum of the horizontal and vertical widths.
This square-Minkowski formula also holds for a segment or point, using its
degenerate convex perimeter.

For a nondegenerate triangle the area lost under square erosion is at most
rho P0_L1. To see this directly, the removed set lies in the union of the three
inner edge strips. For the edge normal n_i above, strip threshold displacement
is q_i=rho||n_i||_1. Write A0=area(K0); the gap between its edge and opposite
vertex in the unnormalised linear functional is H_i=2 A0. When q_i<=H_i,
similar triangles give strip area

\[
A_0\{1-(1-q_i/H_i)^2\}\le q_i.
\]

When q_i>=H_i, the whole-triangle area A0 is also <=q_i. Summing strip areas
(overlaps only reduce the union) yields rho P0_L1. For degenerate K0 its area
loss is zero.

Since I is contained in both K and K0 and O contains both,

\[
\operatorname{area}(K\mathbin\triangle K_0)
\le\operatorname{area}(O)-\operatorname{area}(I)
\le 2\rho P_{0,L1}+4\rho^2.
\]

Thus the proposed weaker bound **2 rho P0_L1 +20 rho^2 is safe**. One way to
derive it is to bound each outward difference by its square-dilation increment
and use P_L1(K)<=P0_L1+12rho from vertex-edge perturbations. The erosion proof
above gives the tighter coefficient 4 without estimating P_L1(K).

For nominal alpha0, a safe fallback is alpha0 plus/minus
(2rho P0_L1+4rho^2)/A_P, intersected with [0,1]. Per-channel prediction radius
is this coverage radius times |G_colour-B_colour|. Prefer the actual clipped
I/O area endpoints where available.

## 5. Axis-aligned projected rectangles

For a positive-Z fronto-parallel world rectangle, enclose the four projected
edges l,r,b,t separately by exact quotient endpoint intervals. Positive depth
preserves the true ordering l<=r and b<=t.

For a pixel [p_l,p_r] x [p_b,p_t], horizontal overlap width obeys

\[
w^- =\max\{0,\min(p_r,r^-)-\max(p_l,l^+)\},\qquad
w^+ =\max\{0,\min(p_r,r^+)-\max(p_l,l^-)\}.
\]

Use the analogous vertical h^-,h^+. Then

\[
{w^-h^-\over A_P}\le\alpha\le{w^+h^+\over A_P}.
\]

These monotone overlap bounds are exact extrema over an independent edge-box
relaxation. They need not be the sharp extrema over the shared z,c parameters.
For example a fixed-width rectangle translated horizontally has correlated
edges; permitting their interval extrema independently can be more conservative.
An empty inner overlap produces zero lower coverage, never a negative width.

## 6. How enclosures support a continuous feasible-depth outer bound

For each depth/camera cell, let [F_k^-,F_k^+] enclose every predicted component.
Observation y with the hard L-infinity error budget epsilon can exclude a cell
only when some component interval is disjoint from
[y_k-epsilon,y_k+epsilon]. Closed endpoints touching remain possible.

A nonexcluded cell is **not** thereby proved jointly feasible: its per-pixel
intervals may be explained by different camera/geometry parameters. It is an
outer enclosure. Take the union of depth projections of all nonexcluded or
unresolved cells. Provided subdivision covers the original declared parameter
domain, that union contains every truly feasible target depth, including the
actual target when the model and error contract hold.

Numerical refinement, stopping-width limits, or work budgets must preserve
unresolved boxes. Deleting them would destroy coverage. The solver may use the
outer depth union or its hull for a worst squared-loss gain certificate; loose
bounds can cause KEEP but cannot justify an unsafe positive lower bound.
No cell's midpoint residual is a substitute for the full enclosure.

The stage-1 finite kappa, pairwise separation and finite-pool safety statements
do not imply a continuous kappa or coverage theorem. This continuous argument
needs the explicit enclosure and domain-covering process above. It also does
not calibrate a real camera error radius: eta is a declared assumption of the
controlled world, whose applicability to real data must be established later.

## 7. Independent acceptance cases to freeze before confirmation

- Positive denominator endpoint tests, including negative/sign-changing u
  numerators; crossing zero must reject the finite projection contract.
- Hand-computable right triangle K0={(0,0),(2,0),(0,2)}, rho=1/10:
  P0_L1=8, area(O)=71/25, area(I)=32/25; the tighter global bound is 41/25.
- Exact point/segment nominal hulls; zero lower area and finite dilation.
- Clockwise vertex order, actual edge-on projection, and orientation reversal
  must retain the same convex area interpretation.
- rho=0 returns exact nominal coverage, including pixels touching an edge.
- Pixels strictly inside I have lower coverage 1; pixels outside O have upper
  coverage 0. A partial-pixel example must bound the area, not a ray-centre test.
- Colour endpoint reversal for dark foreground; equal foreground/background
  has an exactly constant prediction despite geometric uncertainty.
- Rectangle edges l in [-1/10,1/10], r in [9/10,11/10] over pixel [0,1]
  give horizontal overlap [4/5,1]; check correlated shifts are not claimed sharp.
- Closed observation-budget tangency keeps a box; one exact rational increment
  beyond a proven disjoint bound permits exclusion.
- Refinement/work-budget exits retain unresolved boxes and preserve parameter
  domain coverage. No finite depth/camera sample is labelled a continuous proof.

Concrete world samples may supplement these cases as diagnostics, but they are
not the logical basis of the enclosure theorem. Tests verify implementation
against this proof and hand calculations; they do not replace the proof.
