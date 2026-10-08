"""Exact certificates over a declared FINITE hypothesis family.

No truth id, evaluation mesh, stochastic confidence, or floating point arithmetic
is accepted. Completeness of this finite family is a caller contract, not proved
by this solver. The loss concerns one fixed target on a fixed reference ray.
"""
from dataclasses import dataclass
from fractions import Fraction


def rational(value):
    if isinstance(value, float):
        raise TypeError("floats are not exact experiment inputs")
    return Fraction(value)


@dataclass(frozen=True)
class Hypothesis:
    world_id: str
    depth: Fraction
    prediction: tuple

    def __post_init__(self):
        object.__setattr__(self, "depth", rational(self.depth))
        object.__setattr__(self, "prediction", tuple(map(rational, self.prediction)))
        if not self.world_id or not self.prediction:
            raise ValueError("world id and nonempty observation are required")


@dataclass(frozen=True)
class Decision:
    status: str
    output: Fraction
    feasible_ids: tuple
    certified_gain: Fraction | None
    reason: str


def linf(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("fixed nonempty observation dimension required")
    return max(abs(rational(a) - rational(b)) for a, b in zip(left, right))


def feasible_hypotheses(observation, hypotheses, epsilon):
    observation = tuple(map(rational, observation))
    epsilon = rational(epsilon)
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    hypotheses = tuple(hypotheses)
    if not hypotheses:
        raise ValueError("declared model family must be nonempty")
    if len({h.world_id for h in hypotheses}) != len(hypotheses):
        raise ValueError("world ids must be unique")
    return tuple(h for h in hypotheses if linf(h.prediction, observation) <= epsilon)


def certify(observation, hypotheses, epsilon, incumbent, allowed_depths, margin=0):
    """Choose a permitted action only if all observation-consistent worlds gain.

    Includes KEEP independently of the supplied action pool. Ties among certified
    actions favor smaller displacement and then smaller coordinate. The exact
    minimum is a certificate for this finite family ONLY.
    """
    a, margin = rational(incumbent), rational(margin)
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    feasible = feasible_hypotheses(observation, hypotheses, epsilon)
    ids = tuple(h.world_id for h in feasible)
    if not feasible:
        return Decision("INCOMPATIBLE", a, ids, None, "no_consistent_world")
    best = None
    for b in sorted(set(map(rational, allowed_depths)) | {a}):
        if b == a:
            continue
        gain = min((a - h.depth) ** 2 - (b - h.depth) ** 2 for h in feasible)
        if gain <= margin:
            continue
        key = (gain, -abs(b - a), -b)
        if best is None or key > best[0]:
            best = (key, b, gain)
    if best is None:
        return Decision("KEEP", a, ids, Fraction(0), "no_strict_safe_action")
    return Decision("MOVE", best[1], ids, best[2], "finite_family_gain_certificate")


def minimum_residual(observation, hypotheses, incumbent, allowed_depths):
    """Same finite pool baseline: target-class min l-infinity residual.

    KEEP wins ties. This is not a port or reproduction of map-denoise full9.
    If KEEP has no model prediction it has no finite residual score.
    """
    a = rational(incumbent)
    allowed = set(map(rational, allowed_depths)) | {a}
    scored = [(linf(h.prediction, observation), h.depth) for h in hypotheses
              if h.depth in allowed]
    if not scored:
        return a
    best_loss = min(loss for loss, _ in scored)
    tied = {b for loss, b in scored if loss == best_loss}
    if a in tied:
        return a
    return min(tied, key=lambda b: (abs(b - a), b))


def encode(value):
    """Lossless JSON representation, with decimal conversion reserved for display."""
    q = rational(value)
    return {"numerator": q.numerator, "denominator": q.denominator}


def decode(value):
    return Fraction(value["numerator"], value["denominator"])
