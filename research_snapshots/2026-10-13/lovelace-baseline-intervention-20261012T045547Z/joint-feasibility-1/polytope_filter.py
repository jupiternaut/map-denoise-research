"""Exact-proof filter for the frozen box and necessary linear rows.

``classify(rows, bounds, deadline=None)`` accepts upper/lower rows of exact
rationals and finite rational box endpoints. ``deadline`` is an absolute
``time.monotonic()`` deadline. One single-threaded HiGHS phase-I solve, limited
to at most five seconds and the remaining deadline, supplies candidates only.
No box endpoint is changed by this module.

A deletion needs nonnegative rational weights on the original row indices.
Lower rows are negated, giving A*x <= b. If a = lambda*A and B = lambda*b,
the exact box minimum of a*x must be strictly greater than B. In particular,
a need not vanish. A feasible witness is an exact point satisfying every row
and every endpoint. It certifies only this necessary linear relaxation.

Rationals in returned JSON use {numerator: int, denominator: positive int}.
Invalid exact inputs raise TypeError/ValueError. LP errors, timeouts and
unverified candidates return UNKNOWN; they never certify an empty set.
"""

from fractions import Fraction
import math
import os
import time
import warnings

# Set these before importing NumPy/SciPy. The command invoking the complete
# workflow must also set them, in case those libraries were imported earlier.
for _thread_environment in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_environment] = "1"

import numpy as np
from scipy.optimize import OptimizeWarning, linprog


DEFAULT_LP_SECONDS = 5.0
WEIGHT_DENOMINATOR = 1_000_000_000
WITNESS_DENOMINATOR_LIMIT = 1_000_000_000
STATUSES = ("CERTIFIED_INFEASIBLE", "FEASIBLE_WITNESS", "UNKNOWN")
_CERTIFICATE_FIELDS = {
    "sparse_weights", "aggregate_coefficients", "weighted_bound", "box_min", "margin"
}


def _exact(value, name="rational"):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be exact, not bool")
    if isinstance(value, (Fraction, int)):
        return Fraction(value)
    if isinstance(value, dict):
        if set(value) != {"numerator", "denominator"}:
            raise ValueError(f"{name} requires numerator and denominator only")
        numerator, denominator = value["numerator"], value["denominator"]
        if any(not isinstance(item, int) or isinstance(item, bool)
               for item in (numerator, denominator)):
            raise TypeError(f"{name} requires integer numerator and denominator")
        if denominator <= 0:
            raise ValueError(f"{name} denominator must be positive")
        return Fraction(numerator, denominator)
    raise TypeError(f"{name} must be Fraction, int, or an encoded rational")


def encode(value):
    value = _exact(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _prepare(rows, bounds):
    box = []
    for pair in bounds:
        pair = tuple(pair)
        if len(pair) != 2:
            raise ValueError("each variable bound must have two endpoints")
        lo, hi = (_exact(value, "box endpoint") for value in pair)
        if lo > hi:
            raise ValueError("box lower endpoints must not exceed upper endpoints")
        box.append((lo, hi))
    dimension = len(box)
    normalized = []
    for row in rows:
        row = tuple(row)
        if len(row) != 3:
            raise ValueError("rows require kind, coefficients, bound")
        kind, coefficients, bound = row
        if kind not in ("upper", "lower"):
            raise ValueError("row kind must be upper or lower")
        coefficients = tuple(_exact(value, "coefficient") for value in coefficients)
        if len(coefficients) != dimension:
            raise ValueError("coefficient dimension must match bounds")
        bound = _exact(bound, "row bound")
        if kind == "lower":
            coefficients, bound = tuple(-value for value in coefficients), -bound
        normalized.append((coefficients, bound))
    return tuple(normalized), tuple(box)


def _dot(coefficients, point):
    return sum((a * x for a, x in zip(coefficients, point) if a), Fraction(0))


def _box_min(coefficients, box):
    return sum((a * (lo if a >= 0 else hi)
                for a, (lo, hi) in zip(coefficients, box) if a), Fraction(0))


def _aggregate(normalized, dimension, weights):
    coefficients = [Fraction(0)] * dimension
    weighted_bound = Fraction(0)
    for row_index, weight in weights:
        row, bound = normalized[row_index]
        weighted_bound += weight * bound
        for variable, coefficient in enumerate(row):
            if coefficient:
                coefficients[variable] += weight * coefficient
    return tuple(coefficients), weighted_bound


def _make_certificate(normalized, box, weights):
    coefficients, weighted_bound = _aggregate(normalized, len(box), weights)
    minimum = _box_min(coefficients, box)
    margin = minimum - weighted_bound
    if margin <= 0:
        return None
    return {
        "sparse_weights": [{"row": index, "weight": encode(weight)}
                           for index, weight in weights],
        "aggregate_coefficients": [encode(value) for value in coefficients],
        "weighted_bound": encode(weighted_bound),
        "box_min": encode(minimum),
        "margin": encode(margin),
    }


def certificate_check(rows, bounds, certificate):
    """Check a weighted-box contradiction using only exact input arithmetic.

    This checker rejects malformed, duplicate-index, negative-weight, tampered
    aggregate and non-strict certificates. It trusts neither an LP status nor
    certificate-supplied row/box copies. An independent workflow verifier must
    reconstruct the public rows and bounds separately.
    """
    try:
        normalized, box = _prepare(rows, bounds)
        if not isinstance(certificate, dict) or set(certificate) != _CERTIFICATE_FIELDS:
            return False
        if not isinstance(certificate["sparse_weights"], list):
            return False
        weights, seen = [], set()
        for entry in certificate["sparse_weights"]:
            if not isinstance(entry, dict) or set(entry) != {"row", "weight"}:
                return False
            index = entry["row"]
            if (not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(normalized) or index in seen):
                return False
            weight = _exact(entry["weight"], "certificate weight")
            if weight < 0:
                return False
            seen.add(index)
            weights.append((index, weight))
        claimed = certificate["aggregate_coefficients"]
        if not isinstance(claimed, list) or len(claimed) != len(box):
            return False
        claimed = tuple(_exact(value, "aggregate coefficient") for value in claimed)
        coefficients, weighted_bound = _aggregate(normalized, len(box), weights)
        minimum = _box_min(coefficients, box)
        margin = minimum - weighted_bound
        return (margin > 0 and claimed == coefficients
                and _exact(certificate["weighted_bound"]) == weighted_bound
                and _exact(certificate["box_min"]) == minimum
                and _exact(certificate["margin"]) == margin)
    except (TypeError, ValueError, ZeroDivisionError, KeyError, OverflowError):
        return False


def _point_check(normalized, box, point):
    return (len(point) == len(box)
            and all(lo <= value <= hi for value, (lo, hi) in zip(point, box))
            and all(_dot(coefficients, point) <= bound
                    for coefficients, bound in normalized))


def witness_check(rows, bounds, witness):
    """Verify an exact point against the original rows and the entire box."""
    try:
        normalized, box = _prepare(rows, bounds)
        if not isinstance(witness, (list, tuple)):
            return False
        point = tuple(_exact(value, "witness coordinate") for value in witness)
        return _point_check(normalized, box, point)
    except (TypeError, ValueError, ZeroDivisionError, KeyError, OverflowError):
        return False


def _active_point(normalized, box, matrix, right_hand_side, floating_point, seed):
    """Reconstruct selected active equations with exact rational elimination.

    Floating residuals choose equations only. Every resulting point is checked
    against all exact inequalities before it may become a witness. Free
    variables retain the rational seed; a full-rank selection is a vertex.
    There is no second LP, extra depth split, or variable-range optimization.
    """
    dimension = len(box)
    equations = []
    for variable, (lo, hi) in enumerate(box):
        coefficients = [Fraction(0)] * dimension
        coefficients[variable] = Fraction(1)
        coefficients = tuple(coefficients)
        if lo == hi:
            equations.append((-1.0, variable, coefficients, lo))
            continue
        for side, endpoint in enumerate((lo, hi)):
            residual = abs(floating_point[variable] - float(endpoint))
            scale = max(1.0, abs(float(endpoint)), abs(floating_point[variable]))
            relative = residual / scale
            if relative <= 1e-7:
                equations.append((relative, 2 * variable + side, coefficients, endpoint))
    for index, (coefficients, bound) in enumerate(normalized):
        products = matrix[index] * floating_point
        residual = abs(float(np.sum(products)) - right_hand_side[index])
        scale = max(1.0, abs(right_hand_side[index]), float(np.sum(np.abs(products))))
        relative = residual / scale
        if math.isfinite(relative) and relative <= 1e-7:
            equations.append((relative, 2 * dimension + index, coefficients, bound))
    equations.sort(key=lambda entry: (entry[0], entry[1]))
    basis = {}
    for _, _, coefficients, bound in equations:
        coefficients = list(coefficients)
        for pivot in sorted(basis):
            factor = coefficients[pivot]
            if factor:
                old_coefficients, old_bound = basis[pivot]
                coefficients = [a - factor * old
                                for a, old in zip(coefficients, old_coefficients)]
                bound -= factor * old_bound
        pivot = next((index for index, value in enumerate(coefficients) if value), None)
        if pivot is None:
            # Contradictory selected equations do not prove the original set
            # empty: near-active selection can include a non-active equation.
            continue
        divisor = coefficients[pivot]
        basis[pivot] = (tuple(value / divisor for value in coefficients), bound / divisor)
        if len(basis) == dimension:
            break
    point = list(seed)
    for pivot in sorted(basis, reverse=True):
        coefficients, bound = basis[pivot]
        point[pivot] = bound - sum((coefficients[index] * point[index]
                                   for index in range(pivot + 1, dimension)
                                   if coefficients[index]), Fraction(0))
    return tuple(point)


def classify(rows, bounds, deadline=None):
    """Return a JSON-safe three-state proof record for the current exact box.

    Fixed keys are status, certificate, witness, lp_status, seconds,
    lp_seconds, single_row_contradiction, single_row_contradiction_rows,
    row_count, dimension and reason. Unused certificate/witness are null.
    lp_status is SciPy's integer code, or null when no LP ran. The single-row
    diagnostic is computed exactly on this box, including signed/zero rows.
    """
    started = time.monotonic()
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise TypeError("deadline must be an absolute monotonic number")
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
    rows, bounds = tuple(rows), tuple(bounds)
    normalized, box = _prepare(rows, bounds)
    contradictory = [index for index, (coefficients, bound) in enumerate(normalized)
                     if _box_min(coefficients, box) > bound]
    result = {
        "status": "UNKNOWN", "certificate": None, "witness": None,
        "lp_status": None, "seconds": 0.0, "lp_seconds": 0.0,
        "single_row_contradiction": bool(contradictory),
        "single_row_contradiction_rows": contradictory,
        "row_count": len(normalized), "dimension": len(box), "reason": "no_exact_proof",
    }

    def finish(reason, status="UNKNOWN", *, certificate=None, witness=None):
        result.update(status=status, reason=reason, certificate=certificate, witness=witness,
                      seconds=max(0.0, time.monotonic() - started))
        return result

    def expired():
        return deadline is not None and time.monotonic() >= deadline

    if expired():
        return finish("deadline")
    if contradictory:
        certificate = _make_certificate(normalized, box, [(contradictory[0], Fraction(1))])
        if certificate_check(rows, bounds, certificate):
            return finish("exact_single_row", "CERTIFIED_INFEASIBLE", certificate=certificate)
        return finish("single_row_certificate_failed")
    if not normalized or not box:
        point = tuple((lo + hi) / 2 for lo, hi in box)
        witness = [encode(value) for value in point]
        if witness_check(rows, bounds, witness):
            return finish("exact_trivial_witness", "FEASIBLE_WITNESS", witness=witness)
        return finish("trivial_witness_failed")

    try:
        matrix = np.asarray([[float(value) for value in coefficients]
                             for coefficients, _ in normalized], dtype=float)
        right_hand_side = np.asarray([float(bound) for _, bound in normalized], dtype=float)
        floating_bounds = [(float(lo), float(hi)) for lo, hi in box]
        if (not np.isfinite(matrix).all() or not np.isfinite(right_hand_side).all()
                or not all(math.isfinite(value) for pair in floating_bounds for value in pair)):
            return finish("nonfinite_lp_input")
    except (OverflowError, ValueError):
        return finish("nonfinite_lp_input")
    phase_matrix = np.column_stack((matrix, -np.ones(len(normalized))))
    objective = np.zeros(len(box) + 1)
    objective[-1] = 1.0
    lp_started = time.monotonic()
    time_limit = DEFAULT_LP_SECONDS if deadline is None else min(
        DEFAULT_LP_SECONDS, deadline - lp_started)
    if time_limit <= 0:
        return finish("deadline")
    try:
        with warnings.catch_warnings():
            # SciPy 1.15.3 forwards this native HiGHS option verbatim. The
            # installed HighsOptions has been checked to support threads=1.
            warnings.filterwarnings("ignore", category=OptimizeWarning,
                                    message=r"Unrecognized options detected:.*threads")
            lp = linprog(objective, A_ub=phase_matrix, b_ub=right_hand_side,
                         bounds=floating_bounds + [(0.0, None)], method="highs",
                         options={"threads": 1, "time_limit": time_limit})
    except Exception as error:
        result["lp_seconds"] = max(0.0, time.monotonic() - lp_started)
        return finish(f"lp_exception:{type(error).__name__}")
    result["lp_seconds"] = max(0.0, time.monotonic() - lp_started)
    status = getattr(lp, "status", None)
    if isinstance(status, (int, np.integer)) and not isinstance(status, (bool, np.bool_)):
        result["lp_status"] = int(status)
    if expired():
        return finish("deadline")
    if result["lp_status"] != 0 or not bool(getattr(lp, "success", False)):
        return finish("lp_not_successful")
    try:
        floating_point = np.asarray(lp.x, dtype=float)
        marginals = np.asarray(lp.ineqlin.marginals, dtype=float)
        objective_value = float(lp.fun)
        if (floating_point.shape != (len(box) + 1,) or marginals.shape != (len(normalized),)
                or not np.isfinite(floating_point).all() or not np.isfinite(marginals).all()
                or not math.isfinite(objective_value)):
            return finish("nonfinite_or_malformed_lp_result")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return finish("nonfinite_or_malformed_lp_result")

    # Fixed denominator, chosen before any data run. The exact box calculation
    # alone decides whether this quantized candidate establishes a contradiction.
    weights = []
    for index, marginal in enumerate(marginals):
        scaled = max(0.0, -float(marginal)) * WEIGHT_DENOMINATOR
        if not math.isfinite(scaled):
            return finish("nonfinite_weight_candidate")
        numerator = round(scaled)
        if numerator:
            weights.append((index, Fraction(numerator, WEIGHT_DENOMINATOR)))
    certificate = _make_certificate(normalized, box, weights)
    if expired():
        return finish("deadline")
    if certificate is not None and certificate_check(rows, bounds, certificate):
        return finish("exact_weighted_box_certificate", "CERTIFIED_INFEASIBLE",
                      certificate=certificate)

    point = tuple(min(hi, max(lo, Fraction(float(value)).limit_denominator(
        WITNESS_DENOMINATOR_LIMIT)))
                  for value, (lo, hi) in zip(floating_point[:-1], box))
    witness = [encode(value) for value in point]
    if witness_check(rows, bounds, witness):
        return finish("exact_rational_witness", "FEASIBLE_WITNESS", witness=witness)
    if expired():
        return finish("deadline")
    try:
        point = _active_point(normalized, box, matrix, right_hand_side,
                              floating_point[:-1], point)
    except (OverflowError, ValueError, FloatingPointError):
        return finish("active_reconstruction_failed")
    if expired():
        return finish("deadline")
    witness = [encode(value) for value in point]
    if witness_check(rows, bounds, witness):
        return finish("exact_active_witness", "FEASIBLE_WITNESS", witness=witness)
    return finish("no_exact_proof")
