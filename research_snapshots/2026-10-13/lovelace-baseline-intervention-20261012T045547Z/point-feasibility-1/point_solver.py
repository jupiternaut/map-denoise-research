"""Frozen necessary linear relaxation at an exact public depth value.

Depth is fixed by passing lo=hi to the A4-V interval geometry. The public
camera radius still encloses pose uncertainty; this module never selects pose
offsets or invokes a point renderer. A rational witness establishes only this
necessary depth-point relaxation, not full image-model feasibility.

Inputs are the existing public observation record and the existing bounded
arm record. A same-leaf problem must identify an original retained trace node
in that arm. It preserves the original material bounds and, for joint, its
bias bounds. Calibrated rows always marginalize the arm's frozen calibration.
The A5 classifier, certificate checker and witness checker are reused intact.
"""

from fractions import Fraction as Q
from pathlib import Path
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
VARIATION_ROOT = ROOT.parent / "appearance-variation-1"
POLYTOPE_ROOT = ROOT.parent / "joint-feasibility-1"
for _source_root in (VARIATION_ROOT, POLYTOPE_ROOT):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

import variation_solver as variation
import polytope_filter as polytope


for _module, _expected_file in ((variation, VARIATION_ROOT / "variation_solver.py"),
                                (polytope, POLYTOPE_ROOT / "polytope_filter.py"),
                                (variation.geometry, ROOT.parent / "recovery-1" / "baseline_geometry.py")):
    if Path(_module.__file__).resolve() != _expected_file.resolve():
        raise RuntimeError(f"wrong frozen source identity: {_module.__file__}")

geometry = variation.geometry
original, contractor = variation, polytope
BRANCHES, METHODS = variation.BRANCHES, variation.METHODS
DRIFT_BOUND = Q(1, 10)
BASELINE = Q(80)
RETAINED_STATUSES = ("retained_tolerance", "unresolved_budget")
CLASSIFICATION_STATUSES = polytope.STATUSES

# These are the frozen functions themselves, not replacement implementations.
classify = polytope.classify
certificate_check = polytope.certificate_check
witness_check = polytope.witness_check
encode = variation.encode


def exact(value, name="exact value"):
    """Decode the frozen exact scalar/public rational schema, including strings."""
    return variation._exact(value, name)


_exact = exact


def _intervals(values, count, name, *, unit=False):
    intervals = tuple(tuple(_exact(value, name) for value in pair) for pair in values)
    if len(intervals) != count or any(len(pair) != 2 for pair in intervals):
        raise ValueError(f"{name} requires {count} pairs of exact endpoints")
    if any(not 0 <= lo <= hi or (unit and hi > 1) for lo, hi in intervals):
        raise ValueError(f"{name} requires ordered nonnegative intervals"
                         + (" within [0,1]" if unit else ""))
    return intervals


def _retained_node(arm, branch, method, depth, node):
    if not isinstance(node, dict):
        raise TypeError("node must be an original retained trace record")
    if node.get("status") not in RETAINED_STATUSES:
        raise ValueError("node status must be retained_tolerance or unresolved_budget")
    index = node.get("id")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("node id must be a nonnegative exact integer")
    branch_record = arm["branches"][branch]
    if branch_record.get("mode", method) != method:
        raise ValueError("node arm method does not match the requested method")
    matches = [entry for entry in branch_record["trace"] if entry.get("id") == index]
    if (index not in branch_record["retained_ids"] or len(matches) != 1
            or matches[0] != node):
        raise ValueError("node must match the original retained record in the requested branch")
    lo, hi = _exact(node["lo"], "node depth lo"), _exact(node["hi"], "node depth hi")
    if not lo <= depth <= hi:
        raise ValueError("exact depth must lie in the original closed leaf interval")
    return node


def build_problem(public, arm, method, branch, depth, node=None):
    """Return (exact upper/lower rows, exact variable bounds) without solving.

    ``public`` uses observation, radius, baseline and clipped from the frozen
    A4-V public schema. ``arm`` supplies calibration_intervals and, when node
    is given, branches[branch] containing its original trace/retained_ids.
    Global bounds are nine [0,1] intervals plus arm calibration for joint.
    Same-leaf bounds are its original nine darkness intervals plus its own
    three t intervals for joint. No propagation or range tightening is done.
    """
    if not isinstance(public, dict) or not isinstance(arm, dict):
        raise TypeError("public and arm must be frozen record dictionaries")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    if arm.get("appearance_model", "bounded") != "bounded":
        raise ValueError("point diagnosis requires the frozen bounded arm")
    if "drift_bound" in arm and _exact(arm["drift_bound"], "arm drift bound") != DRIFT_BOUND:
        raise ValueError("arm drift bound must remain 1/10")
    if "branches" in arm and arm["branches"][branch].get("mode", method) != method:
        raise ValueError("arm method does not match the requested method")
    clipped = public["clipped"]
    if not isinstance(clipped, bool):
        raise TypeError("public clipped must be bool")
    radius = _exact(public["radius"], "public radius")
    baseline = _exact(public["baseline"], "public baseline")
    if baseline != BASELINE:
        raise ValueError("A6 retains the frozen B80 public baseline")
    depth = _exact(depth, "exact public depth")
    image = variation.appearance_engine._image(public["observation"], clipped)
    calibration = _intervals(arm["calibration_intervals"], 3, "arm calibration")
    if node is None:
        darkness = variation.PER_VIEW_MATERIAL_BOUNDS
        bias = calibration
    else:
        original = _retained_node(arm, branch, method, depth, node)
        darkness = _intervals(original["darkness_bounds"], 9, "node darkness", unit=True)
        bias = calibration
        if method == "joint":
            bias = _intervals(original["t_bounds"], 3, "node t bounds")
            if any(not lo <= sub_lo <= sub_hi <= hi
                   for (lo, hi), (sub_lo, sub_hi) in zip(calibration, bias)):
                raise ValueError("node t bounds must be contained in the arm calibration")
    bounds = darkness + bias if method == "joint" else darkness
    areas = geometry.geometry_bounds(branch, radius, depth, depth, baseline)
    rows = variation.necessary_rows(
        image, areas, method, clipped=clipped, drift_bound=DRIFT_BOUND,
        calibration_intervals=calibration, beta=variation.BETA, epsilon=variation.EPSILON)
    return rows, bounds


def classify_problem(public, arm, method, branch, depth, node=None, deadline=None):
    """Build one point problem and return the unchanged A5 classification schema.

    ``deadline`` is the absolute monotonic deadline accepted by the frozen A5
    classifier. The returned record carries exact proof/witness payloads only
    when that classifier verifies them. There is no scientific early exit.
    """
    rows, bounds = build_problem(public, arm, method, branch, depth, node=node)
    return classify(rows, bounds, deadline=deadline)


def summarize(statuses, empty_status="SOURCE_EXCLUDED"):
    """Existentially aggregate a fully consumed sequence of A5 status strings.

    Any verified witness proves the union relaxation nonempty. Without one,
    all nonempty members must be certified before the union can be certified
    empty. Any UNKNOWN remains UNKNOWN. Empty input is a source exclusion,
    separate from a new A5 infeasibility certificate. Invalid entries are not
    hidden by an earlier witness. Different problems may have different states;
    conflicting proof/witness payloads on one problem need an invariant check.
    """
    if not isinstance(empty_status, str) or not empty_status:
        raise ValueError("empty_status must be a nonempty explicit status string")
    if isinstance(statuses, (str, bytes)):
        raise TypeError("statuses must be an iterable of status strings")
    statuses = tuple(statuses)
    if any(not isinstance(status, str) or status not in CLASSIFICATION_STATUSES
           for status in statuses):
        raise ValueError("all member statuses must be A5 classification status strings")
    if not statuses:
        return empty_status
    if "FEASIBLE_WITNESS" in statuses:
        return "FEASIBLE_WITNESS"
    if all(status == "CERTIFIED_INFEASIBLE" for status in statuses):
        return "CERTIFIED_INFEASIBLE"
    return "UNKNOWN"


def clear_geometry_cache():
    """Expose the original cache clearing operation without changing geometry."""
    variation.clear_geometry_cache()
