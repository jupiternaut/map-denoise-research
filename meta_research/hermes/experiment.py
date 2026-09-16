#!/usr/bin/env python3
"""Compare research policies on a newly generated discovery episode.

The hidden object is a function on a finite domain. Representation,
optimization objective, and search procedure may all be wrong.
The external purpose is fixed: low error on the whole domain. It is never
rewritten to manufacture success.

This file is the executable comparison required by TASK.tla. Parameters
are tagged ASSUMED (chosen for this episode) or MEASURED (produced by
running this program).
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Episode world  [ASSUMED]
# ---------------------------------------------------------------------------

DOMAIN: Tuple[int, ...] = tuple(range(16))


def target(x: int) -> int:
    """Hidden target. Left piece constant, right piece linear. [ASSUMED]"""
    return 2 if x < 8 else x - 6


TRUE_TABLE = {x: target(x) for x in DOMAIN}

# Biased experimental window used by the proxy objective. [ASSUMED]
PROXY_SUPPORT = tuple(range(0, 8))

# Discrete hole ranges. Small enough for exhaustive search. [ASSUMED]
CONST_RANGE = tuple(range(-2, 7))
SLOPE_RANGE = tuple(range(-2, 4))
INTERCEPT_RANGE = tuple(range(-8, 9))
SPLIT_RANGE = (6, 7, 8, 9)

BUDGET = 8  # action steps, not a claim of equal wall-clock cost. [ASSUMED]
QUERY_COST = 1
REFINE_COST = 1
SYNTH_COST = 1
VALIDATE_SCAN_COST = 1


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def predict_constant(phi: Tuple[int, ...], x: int) -> int:
    (k,) = phi
    return k


def predict_linear(phi: Tuple[int, ...], x: int) -> int:
    a, b = phi
    return a * x + b


def predict_piecewise(phi: Tuple[int, ...], x: int) -> int:
    t, k_left, a_right, b_right = phi
    return k_left if x < t else a_right * x + b_right


FAMILIES = {
    "constant": {
        "holes": [(k,) for k in CONST_RANGE],
        "predict": predict_constant,
        "can_express_target": False,
    },
    "linear": {
        "holes": list(product(SLOPE_RANGE, INTERCEPT_RANGE)),
        "predict": predict_linear,
        "can_express_target": False,
    },
    "piecewise": {
        "holes": list(product(SPLIT_RANGE, CONST_RANGE, SLOPE_RANGE, INTERCEPT_RANGE)),
        "predict": predict_piecewise,
        "can_express_target": True,
    },
}

REFINE_ORDER = ("constant", "linear", "piecewise")


def family_error(family: str, phi: Tuple[int, ...], xs: Iterable[int]) -> int:
    pred = FAMILIES[family]["predict"]
    return sum(abs(pred(phi, x) - TRUE_TABLE[x]) for x in xs)


def external_l1(family: str, phi: Tuple[int, ...]) -> int:
    return family_error(family, phi, DOMAIN)


def proxy_l1(family: str, phi: Tuple[int, ...]) -> int:
    return family_error(family, phi, PROXY_SUPPORT)


def synthesize(family: str, observations: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, ...]]:
    """Exact fit on observations if possible; else min L1 on E, then proxy L1.

    Min-L1 fallback when exact fit is impossible is ASSUMED so incomplete
    sketches still return a candidate rather than crashing.
    """
    holes = FAMILIES[family]["holes"]
    pred = FAMILIES[family]["predict"]
    best = None
    best_key = None
    for phi in holes:
        exact = 0
        l1 = 0
        for x, y in observations:
            e = abs(pred(phi, x) - y)
            l1 += e
            if e:
                exact += 1
        key = (exact, l1, proxy_l1(family, phi), phi)
        if best_key is None or key < best_key:
            best_key = key
            best = phi
    return best


def exact_fit_exists(family: str, observations: Sequence[Tuple[int, int]]) -> bool:
    pred = FAMILIES[family]["predict"]
    for phi in FAMILIES[family]["holes"]:
        if all(pred(phi, x) == y for x, y in observations):
            return True
    return False


def adversarial_counterexample(
    family: str, phi: Tuple[int, ...], unseen: Sequence[int]
) -> Optional[int]:
    """Validator: largest residual among unseen x. [ASSUMED adversarial oracle]"""
    pred = FAMILIES[family]["predict"]
    best_x = None
    best_e = 0
    for x in unseen:
        e = abs(pred(phi, x) - TRUE_TABLE[x])
        if e > best_e:
            best_e = e
            best_x = x
    return best_x if best_e > 0 else None


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def terminal_utility(family: str, phi: Tuple[int, ...], cost: int) -> Dict:
    l1 = external_l1(family, phi)
    return {
        "external_l1": l1,
        "proxy_l1": proxy_l1(family, phi),
        "cost": cost,
        "utility": -float(l1) - 0.25 * float(cost),
        "perfect": l1 == 0,
    }


def policy_fixed_proxy_linear() -> Dict:
    """Spend the budget fitting a linear model to the biased window. No new family."""
    family = "linear"
    obs = []
    cost = 0
    traj = []
    for x in PROXY_SUPPORT:
        if cost + QUERY_COST + SYNTH_COST > BUDGET:
            break
        obs.append((x, TRUE_TABLE[x]))
        cost += QUERY_COST
        traj.append({"action": "Observe", "x": x, "y": TRUE_TABLE[x]})
    if not obs:
        phi = FAMILIES[family]["holes"][0]
    elif cost + SYNTH_COST <= BUDGET:
        cost += SYNTH_COST
        traj.append({"action": "ConstructAlgorithm", "family": family})
        phi = synthesize(family, obs)
    else:
        phi = FAMILIES[family]["holes"][0]
        traj.append({"action": "SelectNextAction", "note": "no_budget_to_synthesize"})
    out = terminal_utility(family, phi, cost)
    out.update({"policy": "fixed_proxy_linear", "family": family, "phi": list(phi), "traj": traj})
    return out


def policy_random_queries() -> Dict:
    """Observe a fixed scattered set, then fit linear. [ASSUMED schedule]."""
    family = "linear"
    schedule = (0, 5, 8, 11, 15, 3, 9, 12)
    obs = []
    cost = 0
    traj = []
    for x in schedule:
        if cost + QUERY_COST + SYNTH_COST > BUDGET:
            break
        obs.append((x, TRUE_TABLE[x]))
        cost += QUERY_COST
        traj.append({"action": "Observe", "x": x, "y": TRUE_TABLE[x]})
    if not obs:
        phi = FAMILIES[family]["holes"][0]
    elif cost + SYNTH_COST <= BUDGET:
        cost += SYNTH_COST
        traj.append({"action": "ConstructAlgorithm", "family": family})
        phi = synthesize(family, obs)
    else:
        phi = FAMILIES[family]["holes"][0]
        traj.append({"action": "SelectNextAction", "note": "no_budget_to_synthesize"})
    out = terminal_utility(family, phi, cost)
    out.update({"policy": "random_then_linear", "family": family, "phi": list(phi), "traj": traj})
    return out


def policy_cegis(start_family: str, allow_refine: bool, name: str) -> Dict:
    """Counterexample-guided inductive synthesis on a sketch family.

    Validator uses the external purpose on unseen points. That is a strong
    oracle; billed per scan. Scope: only episodes that have a refutation
    procedure for the external purpose.
    """
    family = start_family
    cost = 0
    traj = []
    E: List[Tuple[int, int]] = []
    seen = set()
    x0 = 0
    if cost + QUERY_COST <= BUDGET:
        E.append((x0, TRUE_TABLE[x0]))
        seen.add(x0)
        cost += QUERY_COST
        traj.append({"action": "Observe", "x": x0, "y": TRUE_TABLE[x0], "role": "seed"})

    phi = None
    steps = 0
    while cost < BUDGET:
        steps += 1
        if cost + SYNTH_COST > BUDGET:
            break
        cost += SYNTH_COST
        phi = synthesize(family, E)
        traj.append(
            {
                "action": "ConstructAlgorithm",
                "family": family,
                "phi": list(phi),
                "n_obs": len(E),
            }
        )
        unseen = [x for x in DOMAIN if x not in seen] or list(DOMAIN)
        if cost + VALIDATE_SCAN_COST > BUDGET:
            traj.append({"action": "Observe", "note": "budget_exhausted_before_validate"})
            break
        cost += VALIDATE_SCAN_COST
        cex = adversarial_counterexample(family, phi, unseen)
        if cex is None:
            traj.append({"action": "Observe", "result": "validated"})
            break
        if cost + QUERY_COST > BUDGET:
            traj.append({"action": "Observe", "result": "cex_found_but_unrecorded", "x": cex})
            break
        E.append((cex, TRUE_TABLE[cex]))
        seen.add(cex)
        cost += QUERY_COST
        traj.append({"action": "Observe", "x": cex, "y": TRUE_TABLE[cex], "role": "counterexample"})
        if allow_refine and not exact_fit_exists(family, E):
            idx = REFINE_ORDER.index(family)
            if idx + 1 < len(REFINE_ORDER) and cost + REFINE_COST <= BUDGET:
                family = REFINE_ORDER[idx + 1]
                cost += REFINE_COST
                traj.append({"action": "ReviseRepresentation", "family": family})
            else:
                traj.append({"action": "SelectNextAction", "note": "sketch_unsat_no_refine"})
                break
        if steps > 20:
            break
    if phi is None:
        phi = synthesize(family, E) or FAMILIES[family]["holes"][0]
    out = terminal_utility(family, phi, cost)
    out.update(
        {
            "policy": name,
            "family": family,
            "phi": list(phi),
            "traj": traj,
            "n_observations": len(E),
            "allow_refine": allow_refine,
        }
    )
    return out


def policy_revise_objective_after_holdout() -> Dict:
    """Fit proxy, then one holdout; if mismatch, change objective but not family."""
    family = "linear"
    obs = [(x, TRUE_TABLE[x]) for x in PROXY_SUPPORT[:4]]
    cost = QUERY_COST * len(obs)
    traj = [{"action": "Observe", "x": x, "y": y, "role": "proxy"} for x, y in obs]
    cost += SYNTH_COST
    phi = synthesize(family, obs)
    traj.append(
        {
            "action": "ConstructAlgorithm",
            "family": family,
            "objective": "proxy",
            "phi": list(phi),
        }
    )
    holdout = 12
    if cost + QUERY_COST + VALIDATE_SCAN_COST <= BUDGET:
        cost += VALIDATE_SCAN_COST + QUERY_COST
        y_h = TRUE_TABLE[holdout]
        traj.append({"action": "Observe", "x": holdout, "y": y_h, "role": "holdout"})
        if FAMILIES[family]["predict"](phi, holdout) != y_h:
            traj.append(
                {
                    "action": "ReviseOptimizationObjective",
                    "from": "proxy",
                    "to": "all_observed",
                }
            )
            obs.append((holdout, y_h))
            if cost + SYNTH_COST <= BUDGET:
                cost += SYNTH_COST
                phi = synthesize(family, obs)
                traj.append(
                    {
                        "action": "ConstructAlgorithm",
                        "family": family,
                        "objective": "all_observed",
                        "phi": list(phi),
                    }
                )
    out = terminal_utility(family, phi, cost)
    out.update(
        {
            "policy": "proxy_then_holdout_linear",
            "family": family,
            "phi": list(phi),
            "traj": traj,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Independent checks
# ---------------------------------------------------------------------------

def independent_checks() -> Dict:
    checks = []

    def record(name: str, ok: bool, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    record("target_left", all(target(x) == 2 for x in range(8)), [target(x) for x in range(8)])
    record(
        "target_right",
        [target(x) for x in range(8, 16)] == [2, 3, 4, 5, 6, 7, 8, 9],
        [target(x) for x in range(8, 16)],
    )

    pw = FAMILIES["piecewise"]
    exact_pw = [h for h in pw["holes"] if external_l1("piecewise", h) == 0]
    record(
        "piecewise_can_express",
        len(exact_pw) >= 1,
        {"n_exact": len(exact_pw), "example": list(exact_pw[0]) if exact_pw else None},
    )
    exact_lin = [h for h in FAMILIES["linear"]["holes"] if external_l1("linear", h) == 0]
    best_lin = min(FAMILIES["linear"]["holes"], key=lambda h: external_l1("linear", h))
    record(
        "linear_cannot_express",
        len(exact_lin) == 0,
        {"n_exact": len(exact_lin), "best_l1": external_l1("linear", best_lin), "phi": list(best_lin)},
    )

    const2 = (2,)
    record("proxy_perfect_constant2", proxy_l1("constant", const2) == 0, {"external_l1": external_l1("constant", const2)})
    record("proxy_mismatch", external_l1("constant", const2) > 0, external_l1("constant", const2))

    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}


def run() -> Dict:
    t0 = time.perf_counter()
    policies = [
        policy_fixed_proxy_linear(),
        policy_random_queries(),
        policy_revise_objective_after_holdout(),
        policy_cegis("linear", allow_refine=False, name="cegis_linear_fixed"),
        policy_cegis("piecewise", allow_refine=False, name="cegis_piecewise_fixed"),
        policy_cegis("constant", allow_refine=True, name="cegis_refine_families"),
    ]
    ranking = sorted(policies, key=lambda r: (-r["utility"], r["cost"], r["policy"]))
    checks = independent_checks()
    elapsed = time.perf_counter() - t0
    return {
        "episode": {
            "domain": list(DOMAIN),
            "budget_steps": BUDGET,
            "query_cost": QUERY_COST,
            "external_purpose": "minimize L1 on the whole domain; never rewritten",
            "unknowns": ["Representation", "OptimizationObjective", "EffectiveAlgorithm"],
            "proxy": "L1 on x in {0..7}",
            "parameter_provenance": {
                "DOMAIN": "ASSUMED",
                "target": "ASSUMED",
                "PROXY_SUPPORT": "ASSUMED",
                "BUDGET": "ASSUMED",
                "hole_ranges": "ASSUMED",
                "policy_utilities": "MEASURED",
                "independent_checks": "MEASURED",
            },
        },
        "policies": [
            {
                "policy": p["policy"],
                "family": p["family"],
                "phi": p["phi"],
                "external_l1": p["external_l1"],
                "proxy_l1": p["proxy_l1"],
                "cost": p["cost"],
                "utility": p["utility"],
                "perfect": p["perfect"],
                "n_actions": len(p["traj"]),
                "traj": p["traj"],
            }
            for p in policies
        ],
        "ranking_by_utility": [p["policy"] for p in ranking],
        "independent_checks": checks,
        "negative_results": [
            "fixed_proxy_linear can have proxy_l1=0 and large external_l1",
            "cegis_linear_fixed cannot reach external_l1=0 because the sketch is incomplete",
            "revising only the objective, keeping a linear family, cannot express the target",
        ],
        "timing_s": elapsed,
        "executed": ["all six policies", "independent_checks"],
        "not_executed": [
            "original CEGAR TACAS 2000 PDF retrieval (HTTP 404/429 this session)",
            "context ablation over prior-project context (host must supply new sessions)",
            "model checking of TASK.tla",
        ],
    }


def main() -> None:
    result = run()
    out = Path(__file__).resolve().parent / "RESULTS.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)
    print("checks_ok", result["independent_checks"]["all_ok"])
    print("ranking", result["ranking_by_utility"])
    for p in result["policies"]:
        print(
            f"{p['policy']:28s} family={p['family']:10s} L1={p['external_l1']:3d} "
            f"proxy={p['proxy_l1']:3d} cost={p['cost']:2d} U={p['utility']:7.2f} perfect={p['perfect']}"
        )


if __name__ == "__main__":
    main()
