"""Exact finite boolean program synthesis; Python standard library only.

Inputs are integers 0..15; xj is (input >> j) & 1. Expressions are trees,
so repeated subexpressions count repeatedly toward the three binary-gate cap.
The learner's API has no target/evaluator argument or hidden-label access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

N_INPUTS = 16
FULL_MASK = (1 << N_INPUTS) - 1
TARGET_SEED = 20260916
SAMPLE_SEEDS = tuple(range(9100, 9108))
TRAIN_SIZES = (4, 8, 16, 32)


@dataclass(frozen=True)
class Expr:
    op: str
    value: int = 0
    left: Expr | None = None
    right: Expr | None = None

    def evaluate(self, x: int) -> int:
        if self.op == "const":
            return self.value
        if self.op == "literal":
            return (x >> self.value) & 1
        if self.op == "neg_literal":
            return 1 - ((x >> self.value) & 1)
        assert self.left is not None and self.right is not None
        a, b = self.left.evaluate(x), self.right.evaluate(x)
        if self.op == "&":
            return a & b
        if self.op == "|":
            return a | b
        if self.op == "^":
            return a ^ b
        raise ValueError(self.op)

    def gate_count(self) -> int:
        if self.op in ("const", "literal", "neg_literal"):
            return 0
        assert self.left is not None and self.right is not None
        return 1 + self.left.gate_count() + self.right.gate_count()


@dataclass(frozen=True)
class Program:
    mask: int
    gates: int
    expression: str
    tree: Expr

    def evaluate(self, x: int) -> int:
        return self.tree.evaluate(x)


@dataclass(frozen=True)
class FitResult:
    program: Program
    train_errors: int
    sample_count: int
    consistent: bool
    candidate_checks: int

    @property
    def train_error(self) -> float:
        return self.train_errors / self.sample_count if self.sample_count else 0.0


def canonical_key(program: Program) -> tuple[int, str]:
    return program.gates, program.expression


def truth_mask(tree: Expr) -> int:
    """Independent recursive evaluation, without consulting cached program masks."""
    return sum(tree.evaluate(x) << x for x in range(N_INPUTS))


@lru_cache(maxsize=None)
def _library_and_stats(max_gates: int) -> tuple[tuple[Program, ...], tuple[tuple[int, int, int], ...]]:
    if max_gates < 0:
        raise ValueError("max_gates must be nonnegative")
    best: dict[int, Program] = {}
    for value in (0, 1):
        tree = Expr("const", value)
        best[truth_mask(tree)] = Program(truth_mask(tree), 0, str(value), tree)
    for j in range(4):
        for op, spelling in (("literal", f"x{j}"), ("neg_literal", f"!x{j}")):
            tree = Expr(op, j)
            best[truth_mask(tree)] = Program(truth_mask(tree), 0, spelling, tree)
    levels: list[tuple[Program, ...]] = [tuple(sorted(best.values(), key=canonical_key))]
    stats = [(0, 10, 10)]
    for gates in range(1, max_gates + 1):
        proposals: dict[int, Program] = {}
        attempts = 0
        for left_gates in range(gates):
            right_gates = gates - 1 - left_gates
            if left_gates > right_gates:
                continue
            left_level, right_level = levels[left_gates], levels[right_gates]
            for i, left in enumerate(left_level):
                for j, right in enumerate(right_level):
                    if left_gates == right_gates and j < i:
                        continue
                    a, b = sorted((left, right), key=lambda program: program.expression)
                    for op in ("&", "|", "^"):
                        attempts += 1
                        mask = {"&": a.mask & b.mask, "|": a.mask | b.mask, "^": a.mask ^ b.mask}[op]
                        if mask in best:
                            continue
                        spelling = f"({a.expression}{op}{b.expression})"
                        previous = proposals.get(mask)
                        if previous is None or spelling < previous.expression:
                            proposals[mask] = Program(mask, gates, spelling, Expr(op, left=a.tree, right=b.tree))
        best.update(proposals)
        levels.append(tuple(sorted(proposals.values(), key=canonical_key)))
        stats.append((gates, attempts, len(proposals)))
    return tuple(sorted(best.values(), key=canonical_key)), tuple(stats)


def build_library(max_gates: int = 3) -> tuple[Program, ...]:
    """All distinct semantics of the signed-literal grammar up to max_gates.

    Keep minimal gate count, then lexicographically first expression with sorted
    children for commutative operators. Retaining minimal children is complete:
    replacing a nonminimal child cannot increase its parent's gate count.
    """
    return _library_and_stats(max_gates)[0]


def affine_masks() -> frozenset[int]:
    return frozenset(
        sum((((x & subset).bit_count() & 1) ^ intercept) << x for x in range(N_INPUTS))
        for subset in range(16)
        for intercept in (0, 1)
    )


def _loss_tables(samples: tuple[tuple[int, int], ...]) -> tuple[list[int], list[int]]:
    counts = [[0, 0] for _ in range(N_INPUTS)]
    for x, y in samples:
        if not isinstance(x, int) or not 0 <= x < N_INPUTS or y not in (0, 1):
            raise ValueError("samples must contain inputs 0..15 and labels 0 or 1")
        counts[x][y] += 1
    tables: list[list[int]] = []
    for offset in (0, 8):
        table = [0] * 256
        table[0] = sum(counts[x][1] for x in range(offset, offset + 8))
        for mask in range(1, 256):
            bit = mask & -mask
            x = offset + bit.bit_length() - 1
            table[mask] = table[mask ^ bit] + counts[x][0] - counts[x][1]
        tables.append(table)
    return tables[0], tables[1]


def fit(samples: Iterable[tuple[int, int]], candidates: Iterable[Program]) -> FitResult:
    """Exact ERM over supplied candidates; counts repeated samples with multiplicity.

    Ties use (binary gate count, canonical expression), independently of order.
    candidates may be the full library, the affine subclass, or a bounded prefix.
    A nonzero optimum means no consistent member of this supplied candidate set;
    only evaluation against the full grammar can distinguish prefix exhaustion.
    """
    observations = tuple(samples)
    low, high = _loss_tables(observations)
    winner: Program | None = None
    minimum = len(observations) + 1
    checked = 0
    for program in candidates:
        checked += 1
        loss = low[program.mask & 255] + high[program.mask >> 8]
        if loss < minimum or (loss == minimum and winner is not None and canonical_key(program) < canonical_key(winner)):
            minimum, winner = loss, program
    if winner is None:
        raise ValueError("candidate set is empty")
    return FitResult(winner, minimum, len(observations), minimum == 0, checked)


def _derived_seed(target_id: str, regime: str, rep: int) -> int:
    message = f"learning-v1|{SAMPLE_SEEDS[rep]}|{target_id}|{regime}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def iid_inputs(seed: int, m: int, support: tuple[int, ...]) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.choice(support) for _ in range(m))


def _error(predicted: int, target: int, support: tuple[int, ...]) -> float | None:
    if not support:
        return None
    return sum(((predicted ^ target) >> x) & 1 for x in support) / len(support)


def _target_draws(library: tuple[Program, ...]) -> list[dict]:
    rng = random.Random(TARGET_SEED)
    in_grammar = {p.mask for p in library}
    affine = affine_masks()
    populations = {
        "affine": sorted(affine),
        "nonlinear_in_grammar": sorted(in_grammar - affine),
        "out_of_grammar": [mask for mask in range(FULL_MASK + 1) if mask not in in_grammar],
    }
    targets = []
    for family, population in populations.items():
        for draw in range(48):
            mask = rng.choice(population)
            targets.append({"family": family, "target_id": f"{family}_{draw:02d}", "target_mask": mask})
    return targets


def _summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in ("family", "regime", "m", "method"))
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for key, group in sorted(grouped.items()):
        aggregate = dict(zip(("family", "regime", "m", "method"), key))
        aggregate["n"] = len(group)
        for field in ("train_error", "test_error", "full_domain_error", "unseen_error", "consistent", "representation_support_failure", "search_budget_failure", "candidate_checks", "seen_unique", "fit_seconds"):
            values = [row[field] for row in group if row[field] is not None]
            aggregate[f"mean_{field}"] = sum(values) / len(values) if values else None
        aggregate["unseen_error_n"] = sum(row["unseen_error"] is not None for row in group)
        aggregates.append(aggregate)
    return {"aggregates": aggregates}


def run(output_dir: Path) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    library_started = time.perf_counter()
    library = build_library()
    library_seconds = time.perf_counter() - library_started
    masks = affine_masks()
    candidates = {
        "exact": library,
        "affine": tuple(p for p in library if p.mask in masks),
        "prefix32": library[:32],
    }
    targets = _target_draws(library)
    stats = _library_and_stats(3)[1]
    catalog = {
        "schema": "learning-v1",
        "input_convention": "xj=(input>>j)&1; mask bit input is its output",
        "grammar": "constants 0/1, signed literals x0..x3, binary AND/OR/XOR, at most 3 binary gates in a tree",
        "canonical_tie_rule": "minimum gate count, then lexicographic expression with lexicographically sorted commutative children",
        "target_seed": TARGET_SEED,
        "sample_seeds": list(SAMPLE_SEEDS),
        "sample_derivation": "big-endian first 8 bytes of SHA256('learning-v1|{sample_seed}|{target_id}|{regime}'); Python Random.choice iid with replacement; m sizes are nested prefixes",
        "target_draw_order": ["affine", "nonlinear_in_grammar", "out_of_grammar"],
        "population_sizes": {"affine": len(masks), "nonlinear_in_grammar": len(library) - len(masks), "out_of_grammar": FULL_MASK + 1 - len(library)},
        "method_candidate_counts": {name: len(items) for name, items in candidates.items()},
        "semantic_counts_by_minimum_gates": {str(g): count for g, attempts, count in stats},
        "construction_attempts_by_gates": {str(g): attempts for g, attempts, count in stats},
        "duplicate_or_previously_represented_construction_attempts": {str(g): attempts - count for g, attempts, count in stats},
        "library_seconds": library_seconds,
        "programs": [{"mask": p.mask, "gates": p.gates, "expression": p.expression} for p in library],
        "targets": targets,
        "target_unique_counts": {family: len({t["target_mask"] for t in targets if t["family"] == family}) for family in ("affine", "nonlinear_in_grammar", "out_of_grammar")},
    }
    (output_dir / "learning_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(json.dumps({"library_count": len(library), "by_gates": catalog["semantic_counts_by_minimum_gates"], "unique_targets": catalog["target_unique_counts"], "library_seconds": library_seconds}), flush=True)
    rows: list[dict] = []
    sample_records: list[dict] = []
    full_domain = tuple(range(16))
    for target_number, target in enumerate(targets):
        target_mask = target["target_mask"]
        for regime in ("uniform", "shift_x3"):
            training_support = full_domain if regime == "uniform" else tuple(range(8))
            deployment_support = full_domain if regime == "uniform" else tuple(range(8, 16))
            for rep in range(len(SAMPLE_SEEDS)):
                seed = _derived_seed(target["target_id"], regime, rep)
                inputs = iid_inputs(seed, max(TRAIN_SIZES), training_support)
                all_samples = tuple((x, (target_mask >> x) & 1) for x in inputs)
                sample_records.append({"target_id": target["target_id"], "regime": regime, "rep": rep, "sample_seed": SAMPLE_SEEDS[rep], "derived_seed": seed, "samples": all_samples})
                for m in TRAIN_SIZES:
                    samples = all_samples[:m]
                    seen = {x for x, y in samples}
                    unseen = tuple(x for x in full_domain if x not in seen)
                    results = {}
                    timings = {}
                    for method, method_candidates in candidates.items():
                        fit_started = time.perf_counter()
                        results[method] = fit(samples, method_candidates)
                        timings[method] = time.perf_counter() - fit_started
                    for method, result in results.items():
                        representation_failure = not result.consistent if method != "prefix32" else not results["exact"].consistent
                        budget_failure = method == "prefix32" and not result.consistent and results["exact"].consistent
                        support_status = "consistent" if result.consistent else ("search_budget_exhausted" if budget_failure else "representation_support_failure")
                        row = {
                            **target, "regime": regime, "m": m, "rep": rep, "method": method,
                            "pred_mask": result.program.mask,
                            "train_error": result.train_error,
                            "test_error": _error(result.program.mask, target_mask, deployment_support),
                            "full_domain_error": _error(result.program.mask, target_mask, full_domain),
                            "unseen_error": _error(result.program.mask, target_mask, unseen),
                            "seen_error": _error(result.program.mask, target_mask, tuple(sorted(seen))),
                            "consistent": int(result.consistent),
                            "candidate_set_support_failure": int(not result.consistent),
                            "representation_support_failure": int(representation_failure),
                            "search_budget_failure": int(budget_failure),
                            "support_status": support_status,
                            "candidate_checks": result.candidate_checks,
                            "seen_unique": len(seen), "unseen_count": len(unseen),
                            "duplicate_training_draws": m - len(seen),
                            "expression": result.program.expression,
                            "gates": result.program.gates,
                            "fit_seconds": timings[method],
                        }
                        rows.append(row)
        if (target_number + 1) % 24 == 0:
            print(f"Completed {target_number + 1}/{len(targets)} targets; rows={len(rows)}; elapsed={time.perf_counter() - started:.2f}s", flush=True)
    csv_path = output_dir / "learning_rows.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "learning_samples.json").write_text(json.dumps({"train_sizes": TRAIN_SIZES, "samples": sample_records}, separators=(",", ":")) + "\n")
    summary = _summarize(rows)
    summary.update({
        "row_count": len(rows), "target_draw_count": len(targets),
        "training_sets": len(rows) // len(candidates),
        "library_seconds": library_seconds, "run_seconds": time.perf_counter() - started,
        "target_unique_counts": catalog["target_unique_counts"],
        "candidate_counts": catalog["method_candidate_counts"],
        "unseen_error_definition": "uniform error over full-domain inputs absent from this training sample; null when none remain",
        "test_error_definition": "uniform over 0..15 for uniform regime; uniform over x3=1 (8..15) for shift_x3 regime",
        "support_definition": "prefix32 failure is a search budget failure when the full grammar has a consistent member; otherwise representation support failure. Affine failure refers to affine representation.",
        "total_candidate_checks": sum(row["candidate_checks"] for row in rows),
    })
    (output_dir / "learning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "aggregates"}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    arguments = parser.parse_args()
    run(arguments.output_dir)
