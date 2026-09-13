"""Read-only numerical audit; not an AgentRx implementation or a new experiment.

No project algorithms are imported. Existing metrics are reaggregated, and four
saved parameter sets are evaluated under the same penalized likelihood.
Only the optional, exclusively created output file is written.
"""
import argparse
import csv
import hashlib
import json
import socket
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
HERE = ROOT / 'research_audits/agentrx_20260913'
RUN = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/parallel-v19-9ob96cyr')
SOURCES = set()


def read(path):
    path = Path(path)
    SOURCES.add(path)
    return json.loads(path.read_text())


def table(path):
    path = Path(path)
    SOURCES.add(path)
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def average(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def objective(model, x, y, sigma):
    assert model['family'] == 'L'
    z = (x[:, 1:] - np.asarray(model['feature_center'])) / np.asarray(model['feature_scale'])
    features = np.column_stack([np.ones(len(y)), z])
    gate = np.asarray(model['gate'])
    logits = features @ gate
    levels = np.asarray(model['means'])[None, :] + (x[:, 1:] @ np.asarray(model['slope']))[:, None]
    log_prior = np.column_stack([-np.logaddexp(0, logits), -np.logaddexp(0, -logits)])
    log_joint = -0.5 * ((y[:, None] - levels) / sigma) ** 2 + log_prior
    nll = float(-np.logaddexp(log_joint[:, 0], log_joint[:, 1]).sum())
    return nll + 0.05 * float(np.sum(gate[1:] ** 2)), nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='New file only; omit for stdout.')
    args = parser.parse_args()
    assert socket.gethostname() == 'liekkas', 'Wrong host; no substitute is allowed.'
    assert ROOT.resolve() == ROOT and ROOT.is_dir()
    pdf = HERE / 'source/AgentRx_2602.02475v2.pdf'
    SOURCES.add(pdf)
    pdf_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert pdf_sha == '19749e87ac7c3ad50f746f457e887cdd93b1963a429a0fbf20e5aa6f80c36388'

    four = []
    for seed in (9191101, 9191111, 9191121, 9191131):
        case = f's{seed}_g8_n1_p0.5_l1'
        cache_path = RUN / 'cache' / f'{case}.json'
        output_path = RUN / 'outputs' / f'{case}__a_insample_gate_map.json'
        cache, output = read(cache_path), read(output_path)
        baseline = cache['baseline']
        training = output['metadata']['gate_diagnostics']['training']
        assert len(training) == 1
        trained = training[0]
        x, y = np.asarray(cache['x']), np.asarray(cache['y'])
        assert trained['train_indices'] == list(range(len(y)))
        assert baseline['family'] == trained['family'] == 'L'
        for key in ('feature_center', 'feature_scale'):
            np.testing.assert_array_equal(baseline[key], trained[key])
        assert -30 <= baseline['gate'][0] <= 30
        old_obj, _ = objective(baseline, x, y, cache['sigma'])
        new_obj, new_nll = objective(trained, x, y, cache['sigma'])
        assert abs(2 * old_obj + 7 * np.log(len(y)) - baseline['bic']) < 1e-8
        assert abs(new_nll - trained['nll']) < 1e-8
        assert new_obj > old_obj
        four.append(dict(case=case, baseline_objective=old_obj,
                         insample_gate_objective=new_obj, excess_objective=new_obj-old_obj,
                         starts=trained['starts'], cache=str(cache_path), output=str(output_path)))

    jobs = read(RUN / 'development_JOBS.json') + read(RUN / 'confirmation_JOBS.json')
    assert len({j['case'] for j in jobs}) == len(jobs) == 306
    schemas = Counter()
    for job in jobs:
        cache = read(job['cache'])
        schemas[tuple(sorted(cache))] += 1
        assert not any(k in cache for k in ('pool', 'retained', 'old', 'solver_diagnostics'))
        assert not any(k in cache['baseline'] for k in ('pool', 'solver_diagnostics', 'candidate_scores'))
    history_count = sum('cached_diagnostic' in j for j in jobs)
    assert history_count == 48

    rows = table(RUN / 'confirmation_ALL_RESULTS.csv')
    dev = table(RUN / 'development_ALL_RESULTS.csv')
    assert len(rows) == 1856 and len(dev) == 592
    assert all(r['status'] == 'OK' for r in rows + dev)
    aggregate = []
    for kind in sorted({r['kind'] for r in rows}):
        for method in ('b_map', 'a_insample_gate_map', 'a_crossfit_gate_map', 'b_quota_map'):
            subset = [r for r in rows if r['kind'] == kind and r['method'] == method and float(r['gap']) > 0]
            assert subset
            aggregate.append(dict(kind=kind, method=method, count=len(subset),
                                  surface_mae_mm=average(subset, 'surface_mae_mm'),
                                  source_mae_mm=average(subset, 'balanced_source_mae_mm'),
                                  vertical_w1_mm=average(subset, 'vertical_w1_mm'),
                                  dual_missed=sum(int(float(r['dual_missed'])) for r in subset)))

    v15 = table(ROOT / 'evidence/runs/challenge-v15-tchfksi_/confirmation_RESULTS.csv')
    v15_groups = []
    for method in sorted({r['method'] for r in v15}):
        subset = [r for r in v15 if r['method'] == method]
        expected_status = 'APPLY' if method == 'old_original' else 'OK'
        assert len(subset) == 144 and all(r['status'] == expected_status for r in subset)
        assert all(r['geometry_scored'] == 'True' for r in subset)
        dual = [r for r in subset if float(r['gap']) > 0]
        v15_groups.append(dict(method=method, count=len(subset),
                              surface_mae_mm=average(subset, 'surface_accuracy_mean_mm'),
                              balanced_source_mae_mm=average(subset, 'layer_balanced_source_surface_mae_mm'),
                              source_gap_error_mm=average(dual, 'source_group_gap_error_mm')))

    result = dict(kind='AgentRx-inspired evidence audit, not an official AgentRx run',
                  host=socket.gethostname(), project=str(ROOT), pdf_sha256=pdf_sha,
                  check_status='all numerical assertions passed; not a scientific success verdict',
                  gate_same_objective=four,
                  cache_inventory=dict(jobs=len(jobs), historical_diagnostic_links=history_count,
                                       new_cases_without_full_pool_archive=len(jobs)-history_count,
                                       schemas=[dict(keys=list(k), count=v) for k, v in schemas.items()]),
                  v19_record_counts=dict(development=len(dev), confirmation=len(rows)),
                  v19_dual_aggregates=aggregate, v15_aggregates=v15_groups,
                  sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(SOURCES)])
    result['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(payload)
        print(json.dumps(dict(output=str(args.output), gate_same_objective=four,
                              v19_dual_aggregates=aggregate, v15_aggregates=v15_groups), indent=2))
    else:
        print(payload)


if __name__ == '__main__':
    main()
