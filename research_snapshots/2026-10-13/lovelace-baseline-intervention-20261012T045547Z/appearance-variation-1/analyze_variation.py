"""Registered paired summaries, only after independent verification PASS.

Reads evaluated endpoints, not private generator facts; does not change inference.
"""
from fractions import Fraction as Q
from pathlib import Path
import argparse
import json
import socket
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1')
ARMS = ('strict_shared', 'independent', 'bounded')
METHODS = ('calibrated', 'joint')
DEPTH = '9007/14'
NEGATIVE = 'equal_full_negative'


def load(path):
    return json.loads(Path(path).read_text())


def metrics(rows):
    if not rows:
        raise ValueError('empty registered analysis cell')
    ae = [Q(r['ae_mm']) for r in rows]
    initial = [Q(r['initial_ae_mm']) for r in rows]
    n = len(rows)
    return {'n': n, 'mae_mm': str(sum(ae, Q(0))/n),
            'mse_mm2': str(sum((x*x for x in ae), Q(0))/n),
            'keep_mse_mm2': str(sum((x*x for x in initial), Q(0))/n),
            'improved': sum(b<a for a,b in zip(initial, ae)),
            'same': sum(b==a for a,b in zip(initial, ae)),
            'harmed': sum(b>a for a,b in zip(initial, ae))}


def subset(rows, group):
    if group == 'correct':
        return [r for r in rows if Q(r['initial_ae_mm']) == 0]
    shifted = [r for r in rows if Q(r['initial_ae_mm']) != 0]
    if group == 'all_shifted':
        return shifted
    if group == 'negative_shifted':
        return [r for r in shifted if r['case'] == NEGATIVE]
    if group == 'primary_depth':
        return [r for r in shifted if r['case'] != NEGATIVE and r['depth_mm'] == DEPTH]
    raise ValueError('unregistered group')


def paired(reference, proposed):
    first = {r['base_task_id']: r for r in reference}
    second = {r['base_task_id']: r for r in proposed}
    if len(first) != len(reference) or len(second) != len(proposed) or set(first) != set(second):
        raise ValueError('paired identities differ')
    pairs = [(first[key], second[key]) for key in sorted(first)]
    return {'n': len(pairs),
            'mse_decrease_mm2': str(sum((Q(a['se_mm2'])-Q(b['se_mm2']) for a,b in pairs), Q(0))/len(pairs)),
            'mae_decrease_mm': str(sum((Q(a['ae_mm'])-Q(b['ae_mm']) for a,b in pairs), Q(0))/len(pairs)),
            'better': sum(Q(b['se_mm2']) < Q(a['se_mm2']) for a,b in pairs),
            'equal': sum(Q(b['se_mm2']) == Q(a['se_mm2']) for a,b in pairs),
            'worse': sum(Q(b['se_mm2']) > Q(a['se_mm2']) for a,b in pairs)}


def build(rows, coverage, diagnostic, timings, receipt):
    if len(rows) != 1944 or len(coverage) != 216 or len(diagnostic) != 36 or len(timings) != 144:
        raise ValueError('incomplete verified population')
    cells, contrasts, coverage_cells, sensitivity = [], [], [], []
    for eta_index in range(2):
        unit = ('appearance_eta_0', 'appearance_eta_1_10')[eta_index]
        for method in METHODS:
            selected = {arm: [r for r in rows if r['unit_id'] == unit and r['method'] == method and r['condition'] == arm] for arm in ARMS}
            for group, expected in (('all_shifted',144), ('correct',18), ('negative_shifted',24), ('primary_depth',40)):
                groups = {arm: subset(selected[arm], group) for arm in ARMS}
                if any(len(value) != expected for value in groups.values()):
                    raise ValueError('changed registered denominator')
                for arm in ARMS:
                    cells.append({'eta_index': eta_index, 'method': method, 'appearance_model': arm, 'group': group, **metrics(groups[arm])})
                for reference, proposed in (('independent','bounded'), ('strict_shared','bounded'), ('independent','strict_shared')):
                    contrasts.append({'eta_index': eta_index, 'method': method, 'group': group,
                                      'reference': reference, 'proposed': proposed, **paired(groups[reference], groups[proposed])})
            for arm in ARMS:
                c = [r for r in coverage if r['variation_index'] == eta_index and r['method'] == method and r['appearance_model'] == arm]
                if len(c) != 18:
                    raise ValueError('changed coverage denominator')
                counts = {key: {'true': sum(r[key] is True for r in c), 'false': sum(r[key] is False for r in c),
                                'not_applicable': sum(r[key] is None for r in c)}
                          for key in ('covered','generating_branch_covered','latent_material_covered','latent_bias_covered','joint_witness_covered')}
                coverage_cells.append({'eta_index': eta_index, 'method': method, 'appearance_model': arm,
                                       'n': 18, 'incompatible': sum(r['incompatible'] for r in c), **counts})
    for method in METHODS:
        for arm in ARMS:
            for group in ('all_shifted','primary_depth','correct'):
                zero = subset([r for r in rows if r['unit_id'] == 'appearance_eta_0' and r['method'] == method and r['condition'] == arm], group)
                changed = subset([r for r in rows if r['unit_id'] == 'appearance_eta_1_10' and r['method'] == method and r['condition'] == arm], group)
                sensitivity.append({'method': method, 'appearance_model': arm, 'group': group,
                                    'reference_eta': '0', 'proposed_eta': '1/10', **paired(zero, changed)})
    nesting = receipt['finite_outer_subset_checks']
    joint_nesting = receipt['finite_method_subset_checks']
    return {'status': 'POST_VERIFICATION_ANALYSIS', 'primary': {'eta_index': 1, 'depth_mm': DEPTH, 'n_per_method_model': 40},
            'independent_sources': 18, 'same_family_development_replay': True,
            'cells': cells, 'paired_contrasts': contrasts, 'appearance_sensitivity': sensitivity,
            'coverage_cells': coverage_cells,
            'outer_nesting_descriptive_not_acceptance': {key: {'passed': sum(r[key] for r in nesting), 'n': len(nesting)}
                 for key in ('bounded_subset_independent','strict_subset_bounded')},
            'joint_subset_calibrated_descriptive_not_acceptance': {'passed': sum(r['joint_subset_calibrated'] for r in joint_nesting), 'n': len(joint_nesting)},
            'image_changed_per_eta': [{'eta_index': i, 'n': 18,
                'changed': sum(not r['image_unchanged'] for r in diagnostic if r['observation_id'].endswith('e'+str(i)))} for i in range(2)],
            'cold_timing': {'n': 144, 'sum_seconds': sum(r['cold_seconds'] for r in timings),
                            'median_seconds': statistics.median(r['cold_seconds'] for r in timings),
                            'max_seconds': max(r['cold_seconds'] for r in timings)},
            'claim_boundary': 'Known public design bound; two intervention levels; no real lighting-bound estimation or new-scene generalization claim.'}


def analyze(output, save=False):
    output = Path(output).resolve()
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED or output.parent != ROOT/'outputs':
        raise RuntimeError('wrong exact target')
    stage, run, receipt = load(output/'STAGE_STATUS.json'), load(output/'RUN_STATUS.json'), load(output/'verification/VERIFY.json')
    if stage.get('status') != 'COMPLETE' or run.get('status') != 'COMPLETE' or receipt.get('status') != 'PASS' or receipt.get('output') != str(output):
        raise RuntimeError('analysis requires exact-target independently verified completion')
    result = build(load(output/'results.json'), load(output/'coverage.json'), load(output/'intervention_diagnostic.json'),
                   load(output/'TIMING.json'), receipt)
    if save:
        with (output/'ANALYSIS.json').open('x',encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/variation-v1')
    parser.add_argument('--save', action='store_true')
    print(json.dumps(analyze(**vars(parser.parse_args())), ensure_ascii=False, allow_nan=False))
