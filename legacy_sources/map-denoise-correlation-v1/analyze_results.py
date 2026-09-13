"""Independent saved-output checks and paired summaries, no estimator execution."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def avg(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    result = json.loads((args.run / 'results.json').read_text())
    entries = manifest['entries']
    valid_rows = [row for row in result['records'] if row['execution'] == 'ok']
    inputs = {e['id']: load_npz(args.data / e['input_path']) for e in entries}
    truths = {e['id']: load_npz(args.data / e['eval_path']) for e in entries}
    pairs = defaultdict(list)
    for entry in entries:
        pairs[entry['pair_id']].append(entry)
    checks, noise_rows = [], []
    for pair_id, group in pairs.items():
        ref_entry = next(e for e in group if e['arm'] == 'correlated')
        ref, rt = inputs[ref_entry['id']], truths[ref_entry['id']]
        for entry in group:
            inp, truth = inputs[entry['id']], truths[entry['id']]
            u = truth['perturbation_mm']
            if u.ndim != 1:
                raise ValueError('Protocol requires scalar total normal perturbation')
            checks.extend([
                {'check': 'same_clean', 'id': entry['id'], 'ok': bool(np.array_equal(truth['clean_xyz_mm'], rt['clean_xyz_mm']))},
                {'check': 'same_frame', 'id': entry['id'], 'ok': bool(np.array_equal(inp['frame'], ref['frame']))},
                {'check': 'same_noise_multiset', 'id': entry['id'], 'ok': bool(np.array_equal(np.sort(u), np.sort(rt['perturbation_mm'])))},
                {'check': 'legal_fields', 'id': entry['id'], 'ok': set(inp) == {'xyz_mm', 'frame', 'sigma_mm'}},
                {'check': 'noise_reconstructs_input', 'id': entry['id'], 'ok': bool(np.allclose(inp['xyz_mm'][:, 2], truth['clean_xyz_mm'][:, 2] + u, rtol=0, atol=1e-12))},
            ])
            if entry.get('input_sha256'):
                checks.append({'check': 'input_sha256', 'id': entry['id'], 'ok': sha(args.data / entry['input_path']) == entry['input_sha256']})
            if entry.get('eval_sha256'):
                checks.append({'check': 'eval_sha256', 'id': entry['id'], 'ok': sha(args.data / entry['eval_path']) == entry['eval_sha256']})
            for label in np.unique(truth['labels']):
                mask = truth['labels'] == label
                checks.append({'check': 'same_layer_noise_multiset', 'id': entry['id'], 'label': int(label),
                               'ok': bool(np.array_equal(np.sort(u[mask]), np.sort(rt['perturbation_mm'][mask])))})
            if entry['family'] != 'curved_dual':
                checks.append({'check': 'same_pooled_normal_coordinates', 'id': entry['id'],
                               'ok': bool(np.array_equal(np.sort(inp['xyz_mm'][:, 2]), np.sort(ref['xyz_mm'][:, 2])))})
            frame = inp['frame']
            between = sum(np.sum(frame == f) * (u[frame == f].mean() - u.mean())**2 for f in np.unique(frame))
            total = float(np.sum((u - u.mean())**2))
            noise_rows.append({k: entry[k] for k in ('id', 'pair_id', 'family', 'amplitude_mm', 'arm', 'seed')} |
                              {'rms_mm': float(np.sqrt(np.mean(u**2))),
                               'frame_explained_variance_fraction': float(between / total) if total else 0.})
    worst_recompute = 0.
    for row in valid_rows:
        out_path = args.run / row['output_path']
        output = load_npz(out_path)['xyz_mm']
        truth = truths[row['id']]
        err = output - truth['clean_xyz_mm']
        values = {
            'normal_mae_mm': float(sum(abs(float(z)) for z in err[:, 2]) / len(err)),
            'normal_rmse_mm': float(np.sqrt(np.dot(err[:, 2], err[:, 2]) / len(err))),
            'point_rmse_mm': float(np.sqrt(np.sum(err * err) / len(err))),
        }
        for name, actual in values.items():
            delta = abs(actual - row['metrics'][name])
            worst_recompute = max(worst_recompute, delta)
        checks.append({'check': 'output_hash_and_shape', 'id': row['id'], 'method': row['method'],
                       'ok': sha(out_path) == row['output_sha256'] and output.shape == truth['clean_xyz_mm'].shape and bool(np.isfinite(output).all())})
    # Average permutation repeats within a fixture, not as independent scenes.
    paired_rows = []
    grouped = defaultdict(list)
    for row in valid_rows:
        grouped[(row['pair_id'], row['sigma_protocol'], row['method'])].append(row)
    for (_, mode, method), rows in grouped.items():
        cor = [r for r in rows if r['arm'] == 'correlated']
        shuffled = [r for r in rows if r['arm'] != 'correlated']
        if len(cor) != 1 or not shuffled:
            continue
        cor = cor[0]
        p = {k: cor[k] for k in ('pair_id', 'family', 'group', 'seed', 'amplitude_mm', 'gap_mm')}
        p.update(sigma_protocol=mode, method=method, shuffle_repeats=len(shuffled),
                 correlated_k=cor['selected_k'], shuffled_k=[r['selected_k'] for r in shuffled])
        for metric in ('normal_mae_mm', 'point_rmse_mm', 'layer_gap_error_mm', 'gap_retention', 'half_gap_error_fraction'):
            cv = cor['metrics'][metric]
            sv = avg([r['metrics'][metric] for r in shuffled])
            p[f'correlated_{metric}'] = cv
            p[f'shuffled_{metric}'] = sv
            p[f'delta_shuffle_minus_correlated_{metric}'] = None if cv is None or sv is None else sv - cv
        paired_rows.append(p)
        if method == 'identity':
            checks.append({'check': 'identity_paired_error', 'id': cor['id'],
                           'ok': abs(p['delta_shuffle_minus_correlated_normal_mae_mm']) < 1e-10})
        if method == 'xyz_mixture' and cor['family'] != 'curved_dual':
            checks.append({'check': 'xyz_only_planar_negative_control', 'id': cor['id'],
                           'ok': abs(p['delta_shuffle_minus_correlated_normal_mae_mm']) < 1e-7})
    # Predictive equality under added per-frame translations, not an accuracy theorem.
    invariant = defaultdict(list)
    for row in valid_rows:
        if row['method'] == 'fast' and row['arm'] == 'correlated' and row['sigma_protocol'] == 'base_sigma':
            invariant[(row['family'], row['seed'])].append(row)
    invariance_rows = []
    for (family, seed), rows in invariant.items():
        zeros = [r for r in rows if r['amplitude_mm'] == 0]
        if not zeros:
            continue
        ref = load_npz(args.run / zeros[0]['output_path'])['xyz_mm']
        base_truth = truths[zeros[0]['id']]['base_xyz_mm']
        for row in rows:
            output = load_npz(args.run / row['output_path'])['xyz_mm']
            unchanged_base = np.array_equal(base_truth, truths[row['id']]['base_xyz_mm'])
            delta = float(np.max(np.abs(output - ref)))
            invariance_rows.append({'family': family, 'seed': seed, 'amplitude_mm': row['amplitude_mm'],
                                    'same_base': unchanged_base, 'max_coordinate_change_mm': delta,
                                    'same_selected_k': row['selected_k'] == zeros[0]['selected_k']})
            checks.append({'check': 'cross_amplitude_same_base', 'id': row['id'], 'ok': bool(unchanged_base)})
    audit = {'checks': len(checks), 'failed': [c for c in checks if not c['ok']],
             'max_independent_metric_difference_mm': worst_recompute,
             'successful_outputs': len(valid_rows), 'execution_errors': result['errors'],
             'input_cases': len(entries), 'paired_fixtures': len(pairs),
             'scene_families': len({e['family'] for e in entries}),
             'seeds': sorted({e['seed'] for e in entries}),
             'fast_correlated_invariance': invariance_rows,
             'scope': 'new runs on procedural local families; random seeds are not independent real scenes'}
    dump(args.out / 'audit.json', audit)
    dump(args.out / 'paired.json', {'records': paired_rows})
    dump(args.out / 'noise_diagnostics.json', {'records': noise_rows})
    family_summaries = []
    groups = defaultdict(list)
    for row in paired_rows:
        groups[(row['family'], row['amplitude_mm'], row['sigma_protocol'], row['method'])].append(row)
    for (family, amplitude, mode, method), rows in sorted(groups.items()):
        rec = {'family': family, 'amplitude_mm': amplitude, 'sigma_protocol': mode, 'method': method, 'paired_fixtures': len(rows)}
        for k in rows[0]:
            if k.startswith(('correlated_normal_', 'shuffled_normal_', 'delta_shuffle_minus_correlated_normal_',
                             'correlated_layer_gap_', 'shuffled_layer_gap_')):
                rec[k] = avg([r[k] for r in rows])
        family_summaries.append(rec)
    dump(args.out / 'family_summary.json', {'records': family_summaries})
    text = ['# 配对误差结果明细', '',
            '每格先对同一输入的两次置换取平均，再对三个种子取平均。正差值表示打散组误差更大。', '',
            '| 场景 | 帧偏差RMS/mm | σ协议 | 方法 | 相关 MAE/mm | 打散 MAE/mm | 差值/mm |',
            '|---|---:|---|---|---:|---:|---:|']
    for row in family_summaries:
        text.append(f"| {row['family']} | {row['amplitude_mm']:g} | {row['sigma_protocol']} | {row['method']} | "
                    f"{row['correlated_normal_mae_mm']:.6f} | {row['shuffled_normal_mae_mm']:.6f} | "
                    f"{row['delta_shuffle_minus_correlated_normal_mae_mm']:.6f} |")
    with (args.out / 'TABLES.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(text) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'fast_correlated_invariance'}, ensure_ascii=False), flush=True)
    if audit['failed'] or result['errors'] or worst_recompute > 1e-9:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
