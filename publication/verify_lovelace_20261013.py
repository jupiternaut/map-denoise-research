#!/usr/bin/env python3
"""Portable, read-only seal and numerical audit of the 2026-10-13 publication.

Uses Python's standard library only. It reads the repository snapshot and
manifest, never the absolute source paths recorded as provenance, and prints
JSON to stdout. This is not a repeat of the original geometric proof audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
ARCHIVE_ROOT = 'research_snapshots/2026-10-13'
MANIFEST_PATH = 'publication/LOVELACE_20261013_MANIFEST.json'


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, dict):
        require(set(value) == {'numerator', 'denominator'} and
                type(value['numerator']) is int and type(value['denominator']) is int and
                value['denominator'] > 0, 'invalid exact rational object')
        answer = Fraction(value['numerator'], value['denominator'])
        require(answer.numerator == value['numerator'] and answer.denominator == value['denominator'],
                'noncanonical rational object')
        return answer
    require(type(value) in (int, str), 'exact arithmetic forbids floating-point input')
    return Fraction(value)


def canonical_relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'invalid repository relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '.' not in path.parts and str(path) == value,
            'noncanonical or escaping repository path: ' + value)
    return path


def digest_stream(stream):
    digest, size = hashlib.sha256(), 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


class Snapshot:
    def __init__(self, repository):
        self.repository = repository.resolve()
        manifest_path = self.repository / MANIFEST_PATH
        self.manifest_bytes = manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        require(self.manifest.get('archive_root') == ARCHIVE_ROOT, 'unexpected archive root')
        require(isinstance(self.manifest.get('files'), list) and self.manifest['files'], 'nonempty files manifest required')
        self.entries, self.sources = {}, {}
        for entry in self.manifest['files']:
            require(isinstance(entry, dict), 'invalid manifest entry')
            require({'path', 'source', 'sha256', 'bytes', 'source_sha256', 'source_bytes', 'encoding', 'source_relative'} <= set(entry),
                    'missing manifest seal fields')
            path = canonical_relative(entry['path'])
            source = canonical_relative(entry['source_relative'])
            require(str(path).startswith(ARCHIVE_ROOT + '/'), 'entry outside publication archive')
            require(entry['path'] not in self.entries and entry['source_relative'] not in self.sources,
                    'duplicate manifest path or source-relative identity')
            require(entry['encoding'] in ('identity', 'gzip'), 'unsupported publication encoding')
            require(all(type(entry[key]) is int and entry[key] >= 0 for key in ('bytes', 'source_bytes')),
                    'invalid byte count')
            for key in ('sha256', 'source_sha256'):
                value = entry[key]
                require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                        'invalid SHA-256 encoding')
            require('.aris' not in source.parts and '__pycache__' not in source.parts and '.cache' not in source.parts,
                    'excluded runtime/cache path in manifest')
            target_parts = ['SOURCE_AGENTS.md' if part == 'AGENTS.md' else part for part in source.parts]
            expected_path = ARCHIVE_ROOT + '/' + '/'.join(target_parts)
            if entry['encoding'] == 'gzip':
                expected_path += '.gz'
            require(str(path) == expected_path, 'publication path does not match reversible source-relative mapping')
            # entry['source'] is provenance text only. Never dereference it.
            require(isinstance(entry['source'], str), 'source provenance must be text')
            self.entries[entry['path']] = entry
            self.sources[entry['source_relative']] = entry

    def safe_path(self, entry):
        path = self.repository / entry['path']
        current = self.repository
        for part in canonical_relative(entry['path']).parts:
            current = current / part
            require(not current.is_symlink(), 'symlink not allowed in sealed archive: ' + entry['path'])
        require(path.is_file() and path.resolve().is_relative_to(self.repository), 'missing/escaping published file: ' + entry['path'])
        return path

    def seal(self):
        archive = self.repository / ARCHIVE_ROOT
        require(archive.is_dir() and not archive.is_symlink(), 'missing archive directory')
        found = set()
        for path in archive.rglob('*'):
            require(not path.is_symlink(), 'symlink in archive: ' + str(path.relative_to(self.repository)))
            if path.is_file():
                found.add(path.relative_to(self.repository).as_posix())
        require(found == set(self.entries), 'archive file set differs from manifest: ' + json.dumps({
            'missing': sorted(set(self.entries) - found)[:20], 'unlisted': sorted(found - set(self.entries))[:20]}))
        encoded_bytes = original_bytes = 0
        encodings = Counter()
        for entry in self.entries.values():
            path = self.safe_path(entry)
            with path.open('rb') as stream:
                digest, size = digest_stream(stream)
            require((digest, size) == (entry['sha256'], entry['bytes']), 'published-byte seal mismatch: ' + entry['path'])
            if entry['encoding'] == 'gzip':
                with gzip.open(path, 'rb') as stream:
                    source_digest, source_size = digest_stream(stream)
            else:
                source_digest, source_size = digest, size
            require((source_digest, source_size) == (entry['source_sha256'], entry['source_bytes']),
                    'decoded-byte seal mismatch: ' + entry['path'])
            encoded_bytes += size
            original_bytes += source_size
            encodings[entry['encoding']] += 1
        return {'files': len(self.entries), 'published_bytes': encoded_bytes, 'decoded_source_bytes': original_bytes,
                'encodings': dict(sorted(encodings.items())), 'exact_archive_file_set': True,
                'both_sha256_and_byte_counts_match': True,
                'manifest_sha256': hashlib.sha256(self.manifest_bytes).hexdigest()}

    def locate(self, stage, suffix):
        matches = [entry for source, entry in self.sources.items()
                   if ('/' + stage + '/') in ('/' + source) and source.endswith(suffix)]
        require(len(matches) == 1, f'expected one archived {stage}/{suffix}; found {len(matches)}')
        return matches[0]

    def read_bytes(self, entry):
        path = self.safe_path(entry)
        return gzip.decompress(path.read_bytes()) if entry['encoding'] == 'gzip' else path.read_bytes()

    def json(self, stage, suffix):
        return json.loads(self.read_bytes(self.locate(stage, suffix)))


def merge_intervals(intervals):
    pairs = sorted((rational(lo), rational(hi)) for lo, hi in intervals)
    result = []
    for lo, hi in pairs:
        require(lo <= hi, 'reversed closed interval')
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return tuple(result)


def metric(rows):
    require(rows, 'empty metric population')
    initial = [abs(rational(row['incumbent_mm']) - rational(row['truth_mm'])) for row in rows]
    final = [abs(rational(row['selected_mm']) - rational(row['truth_mm'])) for row in rows]
    before_mse = sum((value * value for value in initial), Fraction()) / len(rows)
    after_mse = sum((value * value for value in final), Fraction()) / len(rows)
    return {'count': len(rows), 'mae_mm': str(sum(final, Fraction()) / len(rows)),
            'mse_mm2': str(after_mse), 'initial_mae_mm': str(sum(initial, Fraction()) / len(rows)),
            'initial_mse_mm2': str(before_mse),
            'relative_mse_reduction': str((before_mse - after_mse) / before_mse) if before_mse else None,
            'improved': sum(b < a for a, b in zip(initial, final)),
            'unchanged': sum(b == a for a, b in zip(initial, final)),
            'worsened': sum(b > a for a, b in zip(initial, final))}


def index_unique(records, keys, name):
    require(isinstance(records, list), name + ': expected an array')
    indexed = {}
    for record in records:
        key = tuple(record[field] for field in keys)
        require(key not in indexed, name + ': duplicate identity ' + repr(key))
        indexed[key] = record
    return indexed


def support(record):
    outer = merge_intervals(record['outer_intervals'])
    require(tuple((rational(lo), rational(hi)) for lo, hi in record['outer_intervals']) == outer,
            'saved support must already be a canonical closed union')
    hull = None if not outer else (outer[0][0], outer[-1][1])
    saved_hull = None if record['hull'] is None else tuple(map(rational, record['hull']))
    require(saved_hull == hull, 'hull differs from exact union endpoints')
    return outer


def contains(intervals, value):
    return any(lo <= value <= hi for lo, hi in intervals)


def selected(incumbent, intervals):
    return incumbent if not intervals else min(intervals[-1][1], max(intervals[0][0], incumbent))


def numerical_audit(snapshot):
    a9, a10 = 'continuous-anchors-1', 'continuous-tightening-1'
    old_prefix, new_prefix = '/outputs/continuous-v1/', '/outputs/tightening-v1/'
    observations = snapshot.json(a9, old_prefix + 'observations.json')
    tasks = snapshot.json(a9, old_prefix + 'tasks.json')
    truth_records = snapshot.json('appearance-variation-1', '/data/private/truth.json')
    old_sets = snapshot.json(a9, old_prefix + 'sets.json')
    new_sets = snapshot.json(a10, new_prefix + 'sets.json')
    predictions = snapshot.json(a10, new_prefix + 'predictions.json')
    old_predictions = snapshot.json(a9, old_prefix + 'predictions.json')
    rows = snapshot.json(a10, new_prefix + 'evaluation/ROWS.json')
    old_rows = snapshot.json(a9, old_prefix + 'evaluation/ROWS.json')
    summaries = snapshot.json(a10, new_prefix + 'evaluation/SUMMARY.json')
    comparisons = snapshot.json(a10, new_prefix + 'evaluation/COMPARISONS.json')
    saved_coverage = snapshot.json(a10, new_prefix + 'evaluation/COVERAGE.json')
    obs_index = index_unique(observations, ('observation_id',), 'observations')
    task_index = index_unique(tasks, ('task_id',), 'tasks')
    truth_index = index_unique(truth_records, ('observation_id',), 'synthetic truth')
    require(len(obs_index) == len(truth_index) == 36 and set(obs_index) == set(truth_index), '36 observations/truth identities required')
    require(len(task_index) == 324 and Counter(t['observation_id'] for t in tasks) ==
            Counter({key[0]: 9 for key in obs_index}), '324 tasks, exactly nine per observation required')
    require(len(old_sets) == 144 and len(new_sets) == 72 and len(predictions) == 648 and len(rows) == 972,
            'publication population counts differ from the registered experiment')
    baseline_sets = [dict(record, arm='a9_both') for record in old_sets if record['arm'] == 'both']
    all_sets = baseline_sets + new_sets
    set_index = index_unique(all_sets, ('observation_id', 'arm'), 'sets')
    arms = ('a9_both', 'depth', 'camera_tiles')
    require(set(set_index) == {(oid, arm) for oid, in obs_index for arm in arms}, 'incomplete 36-by-three support matrix')
    outer = {}
    coverage = Counter()
    for (oid, arm), record in set_index.items():
        branches = index_unique(record['branches'], ('branch',), 'branches')
        require(set(branches) == {('none',), ('partial',), ('full',)}, 'missing occlusion branch')
        branch_support = {}
        for (branch,), branch_record in branches.items():
            intervals = support(branch_record)
            retained = [n for n in branch_record['trace'] if n['status'].startswith('retained_')]
            require(merge_intervals((n['lo'], n['hi']) for n in retained) == intervals,
                    'saved branch union differs from its retained leaves')
            require(sorted(branch_record['retained_ids']) == sorted(n['id'] for n in retained),
                    'saved retained IDs differ from retained leaves')
            if arm != 'a9_both':
                baseline_branch = next(b for b in set_index[(oid, 'a9_both')]['branches'] if b['branch'] == branch)
                inherited = support(baseline_branch)
                require(merge_intervals(branch_record['source_outer_intervals']) == inherited,
                        'new branch does not inherit saved A9 support')
                require(all(any(sl <= lo <= hi <= sh for sl, sh in inherited) for lo, hi in intervals),
                        'new branch is not a subset of its A9 input')
            branch_support[branch] = intervals
        intervals = support(record)
        require(intervals == merge_intervals(pair for value in branch_support.values() for pair in value),
                'saved arm union differs from the three branch union')
        truth = truth_index[(oid,)]
        z = rational(truth['truth'])
        require(contains(branch_support[truth['rear_branch']], z), 'truth absent from generating branch')
        require(contains(intervals, z) and intervals[0][0] <= z <= intervals[-1][1], 'truth absent from union/hull')
        outer[(oid, arm)] = intervals
        coverage[arm] += 1
    coverage_index = index_unique(saved_coverage, ('observation_id', 'arm'), 'saved coverage')
    require(set(coverage_index) == {(oid, arm) for oid, in obs_index for arm in arms[1:]}, 'incomplete 72 saved coverage rows')
    for (oid, arm), row in coverage_index.items():
        truth = truth_index[(oid,)]
        require(row['covered'] is True and rational(row['truth_mm']) == rational(truth['truth']) and
                row['generating_branch'] == truth['rear_branch'], 'saved coverage metadata mismatch')
    baseline_predictions = [dict(p, arm='a9_both') for p in old_predictions if p['arm'] == 'both' and p['policy'] == 'HULL']
    pred_index = index_unique(baseline_predictions + predictions, ('task_id', 'arm'), 'predictions')
    require(set(pred_index) == {(tid, arm) for tid, in task_index for arm in arms}, 'incomplete 324-by-three HULL matrix')
    for (tid, arm), prediction in pred_index.items():
        task = task_index[(tid,)]
        oid = task['observation_id']
        intervals = outer[(oid, arm)]
        incumbent = rational(task['incumbent'])
        result = selected(incumbent, intervals)
        require(prediction['observation_id'] == oid and prediction['policy'] == 'HULL', 'action identity/policy mismatch')
        require(support(prediction) == intervals and rational(prediction['incumbent']) == incumbent and
                rational(prediction['selected']) == result, 'independent HULL coordinate mismatch')
        require(type(prediction['moved']) is bool and prediction['moved'] == (result != incumbent), 'action movement flag mismatch')
        require(prediction['model_conditional_nonregression'] is True and
                prediction['status'] == ('PROJECTED' if intervals else 'SUPPORT_FAILED'), 'action guarantee/status mismatch')
    row_index = index_unique(rows, ('task_id', 'arm'), 'evaluation rows')
    old_row_index = index_unique([dict(r, arm='a9_both') for r in old_rows if r['arm'] == 'both' and r['policy'] == 'HULL'],
                                 ('task_id', 'arm'), 'old evaluation rows')
    require(set(row_index) == set(pred_index), 'evaluation row population differs from saved actions')
    expected_counts = {'all': 324, 'shifted': 288, 'informative': 240, 'negative': 48, 'correct': 36, 'primary': 40}
    for (tid, arm), row in row_index.items():
        task = task_index[(tid,)]
        truth = truth_index[(task['observation_id'],)]
        pred = pred_index[(tid, arm)]
        z, incumbent, result = rational(truth['truth']), rational(task['incumbent']), rational(pred['selected'])
        initial, final = abs(incumbent-z), abs(result-z)
        expected_values = {'truth_mm': z, 'incumbent_mm': incumbent, 'selected_mm': result,
                           'initial_ae_mm': initial, 'final_ae_mm': final,
                           'ae_gain_mm': initial-final, 'displacement_mm': abs(result-incumbent)}
        for key, value in expected_values.items():
            require(rational(row[key]) == value, 'evaluation arithmetic mismatch: ' + key)
        require(row['observation_id'] == task['observation_id'] and row['policy'] == 'HULL' and
                type(row['moved']) is bool and row['moved'] == (result != incumbent), 'evaluation action metadata mismatch')
        for key in ('case', 'unit_id', 'variation_index'):
            require(row[key] == truth[key], 'evaluation synthetic metadata mismatch: ' + key)
        require(isinstance(row['groups'], list) and len(set(row['groups'])) == len(row['groups']) and
                set(row['groups']) <= set(expected_counts), 'invalid evaluation group labels')
        require(('correct' in row['groups']) == (initial == 0) and ('shifted' in row['groups']) == (initial != 0),
                'correct/shifted group differs from exact initial error')
        require(('negative' in row['groups']) == (initial != 0 and truth['case'] == 'equal_full_negative') and
                ('informative' in row['groups']) == (initial != 0 and truth['case'] != 'equal_full_negative'),
                'informative/negative group differs from frozen synthetic case')
        old_row = old_row_index[(tid, 'a9_both')]
        require(row['groups'] == old_row['groups'], 'new evaluation changes inherited grouping')
        if arm == 'a9_both':
            require(row == old_row, 'A10 copied A9 evaluation row differs from sealed A9 row')
    for arm in arms:
        require(Counter(group for row in rows if row['arm'] == arm for group in row['groups']) == Counter(expected_counts),
                'evaluation group counts differ from fixed population')
    summary_index = index_unique(summaries, ('arm', 'policy', 'unit', 'group'), 'summaries')
    units = ['ALL'] + sorted({row['unit_id'] for row in rows})
    expected_summary = {(arm, 'HULL', unit, group) for arm in arms for unit in units for group in expected_counts
                        if any(r['arm'] == arm and group in r['groups'] and (unit == 'ALL' or r['unit_id'] == unit) for r in rows)}
    require(set(summary_index) == expected_summary and len(summaries) == 51, 'incomplete 51 summary cells')
    metrics = {}
    for (arm, policy, unit, group), saved in summary_index.items():
        chosen = [r for r in rows if r['arm'] == arm and group in r['groups'] and (unit == 'ALL' or r['unit_id'] == unit)]
        value = metric(chosen)
        numeric = {'initial_MAE_mm': value['initial_mae_mm'], 'final_MAE_mm': value['mae_mm'],
                   'initial_MSE_mm2': value['initial_mse_mm2'], 'final_MSE_mm2': value['mse_mm2'],
                   'MAE_gain_mm': rational(value['initial_mae_mm']) - rational(value['mae_mm']),
                   'MSE_gain_mm2': rational(value['initial_mse_mm2']) - rational(value['mse_mm2']),
                   'mean_displacement_mm': sum((rational(r['displacement_mm']) for r in chosen), Fraction()) / len(chosen),
                   'max_AE_increase_mm': max([Fraction()] + [rational(r['final_ae_mm']) - rational(r['initial_ae_mm']) for r in chosen])}
        for key, number in numeric.items():
            expected_number = number if isinstance(number, Fraction) else rational(number)
            require(rational(saved[key]) == expected_number, 'summary mismatch: ' + repr((arm, unit, group, key)))
        for key, number in {'n': value['count'], 'improved': value['improved'], 'equal': value['unchanged'],
                            'worsened': value['worsened'], 'moved': sum(r['moved'] for r in chosen)}.items():
            require(type(saved[key]) is int and saved[key] == number, 'summary count mismatch: ' + key)
        percent = None if value['relative_mse_reduction'] is None else 100 * float(rational(value['relative_mse_reduction']))
        require(saved['MSE_reduction_percent'] is None if percent is None else
                type(saved['MSE_reduction_percent']) in (int, float) and abs(saved['MSE_reduction_percent']-percent) <= 1e-10,
                'summary displayed percentage mismatch')
        if unit == 'ALL':
            metrics.setdefault(arm, {})[group] = value
    old_unchanged = {tid for (tid, arm), row in row_index.items() if arm == 'a9_both' and
                     'informative' in row['groups'] and rational(row['ae_gain_mm']) == 0}
    require(len(old_unchanged) == 118, 'A9 informative unchanged denominator mismatch')
    comparison_index = index_unique(comparisons, ('arm', 'group'), 'comparisons')
    groups = tuple(expected_counts) + ('old_informative_unchanged',)
    require(set(comparison_index) == {(arm, group) for arm in arms[1:] for group in groups}, 'incomplete 14 comparison cells')
    pair_comparisons = {}
    for (arm, group), saved in comparison_index.items():
        chosen = [r for r in rows if r['arm'] == arm and
                  (r['task_id'] in old_unchanged if group == 'old_informative_unchanged' else group in r['groups'])]
        gains = [rational(row_index[(r['task_id'], 'a9_both')]['final_ae_mm']) - rational(r['final_ae_mm']) for r in chosen]
        expected = {'n': len(chosen), 'improved_over_a9': sum(g > 0 for g in gains),
                    'equal_to_a9': sum(g == 0 for g in gains), 'worsened_over_a9': sum(g < 0 for g in gains)}
        require(all(type(saved[k]) is int and saved[k] == value for k, value in expected.items()) and
                rational(saved['MAE_gain_over_a9_mm']) == sum(gains, Fraction()) / len(gains), 'A9 comparison mismatch')
        if group in ('shifted', 'old_informative_unchanged'):
            pair_comparisons[arm + '/' + group] = expected
    dc_minus_d = [rational(row_index[(tid, 'depth')]['final_ae_mm']) - rational(row_index[(tid, 'camera_tiles')]['final_ae_mm'])
                  for tid, in task_index if 'shifted' in row_index[(tid, 'depth')]['groups']]
    contrast = {'better': sum(g > 0 for g in dc_minus_d), 'equal': sum(g == 0 for g in dc_minus_d),
                'worse': sum(g < 0 for g in dc_minus_d)}
    require(contrast == {'better': 36, 'equal': 240, 'worse': 12}, 'DC-vs-D report comparison changed')
    report = snapshot.read_bytes(snapshot.locate(a10, '/continuous-tightening-1/REPORT.md')).decode('utf-8')
    for label, arm in (('A9 both', 'a9_both'), ('D：深度细化', 'depth'), ('DC：另加相机分盒', 'camera_tiles')):
        table_rows = [[cell.strip().replace('**', '') for cell in line.strip().strip('|').split('|')]
                      for line in report.splitlines() if line.startswith('| ' + label + ' |')]
        require(len(table_rows) == 1, 'missing unique report metric row: ' + label)
        cells, value = table_rows[0], metrics[arm]['shifted']
        require(len(cells) == 5 and abs(rational(cells[1])-rational(value['mae_mm'])) <= Fraction(1, 2000000) and
                abs(rational(cells[2])-rational(value['mse_mm2'])) <= Fraction(1, 2000000) and
                abs(rational(cells[3].removesuffix('%'))-100*rational(value['relative_mse_reduction'])) <= Fraction(1, 200000),
                'report rounded MAE/MSE/percentage mismatch: ' + label)
        require(cells[4] == '/'.join(str(value[key]) for key in ('improved', 'unchanged', 'worsened')).replace('/', '／'),
                'report improvement counts mismatch: ' + label)
    require(pair_comparisons['depth/old_informative_unchanged']['improved_over_a9'] == 31 and '新增修复31个' in report,
            'report newly repaired count mismatch')
    require(all(metrics[arm]['correct']['unchanged'] == 36 and metrics[arm]['negative']['unchanged'] == 48 for arm in arms),
            'report correct-input/negative-control counts mismatch')
    probes = snapshot.json(a10, new_prefix + 'probes.json')
    probe_rows = snapshot.json(a10, new_prefix + 'evaluation/PROBES.json')
    probe_index = index_unique(probes, ('observation_id', 'side'), 'probes')
    probe_row_index = index_unique(probe_rows, ('observation_id', 'side'), 'probe evaluations')
    require(set(probe_index) == set(probe_row_index) == {(oid, side) for oid, in obs_index for side in ('left', 'right')},
            'incomplete 72 probe matrix')
    distances, statuses = [], Counter()
    for (oid, side), probe in probe_index.items():
        row = probe_row_index[(oid, side)]
        z, truth = rational(probe['depth']), rational(truth_index[(oid,)]['truth'])
        status = probe['result']['status']
        require(status in ('FORWARD_WITNESS', 'UNKNOWN') and row['status'] == status and
                rational(row['depth_mm']) == z and rational(row['truth_mm']) == truth,
                'probe evaluation identity/depth mismatch')
        statuses[status] += 1
        if status == 'FORWARD_WITNESS':
            require(rational(row['distance_from_true_mm']) == abs(z-truth) and z != truth, 'witness-distance arithmetic mismatch')
            distances.append(abs(z-truth))
    require(statuses == Counter({'FORWARD_WITNESS': 9, 'UNKNOWN': 63}) and max(distances) == Fraction(14009, 5632),
            'reported probe-status counts/maximum distance differ from saved probe records')
    return {'observations': 36, 'tasks': 324, 'new_sets': 72, 'new_predictions': 648,
            'all_evaluation_rows_checked': 972, 'all_summary_cells_checked': 51, 'all_comparison_cells_checked': 14,
            'independent_hull_actions_checked': 972, 'truth_covered_generating_branch_union_and_hull': dict(coverage),
            'summary_metrics': metrics, 'improvement_over_a9': pair_comparisons, 'dc_versus_d_shifted': contrast,
            'core_report_table_matches_recomputed_values': True,
            'recorded_probe_status_counts': dict(statuses), 'recorded_probe_max_truth_distance_mm': str(max(distances)),
            'probe_forward_equations_rechecked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        snapshot = Snapshot(args.repo_root)
        result = {'status': 'PASS', 'seal': snapshot.seal()}
        result['numerical_audit'] = numerical_audit(snapshot)
        result['limitations'] = ['No solver or geometry/certificate verification is rerun.',
                                'The seal checks repository bytes against the manifest; it is not a digital signature.',
                                'Absolute source provenance paths are never accessed.',
                                'Synthetic generating-world and probe forward equations are not rechecked here.',
                                'The truth-coverage check is exact interval membership for the archived synthetic truth.',
                                'No claim of real-scene generalization or 324 independent scenes is made.']
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'FAIL', 'error': str(error)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())
