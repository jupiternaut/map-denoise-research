"""Deterministic read-only publication tests; mutations stay in temp copies.

Arithmetic fixtures use only checked-in archives, with deliberately unusable
source provenance. Re-sealing mutations demonstrates checks beyond file hashes.
"""
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_lovelace_a15_20261014 import (
    A14, A15, ARCHIVE_ROOT, MANIFEST_PATH, FORMAL, AuditError, Comparison,
    Snapshot, _history, numerical_audit,
)


def write_manifest(root, entries):
    path = root / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(archive_root=ARCHIVE_ROOT, source_roots=[A15], files=entries)))
    return path


def install(root, relative, raw, encoding='identity'):
    payload = gzip.compress(raw, mtime=0) if encoding == 'gzip' else raw
    target = ARCHIVE_ROOT + '/' + relative.replace('AGENTS.md', 'SOURCE_AGENTS.md')
    if encoding == 'gzip':
        target += '.gz'
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return dict(path=target, source='/nonexistent-provenance-never-open/' + relative,
                source_relative=relative, encoding=encoding, sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload), source_sha256=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw))


def tiny_fixture(root, encoding='identity'):
    entry = install(root, A15 + '/record.json', b'{"answer":42}\n', encoding)
    return root / entry['path'], write_manifest(root, [entry])


def arithmetic_fixture(root):
    """Copy sealed A15 and only the archived A14 arithmetic dependencies."""
    repository = Path(__file__).resolve().parents[1]
    original, historical = Snapshot(repository), _history.Snapshot(repository)
    entries = []
    for entry in original.entries.values():
        entry = copy.deepcopy(entry)
        entry['source'] = '/nonexistent-provenance-never-open/' + entry['source_relative']
        target = root / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original.safe_path(entry), target)
        entries.append(entry)
    write_manifest(root, entries)
    names = {'data/confirm/tasks.json', 'data/confirm/private/truth.json', 'data/confirm/public.json'}
    names.update(FORMAL + name for name in
                 ('PREDICTIONS_READY.json', 'MANIFEST.json', 'METRICS.csv', 'SUMMARY.json', 'COSTS.json', 'ANALYSIS.json'))
    names.update(FORMAL + f'jobs/{i:03}.{kind}.json' for i in range(90) for kind in ('job', 'result', 'receipt'))
    entries = []
    for name in sorted(names):
        entry = copy.deepcopy(historical.sources[A14 + '/' + name])
        entry['source'] = '/nonexistent-provenance-never-open/' + entry['source_relative']
        target = root / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(historical.safe_path(entry), target)
        entries.append(entry)
    path = root / _history.MANIFEST_PATH
    path.write_text(json.dumps(dict(archive_root=_history.ARCHIVE_ROOT,
                                   source_roots=list(_history.ROOTS), files=entries)))
    return Snapshot(root)


def replace_and_reseal(root, relative, raw):
    manifest_path = root / MANIFEST_PATH
    manifest = json.loads(manifest_path.read_bytes())
    for i, entry in enumerate(manifest['files']):
        if entry['source_relative'] == relative:
            manifest['files'][i] = install(root, relative, raw, entry['encoding'])
            break
    else:
        raise AssertionError('fixture record not found: ' + relative)
    manifest_path.write_text(json.dumps(manifest))
    return Snapshot(root)


def replace_json(root, snapshot, name, value):
    return replace_and_reseal(root, A15 + '/' + name, json.dumps(value, allow_nan=False).encode())


def replace_result(root, snapshot, stage, index, output):
    """Update both hash-bearing receipts so the state audit is the rejection."""
    stem = f'outputs/{stage}/jobs/{index:03}'
    raw = json.dumps(output, allow_nan=False).encode()
    snapshot = replace_and_reseal(root, A15 + '/' + stem + '.result.json', raw)
    ready_name = f'outputs/{stage}/READY.json'
    ready = snapshot.source_json(A15 + '/' + ready_name)
    receipt = ready['receipts'][index]
    receipt['sha256'] = hashlib.sha256(raw).hexdigest()
    snapshot = replace_json(root, snapshot, stem + '.receipt.json', receipt)
    return replace_json(root, snapshot, ready_name, ready)


class SealTests(unittest.TestCase):
    def test_identity_and_gzip_never_need_original_source(self):
        for encoding in ('identity', 'gzip'):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                tiny_fixture(root, encoding)
                result = Snapshot(root).seal()
                self.assertEqual(result['files'], 1)
                self.assertEqual(result['decoded_source_bytes'], len(b'{"answer":42}\n'))

    def test_missing_extra_and_corrupt_paths_rejected(self):
        for change, message in (('missing', 'file set differs'), ('extra', 'file set differs'),
                                ('corrupt', 'published-byte seal mismatch')):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                path, _ = tiny_fixture(root)
                if change == 'missing':
                    path.unlink()
                elif change == 'extra':
                    path.with_name('extra.json').write_text('{}')
                else:
                    path.write_bytes(b'{"answer":43}\n')
                with self.assertRaisesRegex(AuditError, message):
                    Snapshot(root).seal()

    def test_wrong_stored_or_decoded_seal_rejected(self):
        for field, value, message in (('bytes', 1, 'published-byte seal mismatch'),
                                       ('source_bytes', 900, 'decoded-byte seal mismatch'),
                                       ('source_sha256', '0'*64, 'decoded-byte seal mismatch')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                _, path = tiny_fixture(root, 'gzip')
                manifest = json.loads(path.read_bytes())
                manifest['files'][0][field] = value
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(AuditError, message):
                    Snapshot(root).seal()

    def test_escaping_and_duplicate_paths_rejected(self):
        for change in ('escape', 'duplicate'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                _, path = tiny_fixture(root)
                manifest = json.loads(path.read_bytes())
                if change == 'escape':
                    manifest['files'][0]['path'] = ARCHIVE_ROOT + '/../escape.json'
                else:
                    manifest['files'].append(copy.deepcopy(manifest['files'][0]))
                path.write_text(json.dumps(manifest))
                with self.assertRaises(AuditError):
                    Snapshot(root)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = tiny_fixture(root)
            outside = root / 'outside.json'
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(AuditError, 'symlink'):
                Snapshot(root).seal()

    def test_older_verifier_globals_unchanged(self):
        import verify_lovelace_20261013 as older
        import verify_lovelace_methods_20261013 as methods
        self.assertEqual(older.ARCHIVE_ROOT, 'research_snapshots/2026-10-13')
        self.assertEqual(older.MANIFEST_PATH, 'publication/LOVELACE_20261013_MANIFEST.json')
        self.assertEqual(methods.ARCHIVE_ROOT, 'research_snapshots/2026-10-13-method-search')
        self.assertEqual(methods.MANIFEST_PATH, 'publication/LOVELACE_METHODS_20261013_MANIFEST.json')


class ArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_dir = tempfile.TemporaryDirectory()
        cls.baseline = Path(cls.baseline_dir.name) / 'baseline'
        cls.baseline.mkdir()
        arithmetic_fixture(cls.baseline)

    @classmethod
    def tearDownClass(cls):
        cls.baseline_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'copy'
        shutil.copytree(self.baseline, self.root)
        self.snapshot = Snapshot(self.root)

    def assert_resealed_failure(self, snapshot, message):
        snapshot.seal()
        with self.assertRaisesRegex(AuditError, message):
            numerical_audit(snapshot)

    def test_archive_arithmetic_passes_without_source_paths(self):
        self.snapshot.seal()
        _history.Snapshot(self.root).seal()
        result = numerical_audit(self.snapshot)
        self.assertEqual((result['cold_calls'], result['task_outputs_recomputed']), (306, 2142))
        self.assertEqual(result['full_state_equivalences'], 216)
        self.assertEqual(result['per_stage']['tolerance']['calls'], 36)
        self.assertEqual(result['per_stage']['switches']['calls'], 270)
        self.assertEqual(result['boundary']['tile_counts'],
                         {'exact_infeasible': 125, 'exact_necessary_lp_feasible': 163})
        pure = result['per_stage']['tolerance']['arms']['pure']
        self.assertEqual((pure['improved'], pure['same'], pure['harmed']), (75, 33, 0))
        self.assertAlmostEqual(pure['shifted_mse'], 6.763968234545869, places=12)
        self.assertEqual(result['per_stage']['tolerance']['exclusions'], 784)
        self.assertEqual(result['per_stage']['switches']['exclusions'], 6870)
        counts = result['profile']['arms']['hot']['completed_lp_and_reuse_counts']
        self.assertEqual(counts['tile_float_lp'], 4947)
        self.assertEqual(counts['actual_first_nonzero_tile'], 1128)

    def test_resealed_false_metric_rejected(self):
        relative = A15 + '/outputs/tolerance/METRICS.csv'
        reader = csv.DictReader(io.StringIO(self.snapshot.source_bytes(relative).decode()))
        fields, rows = reader.fieldnames, list(reader)
        rows[0]['final_se'] = str(float(rows[0]['final_se']) + 1.)
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        snapshot = replace_and_reseal(self.root, relative, output.getvalue().encode())
        self.assert_resealed_failure(snapshot, r'tolerance.METRICS.csv\[0\].final_se')

    def test_resealed_false_summary_rejected(self):
        name = 'outputs/tolerance/SUMMARY.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['arms']['pure']['shifted_mse'] += .01
        snapshot = replace_json(self.root, self.snapshot, name, value)
        self.assert_resealed_failure(snapshot, r'SUMMARY.json.arms.pure.shifted_mse')

    def test_resealed_raw_state_change_rejected_after_receipt_hash_updates(self):
        name = 'outputs/switches/jobs/001.result.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        self.assertEqual(value['arm'], 'off')
        value['state']['cells'][0]['checked'] = not value['state']['cells'][0]['checked']
        snapshot = replace_result(self.root, self.snapshot, 'switches', 1, value)
        self.assert_resealed_failure(snapshot, 'full state equivalence differs')

    def test_resealed_raw_counter_change_rejected(self):
        name = 'outputs/switches/jobs/001.result.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['state']['inner_counts']['tile_float_lp'] += 1
        snapshot = replace_result(self.root, self.snapshot, 'switches', 1, value)
        self.assert_resealed_failure(snapshot, 'completed floating LP counter mismatch')

    def test_resealed_false_profile_counter_rejected(self):
        name = 'outputs/switches/PROFILE.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['arms']['off']['completed_lp_and_reuse_counts']['tile_float_lp'] += 1
        snapshot = replace_json(self.root, self.snapshot, name, value)
        self.assert_resealed_failure(snapshot, 'PROFILE.json.arms.off.completed_lp_and_reuse_counts.tile_float_lp')

    def test_resealed_false_boundary_counts_rejected(self):
        name = 'outputs/boundary-tolerance.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['tile_counts']['exact_infeasible'] += 1
        snapshot = replace_json(self.root, self.snapshot, name, value)
        self.assert_resealed_failure(snapshot, 'boundary tile_counts.exact_infeasible')

    def test_resealed_wrong_boundary_source_rejected(self):
        name = 'outputs/boundary-tolerance.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['records'][0]['sources'][0]['arm'] = 'pure'
        snapshot = replace_json(self.root, self.snapshot, name, value)
        self.assert_resealed_failure(snapshot, r'boundary source record.sources\[0\].arm')

    def test_resealed_ready_count_rejected(self):
        name = 'outputs/tolerance/READY.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['calls'] += 1
        snapshot = replace_json(self.root, self.snapshot, name, value)
        self.assert_resealed_failure(snapshot, 'READY call count')

    def test_resealed_false_historical_group_rejected(self):
        name = 'outputs/tolerance/SUMMARY.json'
        value = self.snapshot.source_json(A15 + '/' + name)
        value['historical_groups']['both']['remaining_legacy_losses'] += 1
        snapshot = replace_json(self.root, self.snapshot, name, value)
        self.assert_resealed_failure(snapshot, 'SUMMARY.json.historical_groups.both.remaining_legacy_losses')

    def test_nonfinite_numeric_comparison_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(AuditError):
                Comparison().equal(1.0, value, 'number')


if __name__ == '__main__':
    unittest.main()
