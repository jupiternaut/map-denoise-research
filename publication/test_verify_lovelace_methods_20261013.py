"""Read-only publication tests; every mutation is confined to a temp directory.

Seal tests use tiny synthetic files. Arithmetic tests copy only the necessary
checked-in A14 records, never the original runs or their provenance paths.
"""
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from verify_lovelace_methods_20261013 import (
    A14, ARCHIVE_ROOT, FORMAL, MANIFEST_PATH, ROOTS,
    AuditError, Comparison, Snapshot, aggregate, numerical_audit,
)


def write_manifest(root, entries):
    path = root / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'archive_root': ARCHIVE_ROOT, 'source_roots': list(ROOTS), 'files': entries}))
    return path


def install(root, relative, raw, encoding='identity'):
    payload = gzip.compress(raw, mtime=0) if encoding == 'gzip' else raw
    target = ARCHIVE_ROOT + '/' + relative.replace('AGENTS.md', 'SOURCE_AGENTS.md')
    if encoding == 'gzip':
        target += '.gz'
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return dict(path=target, source='/nonexistent-provenance-must-never-be-opened/' + relative,
                source_relative=relative, encoding=encoding, sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload), source_sha256=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw))


def tiny_fixture(root, encoding='identity'):
    entry = install(root, ROOTS[0] + '/record.json', b'{"answer":42}\n', encoding)
    return root / entry['path'], write_manifest(root, [entry])


def arithmetic_fixture(root):
    repository = Path(__file__).resolve().parents[1]
    original = Snapshot(repository)
    names = {'data/confirm/tasks.json', 'data/confirm/private/truth.json', 'data/confirm/public.json'}
    names.update(FORMAL + name for name in (
        'PREDICTIONS_READY.json', 'MANIFEST.json', 'METRICS.csv', 'SUMMARY.json', 'COSTS.json', 'ANALYSIS.json'))
    names.update(FORMAL + f'jobs/{i:03}.{kind}.json'
                 for i in range(90) for kind in ('job', 'result', 'receipt'))
    entries = []
    for name in sorted(names):
        relative = A14 + '/' + name
        entry = copy.deepcopy(original.sources[relative])
        entry['source'] = '/nonexistent-provenance-must-never-be-opened/' + relative
        target = root / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.safe_path(entry).read_bytes())
        entries.append(entry)
    write_manifest(root, entries)
    return Snapshot(root)


def replace_and_reseal(root, relative, raw):
    """Deliberately make the byte seal valid after semantic corruption."""
    manifest_path = root / MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text())
    for i, entry in enumerate(manifest['files']):
        if entry['source_relative'] == relative:
            manifest['files'][i] = install(root, relative, raw, entry['encoding'])
            break
    else:
        raise AssertionError('fixture record not found')
    manifest_path.write_text(json.dumps(manifest))
    snapshot = Snapshot(root)
    snapshot.seal()
    return snapshot


class SealTests(unittest.TestCase):
    def test_identity_and_gzip_without_source_access(self):
        for encoding in ('identity', 'gzip'):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                tiny_fixture(root, encoding)
                result = Snapshot(root).seal()
                self.assertEqual(result['files'], 1)
                self.assertEqual(result['decoded_source_bytes'], len(b'{"answer":42}\n'))

    def test_corrupted_archive_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = tiny_fixture(root)
            path.write_bytes(b'{"answer":43}\n')
            with self.assertRaisesRegex(AuditError, 'published-byte seal mismatch'):
                Snapshot(root).seal()

    def test_missing_archive_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = tiny_fixture(root)
            path.unlink()
            with self.assertRaisesRegex(AuditError, 'file set differs'):
                Snapshot(root).seal()

    def test_unlisted_archive_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = tiny_fixture(root)
            path.with_name('extra.json').write_text('{}')
            with self.assertRaisesRegex(AuditError, 'file set differs'):
                Snapshot(root).seal()

    def test_wrong_decoded_sha_or_size_rejected(self):
        for field, value in (('source_sha256', '0' * 64), ('source_bytes', 900)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                _, path = tiny_fixture(root, 'gzip')
                manifest = json.loads(path.read_text())
                manifest['files'][0][field] = value
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(AuditError, 'decoded-byte seal mismatch'):
                    Snapshot(root).seal()

    def test_wrong_stored_size_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, path = tiny_fixture(root)
            manifest = json.loads(path.read_text())
            manifest['files'][0]['bytes'] += 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AuditError, 'published-byte seal mismatch'):
                Snapshot(root).seal()

    def test_escaping_path_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, path = tiny_fixture(root)
            manifest = json.loads(path.read_text())
            manifest['files'][0]['path'] = ARCHIVE_ROOT + '/../escape.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AuditError, 'escaping repository path'):
                Snapshot(root)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = tiny_fixture(root)
            other = root / 'outside-archive.json'
            other.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(other)
            with self.assertRaisesRegex(AuditError, 'symlink'):
                Snapshot(root).seal()

    def test_older_verifier_constants_are_unchanged(self):
        import verify_lovelace_20261013 as older
        self.assertEqual(older.ARCHIVE_ROOT, 'research_snapshots/2026-10-13')
        self.assertEqual(older.MANIFEST_PATH, 'publication/LOVELACE_20261013_MANIFEST.json')


class ArithmeticTests(unittest.TestCase):
    def test_archived_arithmetic_fixture_passes(self):
        with tempfile.TemporaryDirectory() as name:
            snapshot = arithmetic_fixture(Path(name))
            snapshot.seal()
            result = numerical_audit(snapshot)
            self.assertEqual(result['task_outputs_recomputed'], 630)
            self.assertEqual(result['cold_calls'], 90)
            joint = result['per_configuration']['joint@6']['shifted']
            self.assertEqual((joint['improved'], joint['same'], joint['harmed']), (75, 33, 0))
            self.assertAlmostEqual(joint['mse'], 6.794983958823187, places=12)
            self.assertAlmostEqual(result['report_key_values']['joint_gap_recovery_percent'], 86.35840764113084, places=10)

    def test_resealed_summary_inconsistency_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = arithmetic_fixture(root)
            relative = A14 + '/' + FORMAL + 'SUMMARY.json'
            summary = snapshot.source_json(relative)
            summary['joint@6']['shifted']['mse'] += 0.01
            snapshot = replace_and_reseal(root, relative, json.dumps(summary).encode())
            with self.assertRaisesRegex(AuditError, 'SUMMARY.json.joint@6.shifted.mse'):
                numerical_audit(snapshot)

    def test_resealed_metric_inconsistency_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = arithmetic_fixture(root)
            relative = A14 + '/' + FORMAL + 'METRICS.csv'
            reader = csv.DictReader(io.StringIO(snapshot.source_bytes(relative).decode()))
            fields, rows = reader.fieldnames, list(reader)
            rows[0]['final_se'] = str(float(rows[0]['final_se']) + 1)
            output = io.StringIO(newline='')
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            snapshot = replace_and_reseal(root, relative, output.getvalue().encode())
            with self.assertRaisesRegex(AuditError, r'METRICS.csv\[0\].final_se'):
                numerical_audit(snapshot)

    def test_resealed_cost_inconsistency_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = arithmetic_fixture(root)
            relative = A14 + '/' + FORMAL + 'COSTS.json'
            costs = snapshot.source_json(relative)
            costs[0]['process_seconds'] += 0.01
            snapshot = replace_and_reseal(root, relative, json.dumps(costs).encode())
            with self.assertRaisesRegex(AuditError, r'COSTS.json\[0\].process_seconds'):
                numerical_audit(snapshot)

    def test_aggregate_counts_harm_and_preserves_correct_input(self):
        rows = [dict(observation_id='a', initial_ae=2.0, final_ae=1.0, initial_se=4.0, final_se=1.0, failure=False),
                dict(observation_id='a', initial_ae=0.0, final_ae=0.5, initial_se=0.0, final_se=0.25, failure=False),
                dict(observation_id='b', initial_ae=1.0, final_ae=1.0, initial_se=1.0, final_se=1.0, failure=True)]
        result = aggregate(rows)
        self.assertEqual((result['improved'], result['same'], result['harmed'], result['failures']), (1, 1, 1, 1))
        self.assertEqual(result['mse'], 0.75)
        self.assertEqual(result['mae'], 2.5 / 3)

    def test_nonfinite_numeric_comparison_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(AuditError):
                Comparison().equal(1.0, value, 'number')


if __name__ == '__main__':
    unittest.main()
