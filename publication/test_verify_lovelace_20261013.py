"""Small independent negative fixtures for the publication verifier."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_lovelace_20261013 import ARCHIVE_ROOT, MANIFEST_PATH, AuditError, Snapshot, metric


class PublicationTests(unittest.TestCase):
    def fixture(self, root, encoding='identity'):
        raw = b'{"answer":42}\n'
        payload = gzip.compress(raw, mtime=0) if encoding == 'gzip' else raw
        relative = ARCHIVE_ROOT+'/example/record.json'+('.gz' if encoding == 'gzip' else '')
        record = dict(path=relative, source='/nonexistent/source/example/record.json',
                      source_relative='example/record.json', encoding=encoding,
                      sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                      source_sha256=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw))
        path = root/relative
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        manifest = root/MANIFEST_PATH
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(dict(archive_root=ARCHIVE_ROOT, files=[record])))
        return path, manifest

    def test_plain_and_gzip_no_source_access(self):
        for encoding in ('identity', 'gzip'):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                self.fixture(root, encoding)
                self.assertEqual(Snapshot(root).seal()['files'], 1)

    def test_changed_payload_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = self.fixture(root)
            path.write_bytes(b'{"answer":43}\n')
            with self.assertRaises(AuditError):
                Snapshot(root).seal()

    def test_wrong_decoded_hash_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, manifest = self.fixture(root, 'gzip')
            data = json.loads(manifest.read_text())
            data['files'][0]['source_sha256'] = '0'*64
            manifest.write_text(json.dumps(data))
            with self.assertRaises(AuditError):
                Snapshot(root).seal()

    def test_unlisted_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path, _ = self.fixture(root)
            path.with_name('extra.txt').write_text('extra')
            with self.assertRaises(AuditError):
                Snapshot(root).seal()

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, manifest = self.fixture(root)
            data = json.loads(manifest.read_text())
            data['files'][0]['path'] = ARCHIVE_ROOT+'/../escape'
            manifest.write_text(json.dumps(data))
            with self.assertRaises(AuditError):
                Snapshot(root)

    def test_exact_metric_includes_damage(self):
        rows = [dict(incumbent_mm='2', truth_mm='0', selected_mm='1'),
                dict(incumbent_mm='0', truth_mm='0', selected_mm='1/2'),
                dict(incumbent_mm='3', truth_mm='2', selected_mm='3')]
        result = metric(rows)
        self.assertEqual((result['improved'], result['unchanged'], result['worsened']), (1, 1, 1))
        self.assertEqual(result['mse_mm2'], '3/4')
        self.assertEqual(result['mae_mm'], '5/6')


if __name__ == '__main__':
    unittest.main()
