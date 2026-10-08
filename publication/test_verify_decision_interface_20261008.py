"""Small corruption and strict-gate regressions; temporary fixtures only."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import verify_decision_interface_20261008 as verifier


class PublicationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.archive = self.root / verifier.ARCHIVE
        self.archive.mkdir(parents=True)
        self.data = self.archive / "example.json"
        self.data.write_bytes(b"{}")
        manifest = dict(source_host="liekkas", archive_roots=[verifier.ARCHIVE], excluded=[], files=[
            dict(path=verifier.ARCHIVE + "/example.json", source=verifier.SOURCE + "/example.json",
                 bytes=2, sha256=hashlib.sha256(b"{}").hexdigest(), transformation="none")])
        manifest_path = self.root / verifier.MANIFEST
        manifest_path.parent.mkdir()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_archive_needs_no_source_directory(self):
        self.assertEqual(verifier.check_manifest(self.root), {"files": 1, "bytes": 2})

    def test_same_size_byte_change_fails(self):
        self.data.write_bytes(b"[]")
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 mismatch"):
            verifier.check_manifest(self.root)

    def test_unlisted_file_fails(self):
        (self.archive / "unlisted.json").write_bytes(b"{}")
        with self.assertRaisesRegex(verifier.VerificationError, "path-set mismatch"):
            verifier.check_manifest(self.root)

    def test_missing_file_fails(self):
        self.data.unlink()
        with self.assertRaisesRegex(verifier.VerificationError, "path-set mismatch"):
            verifier.check_manifest(self.root)

    def test_source_identity_change_fails(self):
        path = self.root / verifier.MANIFEST
        data = json.loads(path.read_text(encoding="utf-8"))
        data["files"][0]["source"] = "/other/example.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(verifier.VerificationError, "source identity"):
            verifier.check_manifest(self.root)


class ScientificChecks(unittest.TestCase):
    def test_error_is_recomputed(self):
        row = dict(id="scene", arm="ED_S1_M", initial_depth=600.0, selected_depth=610.0,
                   true_depth=600.0, candidates=[600.0, 610.0], error=10.0, initial_error=0.0,
                   change=10.0, initial_kind="correct", regret=10.0, move=True)
        verifier.recompute_row(row)
        row["error"] = 0.0
        with self.assertRaisesRegex(verifier.VerificationError, "/error"):
            verifier.recompute_row(row)

    def test_sub_practical_harm_still_fails_strict_gate(self):
        rows = [dict(id="scene", arm="ED_S1_M", initial_kind=kind, initial_depth=initial,
                     initial_error=abs(initial - 600.0), error=error,
                     change=error - abs(initial - 600.0), move=True, mechanism="textured_single")
                for kind, initial, error in [("correct", 600.0, 0.7), ("minus", 540.0, 10.0),
                                             ("plus", 660.0, 10.0)]]
        gate = verifier.strict_gate(rows, "confirmation")
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["correct_no_harm"])
        self.assertTrue(gate["checks"]["minus_better_keep"])
        self.assertTrue(gate["checks"]["plus_better_keep"])
        verifier.same(gate, gate, "A recorded scientific FAIL is consistent evidence")


if __name__ == "__main__":
    unittest.main()
