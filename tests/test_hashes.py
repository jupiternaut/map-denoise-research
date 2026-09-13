import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1")
sys.path.insert(0, str(ROOT))

from hashutil import sha256_file

FROZEN = {
    OLD / "operators.py": "1240562f61aa98bcca4d54837fe11b6295386efe908420c952e96c406845ae59",
    OLD / "generate_cases.py": "98a586aec3c0b6ce80abb97515c5f1655c85e97fd9a3b94493182b1d803e64ee",
    Path("/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py"):
        "40df63459fe620788acb714f44bd794757f8197542eb03411e685e33ad384f57",
    Path("/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/baseline/scalar_reference.py"):
        "ebc82a17ed431ee10056ef3c64113e8897f10821c673c4c0c99e469ef1773a9a",
}


class HashTests(unittest.TestCase):
    def test_old_checkpoint_unchanged(self):
        for path, expected in FROZEN.items():
            self.assertTrue(path.exists(), path)
            self.assertEqual(sha256_file(path), expected, path)

    def test_download_records_if_present(self):
        log = Path("/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/logs/eth3d_download.json")
        if not log.exists():
            self.skipTest("ETH3D download log not written yet")
        payload = json.loads(log.read_text())
        for rec in payload["files"]:
            if rec.get("status") in {"downloaded", "already_present"}:
                self.assertEqual(sha256_file(Path(rec["path"])), rec["sha256"])


if __name__ == "__main__":
    unittest.main()
