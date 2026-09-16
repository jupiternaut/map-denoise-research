#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path("/home/grf/.hermes/attachments/outputs/20260914T124618Z")
sys.path.insert(0, str(HERE))
import horizon_experiment as h

sys.path.insert(0, "/home/grf/.hermes/attachments/outputs/20260914T111013Z")
import exact_experiment as e

cat = e.build_catalog(e.CATALOG_BUNDLES, e.CATALOG_SEED0, "catalog", 0)
hist = h.history_unit(cat)
print("history", hist)
assert hist["ok"]
assert hist["example"]["y0"] != hist["example"]["y2"]
assert hist["example"]["orig_unchanged"]
assert hist["example"]["n_tape"] < hist["example"]["n_cur_only"]
print("smoke ok")
