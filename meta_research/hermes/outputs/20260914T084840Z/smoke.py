#!/usr/bin/env python3
import sys
import time
from pathlib import Path

HERE = Path("/home/grf/.hermes/attachments/outputs/20260914T084840Z")
sys.path.insert(0, str(HERE))
import compete_experiment as c

print("COVER head", c.COVER[:8].tolist())
checks = c.library_checks()
for ch in checks:
    if not ch["ok"]:
        print("FAIL", ch)
        raise SystemExit(1)
print("checks_ok", len(checks))
print("iso...", flush=True)
iso = c.dual_env_isolation()
print("iso", iso)
if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
    raise SystemExit("iso fail")

t0 = time.perf_counter()
b = c.make_bundle("dev", 0, c.DEV_SEED0)
print("init", sorted(b[0].init_obs), "dirty", b[0].dirty_x, "t", b[0].t_break)
for t in b:
    for kind in ("Q", "QR"):
        r = c.collect_frozen(t, kind)
        s16 = r["checkpoints"]["16"]
        print(
            t.cause,
            kind,
            "old",
            round(s16["old"]["mae"], 3),
            s16["old"]["family"],
            "new",
            round(s16["new"]["mae"], 3),
            s16["new"]["family"],
            s16["new"]["isol"],
        )
print("A bundle_s", round(time.perf_counter() - t0, 3))

t0 = time.perf_counter()
for t in b:
    for p in c.POLICIES_B:
        r = c.run_policy_B(t, p)
        f = r["final"]
        print(
            t.cause,
            p,
            "mae16",
            round(f["mae"], 3),
            "fam",
            f["family"],
            "isol",
            f["isol"],
            "ret",
            f["n_retest"],
            "n_score",
            f["n_score"],
            "traj",
            [a["a"] for a in r["traj"][:8]],
        )
print("B bundle_s", round(time.perf_counter() - t0, 3))
