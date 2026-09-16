#!/usr/bin/env python3
import sys
import time

sys.path.insert(0, "/home/grf/.hermes/attachments/outputs/20260914T051316Z")
import conflict_experiment as c

print("INIT", c.INIT_X)
checks = c.library_checks()
for ch in checks:
    print(ch["name"], ch["ok"], ch.get("detail"))
iso = c.dual_env_isolation()
print("iso", iso)
t0 = time.perf_counter()
b = c.make_bundle("dev", 0, c.DEV_SEED0)
for t in b:
    for p in c.POLICIES:
        r = c.run_policy(t, p)
        print(
            t.cause,
            p,
            "mae16",
            r["checkpoints"]["16"]["mae"],
            "exp",
            r["final"]["n_expand"],
            "ret",
            r["final"]["n_retest"],
            "unres",
            r["unresolved"],
            "traj",
            [a["a"] for a in r["traj"]],
        )
print("bundle_s", time.perf_counter() - t0)
