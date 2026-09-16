#!/usr/bin/env python3
import sys
import time
from pathlib import Path

HERE = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(HERE))
import exact_experiment as e

print("N", e.N, "H0", len(e.H0), "H1", len(e.H1), "COVER", e.COVER.tolist())
cat = e.build_catalog(e.CATALOG_BUNDLES, e.CATALOG_SEED0, "catalog", 0)
print("catalog worlds", len(cat))
checks = e.library_checks(cat)
for ch in checks:
    print(ch["name"], ch["ok"], ch.get("detail"))
    if not ch["ok"]:
        raise SystemExit("check fail")
print("iso...")
iso = e.dual_env_isolation(cat)
print("iso", iso)
if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
    raise SystemExit("iso fail")

w = cat[0]
env = e.TapeEnv(w)
e.apply_init(env)
print("init orig", env.orig, "cur", env.cur)
x = e.next_cover(env.orig)
if x is not None:
    env.query(x)
before = dict(env.orig)
z = next(iter(env.orig))
env.retest(z)
print("after retest orig", env.orig, "cur", env.cur, "orig_stable", env.orig == before)

t0 = time.perf_counter()
dp = e.ExactDP(cat)
env0 = e.TapeEnv(cat[0])
e.apply_init(env0)
ids = frozenset(
    u.wid for u in cat if all(u.observe(x) == env0.cur[x] for x in env0.orig)
)
v, act = dp.value(ids, frozenset(env0.orig), frozenset(), e.MAX_BUDGET - env0.cost)
print("dp one-root", v, act, "states", len(dp._memo), "s", round(time.perf_counter() - t0, 3))

t0 = time.perf_counter()
for p in ("cover", "retest_then_cover", "lookahead"):
    r = e.run_policy(w, p, None, frozenset())
    print(w.cause, p, "mae6", r["final"]["mae"], "isol", r["final"]["isol"], "orig_ok", r["orig_untouched_by_retest"])
print("fixed policies s", round(time.perf_counter() - t0, 3))
