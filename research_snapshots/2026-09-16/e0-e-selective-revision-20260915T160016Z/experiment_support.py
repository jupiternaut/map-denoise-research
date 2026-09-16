"""Frozen adapters, loss accounting, and artifact output. No old files written."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
HISTORY = Path('/home/grf/Documents/Codex/2026-09-15')
DIAG = HISTORY / 'e0-diagnostics-20260915T072044Z'
sys.dont_write_bytecode = True
sys.path.insert(0, str(DIAG))
from common import load_old
from projection import prepared_population, calibration_for
from projection_boundary import linear_projection

OLD = load_old()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_hash(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        if a.dtype.hasobject:
            a = a.astype(str)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def snapshot():
    folders = [HISTORY / 'e0', DIAG, HISTORY / 'CPR2_E0D_FORMAL_REPORT_20260915']
    files = [p for folder in folders for p in folder.rglob('*') if p.is_file()]
    files += [HISTORY / n for n in ('INITIAL_CONSTRUCTION.md', 'CONSTRUCTION_REVISION_1.md')]
    return {str(p): sha(p) for p in sorted(files)}


def calibrate_with_roughness(seed, fbig, kind, cuts, props, alpha):
    from selective_operator import calibration_roughness
    code, ci = int(round(fbig * 100)), list(OLD.CALIBS).index(kind)
    K = OLD.gen_calset(np.random.default_rng([seed, code, 10 + ci, 1]), 2000, kind, props)
    s, m, _, contrast = OLD.scores(K['c'])
    strata = np.digitize(contrast, cuts)
    cal = OLD.calibrate(OLD.interp_rows(s, K['tstar']), m, strata, alpha)
    return cal, calibration_roughness(K['c'], strata), arrays_hash(K['c'], K['tstar'])


def evaluate(theta, xyz0, truth_delta, subset, mm):
    """Evaluator is separate from all decision calls. Units: mm."""
    before = np.abs(truth_delta)
    after = np.abs(theta - truth_delta)
    delta = after - before
    harm, gain = np.maximum(delta, 0), np.maximum(-delta, 0)
    moved = theta != 0
    harmful = delta > 1e-9
    beneficial = delta < -1e-9
    masks = {'ALL': np.ones(len(theta), bool)}
    for label in np.unique(subset):
        masks[str(label)] = subset == label
        masks[str(label) + ':mm'] = (subset == label) & mm
        masks[str(label) + ':single'] = (subset == label) & ~mm
    masks['mm'] = mm
    masks['single'] = ~mm
    metrics = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if not n:
            continue
        nm = int((moved & mask).sum())
        metrics[name] = dict(n=n, moved=nm, harm_count=int((harmful & mask).sum()),
            benefit_count=int((beneficial & mask).sum()), move_rate=nm / n,
            harm_rate=float(harmful[mask].mean()),
            harm_among_moved=float(harmful[mask & moved].mean()) if nm else None,
            harm_sum=float(harm[mask].sum()), gain_sum=float(gain[mask].sum()),
            harm_p95_moved=float(np.percentile(harm[mask & moved], 95)) if nm else None,
            mae=float(after[mask].mean()), rmse=float(np.sqrt(np.mean(after[mask]**2))),
            identity_mae=float(before[mask].mean()), delta_mae=float(delta[mask].mean()),
            weighted_delta=float(delta[mask].sum() / len(theta)),
            half_error_rate=float(np.mean(after[mask] < .5 * before[mask])))
    base_names = list(map(str, np.unique(subset)))
    np.testing.assert_allclose(sum(metrics[k]['weighted_delta'] for k in base_names),
                               metrics['ALL']['delta_mae'], atol=1e-12)
    out = xyz0.copy()
    out[:, 2] += theta
    target = xyz0.copy()
    target[:, 2] += truth_delta
    a = cKDTree(target).query(out, workers=1)[0]
    b = cKDTree(out).query(target, workers=1)[0]
    geometry = dict(nn_accuracy=float(a.mean()), nn_completeness=float(b.mean()),
                    nn_symmetric=float((a.mean()+b.mean())/2),
                    precision_1mm=float(np.mean(a <= 1)), recall_1mm=float(np.mean(b <= 1)))
    # Fixed source-group identity. This proxy is not a topological ground truth.
    layer = {}
    for name, other in [('twosheet_back', 'twosheet_front'), ('twosheet_front', 'twosheet_back')]:
        sel, alt = subset == name, subset == other
        if sel.any() and alt.any():
            own_center, other_center = np.median(target[sel, 2]), np.median(target[alt, 2])
            flip = np.abs(out[sel, 2]-other_center) < np.abs(out[sel, 2]-own_center)
            layer[name] = dict(n=int(sel.sum()), flips=int(flip.sum()), rate=float(flip.mean()))
    return dict(groups=metrics, geometry=geometry, layer_identity_proxy=layer)


def veto_account(proposal, guarded, truth, subset, mm, strata):
    before = np.abs(truth)
    after = np.abs(proposal-truth)
    blocked = (proposal != 0) & (guarded == 0)
    harm, gain = np.maximum(after-before, 0), np.maximum(before-after, 0)
    selections = {'ALL': np.ones(len(truth), bool)}
    selections.update({str(k): subset == k for k in np.unique(subset)})
    selections.update({'base:mm': (subset == 'base') & mm,
                       'big:single:sharp': (subset == 'big') & ~mm & (strata == 2)})
    result = {}
    for name, mask in selections.items():
        m = mask & (proposal != 0)
        b = mask & blocked
        result[name] = dict(proposed=int(m.sum()), blocked=int(b.sum()),
            blocked_fraction=float(b.sum()/m.sum()) if m.any() else None,
            harmful_blocked=int((b & (harm > 1e-9)).sum()),
            beneficial_blocked=int((b & (gain > 1e-9)).sum()),
            avoided_harm=float(harm[b].sum()), lost_gain=float(gain[b].sum()),
            available_gain=float(gain[mask].sum()),
            retained_gain_fraction=float(1-gain[b].sum()/gain[mask].sum()) if gain[mask].sum()>0 else None)
    return result


def write_ply(path, xyz):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
                 'property float x\nproperty float y\nproperty float z\nend_header\n' % len(xyz)).encode())
        f.write(np.asarray(xyz, dtype='<f4').tobytes())
