"""GT-free runtime. All scene data enter as point/camera arrays, never paths.

The one file-based API, load_models, reads packaged trusted model artifacts.
Dataset loading, ROI definition and ordered source selection are adapter duties.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from .graph_field import estimate_normals, interpolation_weights, interpolate
from .surfacelet import score_surfacelets, solve_surfacelets

PARAMS = dict(loss='squared_error', learning_rate=.08, max_iter=80,
              max_leaf_nodes=7, max_depth=3, min_samples_leaf=80,
              l2_regularization=1., random_state=20260922, early_stopping=False)
MODELS = {'pre_A': ('pre', 'A_gain'), 'post_A': ('A_all_post', 'A_gain'),
          'post_B': ('B_all_post', 'B_gain')}
SEED = 20260922
POST_KEYS = ('offset', 'hypothesis', 'training_cost', 'mode_gap', 'depth_curvature',
             'cost_at_zero', 'source_agreement', 'source_cost',
             'heldout_incumbent_cost', 'heldout_proposal_cost', 'valid',
             'cost_at_zero_valid', 'mode_gap_valid', 'depth_curvature_valid',
             'source_valid', 'heldout_incumbent_valid', 'heldout_proposal_valid',
             'source_agreement_valid')


def schema():
    return json.loads(Path(__file__).with_name('FEATURE_SCHEMA.json').read_text())


def load_models(directory=None):
    """Load only trusted release joblib files; validate their frozen hashes."""
    directory = Path(directory) if directory is not None else Path(__file__).with_name('models')
    manifest = json.loads((directory / 'MODEL_LOCK.json').read_text())
    result = {}
    for name in MODELS:
        path = directory / (name + '.joblib')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['models'][name]['sha256']:
            raise ValueError('model hash mismatch: ' + name)
        result[name] = joblib.load(path)
    return result


def policy_masks(predictions):
    n = len(predictions['pre_A'])
    return {
        'pre_A_keep': (predictions['pre_A'] > 0).astype(np.int8),
        'post_A_keep': (predictions['post_A'] > 0).astype(np.int8),
        'post_AB_keep': np.argmax(np.column_stack([
            np.zeros(n), predictions['post_A'], predictions['post_B']
        ]), axis=1).astype(np.int8),
    }


def apply_arrays(points, a, b, features, models=None, seed=SEED):
    """Exact V28 routing. Primary is post_A_keep; post_AB_keep is secondary."""
    points, a, b = (np.asarray(v) for v in (points, a, b))
    if points.ndim != 2 or points.shape[1] != 3 or any(v.shape != points.shape for v in (a, b)):
        raise ValueError('points, a and b must be matching [N,3] arrays')
    if not all(np.isfinite(v).all() for v in (points, a, b)):
        raise ValueError('geometry must be finite')
    definition = schema()
    for key, _ in MODELS.values():
        x = np.asarray(features[key])
        if x.shape != (len(points), len(definition[key])) or not np.isfinite(x).all():
            raise ValueError('feature schema or finite-value violation: ' + key)
    models = load_models() if models is None else models
    with threadpool_limits(limits=1):
        pred = {name: models[name].predict(features[key]) for name, (key, _) in MODELS.items()}
    masks = policy_masks(pred)
    random = np.zeros(len(points), dtype=np.int8)
    count = int(np.sum(masks['post_A_keep'] != 0))
    random[np.random.default_rng(seed).choice(len(points), count, replace=False)] = 1
    masks['random_A_keep'] = random
    outputs = {name: np.where((mask == 1)[:, None], a,
                             np.where((mask == 2)[:, None], b, points))
               for name, mask in masks.items()}
    return outputs, pred, masks


def _post_features(solution, pre):
    pieces = [pre]
    for key in POST_KEYS:
        if key not in solution:
            continue
        value = np.asarray(solution[key]).reshape(len(pre), -1).astype(float)
        pieces.append(np.column_stack([
            np.nan_to_num(value, nan=0., posinf=1e4, neginf=-1e4), ~np.isfinite(value)]))
    return np.column_stack(pieces)


def _construct(points, reference, sources, batch_size):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('points must be finite [N,3], with N >= 3, in millimetres')
    if len(sources) != 4:
        raise ValueError('exactly four ordered source cameras are required')
    rays = p - np.asarray(reference['center'])
    lengths = np.linalg.norm(rays, axis=1, keepdims=True)
    if (lengths <= 1e-12).any():
        raise ValueError('points must not coincide with reference camera center')
    rays /= lengths
    ns = [estimate_normals(p, k) for k in (8, 24, 64)]
    optical = np.asarray(reference['P'])[2, :3]
    optical = optical / np.linalg.norm(optical)
    nb = np.stack(ns + [np.broadcast_to(optical, p.shape)], axis=1)
    _, ids = np.unique(np.floor((p-p.min(axis=0))/1.5).astype(np.int64), axis=0, return_index=True)
    if len(ids) > 1000:
        ids = np.random.default_rng(SEED).choice(ids, 1000, replace=False)
    ids = np.sort(ids)
    pa, bn = p[ids], nb[ids]
    offsets = np.arange(-6, 6.00001, .25)
    ev = score_surfacelets(pa, bn, offsets, reference, sources, batch_size=batch_size)
    sol = solve_surfacelets(ev['scores'], offsets)
    interp_ids, weights = interpolation_weights(p, nb[:, 1], pa, bn[:, 1])
    angles = []
    for cam in sources:
        direction = pa-np.asarray(cam['center'])
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        angles.append(np.arccos(np.clip(np.sum(direction*ev['directions'], axis=1), -1, 1)))
    distance, _ = cKDTree(p).query(pa, k=min(24, len(p)), workers=1)
    pre = np.column_stack([
        np.sqrt(ev['ref_variance'][:, 0]), ev['ref_fraction'][:, 0],
        abs(np.sum(bn[:, 0]*bn[:, 2], axis=1)), abs(np.sum(bn[:, 1]*bn[:, 0], axis=1)),
        abs(np.sum(bn[:, 1]*ev['directions'], axis=1)), np.median(distance[:, 1:], axis=1),
        np.mean(angles, axis=0), np.max(angles, axis=0)])
    pre = np.nan_to_num(pre, nan=0., posinf=1e4, neginf=-1e4)
    features = {'pre': np.sum(pre[interp_ids]*weights[..., None], axis=1).astype(np.float32)}
    outputs = {'identity': p.copy()}
    state = {'anchor_ids': ids, 'offsets': offsets, 'normal_bank': bn,
             'interpolation_ids': interp_ids, 'interpolation_weights': weights}
    for arm in ('A_all', 'B_all', 'A_fit', 'B_fit'):
        s = sol[arm]
        full = interpolate(s['offset'], interp_ids, weights)
        q = p+full[:, None]*rays
        outputs[arm] = q
        state[arm+'_offset'] = full
        post = _post_features(s, pre)
        features[arm+'_post'] = np.sum(post[interp_ids]*weights[..., None], axis=1).astype(np.float32)
        if arm.endswith('fit'):
            old, new = s['heldout_incumbent_cost'], s['heldout_proposal_cost']
            pass_anchor = (s['heldout_incumbent_valid'].all(1) &
                           s['heldout_proposal_valid'].all(1) & (new < old).all(1))
            keep = np.all(pass_anchor[interp_ids] | (weights == 0), axis=1)
            outputs[arm+'_reserved'] = np.where(keep[:, None], q, p)
            state[arm+'_reserved_accept'] = keep
    return outputs, features, state


def construct(points, reference, sources, batch_size=32):
    """Frozen full V28 array constructor, single CPU thread.

    Each camera has image (float grayscale [H,W] at the frozen half resolution),
    P (physical 3x4 projection with matching pixel convention), center (3, mm).
    Caller fixes one reference and four ordered sources before evaluation.
    Image RGB-to-gray and half-resize conventions are documented in README.
    """
    with threadpool_limits(limits=1):
        return _construct(points, reference, sources, batch_size)
