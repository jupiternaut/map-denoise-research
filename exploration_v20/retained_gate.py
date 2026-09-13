"""V19 gate-search repair; fixed family and unchanged final geometry action."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exploration_v18'))
import v18_operator as v18
base = v18.base
spec = importlib.util.spec_from_file_location('v19a_frozen', ROOT / 'exploration_v19/track_a/operator.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def family_pool(diagnostic, x, y, sigma):
    pool = {k: v18.arrays(v) for k, v in diagnostic['pool'].items()}
    legacy = v18.arrays(diagnostic['old'])
    if legacy['k'] == 2:
        candidate = dict(legacy, family='R', feature_center=pool['R']['feature_center'],
                         feature_scale=pool['R']['feature_scale'], parameter_count=16,
                         candidate_origin='old_spatial36', solver_diagnostics=[])
        candidate.update(base.evaluate(candidate, x, y, sigma))
        candidate['nll'] = float(-candidate['log_density'].sum())
        candidate['ridge_penalty'] = float(base.RIDGE * np.sum(candidate['gate'][1:] ** 2))
        candidate['bic'] = 2*candidate['nll'] + candidate['ridge_penalty'] + 16*np.log(len(y))
        if candidate['bic'] < pool['R']['bic']:
            pool['R'] = candidate
    return pool


def training_family(x, y, sigma, family):
    _, diagnostic = v18.construct(x, y, sigma)
    return family_pool(diagnostic, x, y, sigma)[family], diagnostic


def crossfit_priors(x, y, sigma, family):
    folds = base.folds_from_x(x, 3)
    prior = np.empty((len(y), 2))
    models, records = [], []
    for fold in range(3):
        valid = folds == fold
        train = ~valid
        model, full = training_family(x[train], y[train], sigma, family)
        prior[valid] = old.prior_at(model, x[valid])
        models.append(model)
        records.append(dict(train_indices=np.flatnonzero(train), validation_indices=np.flatnonzero(valid),
                            model=model, full_diagnostic=full))
    return prior, models, dict(folds=folds, training=records)


def run(x, y, sigma, baseline, full_diagnostic):
    x, y = old.validate(x, y, sigma)
    if baseline['k'] == 1:
        return {'repaired_insample_map': dict(baseline), 'repaired_crossfit_map': dict(baseline)}, dict(single_passthrough=True)
    family = baseline['family']
    gate = family_pool(full_diagnostic, x, y, sigma)[family]
    np.testing.assert_allclose(gate['prediction'], baseline['prediction'], rtol=0, atol=1e-8)
    prior = old.prior_at(gate, x)
    inside = old.fit_geometry(x, y, sigma, prior, baseline, [gate])
    cross_prior, models, folds = crossfit_priors(x, y, sigma, family)
    cross = old.fit_geometry(x, y, sigma, cross_prior, baseline, models)
    for model in (inside, cross):
        model.update(family=family, truth_fields_used=[], full_pipeline_independent=False)
    return {'repaired_insample_map': inside, 'repaired_crossfit_map': cross}, dict(insample_gate=gate, crossfit=folds)
