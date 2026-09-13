"""Paired statistical support intervention, explicitly NOT a first-return simulator."""
import numpy as np

SEEDS = tuple(9161101+10*j for j in range(10))


def make(seed, gap, sigma, dependence, n=768):
    if n % 8 or dependence not in (0., .5, 1.):
        raise ValueError('n divisible by 8 and registered dependence levels required')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1601]))
    xy = rng.uniform([-60., -50.], [60., 50.], size=(n, 2))
    eps_unit = rng.normal(size=n)
    scan = np.arange(n) % 2
    labels = np.zeros(n, dtype=int)
    base = labels.copy()
    for f in (0, 1):
        ids = np.flatnonzero(scan == f)
        ranks = ids[np.argsort(xy[ids, 0], kind='stable')]
        base[ranks[len(ids)//2:]] = 1
        labels[ids] = base[ids]
        perm_rng = np.random.default_rng(np.random.SeedSequence([seed, 1602, f]))
        chosen = perm_rng.permutation(ids)[:int(round((1-dependence)*len(ids)))]
        if len(chosen):
            labels[chosen] = base[perm_rng.permutation(chosen)]
        assert labels[ids].sum() == len(ids)//2
    if gap == 0:
        labels[:] = 0
    clean = np.c_[xy, gap*labels]
    observed = clean.copy()
    observed[:, 2] += sigma*eps_unit
    # Normalize XY by the same fixed scale for all cases, independent of label/height.
    x = np.c_[np.ones(n), xy/50.]
    inp = dict(design=x, height_mm=observed[:, 2], xyz_world=observed/1000.,
               scan_id=scan, sigma_mm=np.asarray(sigma))
    truth = dict(gt_layer=labels, gt_clean_xyz_world=clean/1000., eps_mm=sigma*eps_unit,
                 true_slope=np.zeros(2), true_means_mm=np.array([0., gap]) if gap else np.zeros(1),
                 base_layer=base, nominal_lambda=np.asarray(dependence), true_gap_mm=np.asarray(gap))
    return inp, truth
