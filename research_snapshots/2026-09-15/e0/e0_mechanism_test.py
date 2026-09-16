#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E0: pre-registered synthetic mechanism test for the CPR-2 operator
(confidence-set projection revision). Specification: PROTOCOL.md in this
directory (its sha256 is recorded in COMMANDS.md before any run).

Truth t* is used ONLY (a) to compute calibration scores on K / K' and (b) in
evaluation metrics. The decision path for the test set I (`calibrate`,
`decide`, `build_arms`, `apply_guard`) never receives the truth of I.

Implementation choices that the protocol leaves open (documented here, fixed
before the first run):
  * grid index i -> (ix = i % Nx, iy = i // Nx), x = ix*h, y = iy*h.
  * two-sheet region = columns [Nx//2 - c//2, Nx//2 + c//2) with c = Nx//10
    (20 columns for Nx=200); checkerboard (ix+iy) even -> BACK, odd -> FRONT.
  * big patches: integer radius r ~ U{3,4,5,6}; a patch is accepted only if
    it brings the big count closer to the target; placement retries avoid
    overlap with the region (hard) and with previous patches (soft);
    U(2,8) magnitude drawn per cell, sign per patch.
  * width of a level-set component with grid indices a..b (inclusive):
    w = (b - a + 1) * 0.05 (grid samples treated as cells of width 0.05).
  * dist(0, component) = distance from t=0 to the nearest grid point of the
    component (0 if the component contains t=0).
  * voxel key for set metrics = round(coord / 0.8) (robust to float error at
    the grid coordinates, which are exact multiples of 0.8).
  * smooth noise: white N(0,1) on an extended grid convolved with a Gaussian
    kernel (sigma = 0.4 / 0.05 = 8 samples, truncated at 4 sigma), kernel
    normalised so that sum(k^2) = 1 (unit marginal std), then the central 481
    samples are kept (no edge effects).
  * Wilson tolerance: a coverage "passes" iff the 95% Wilson upper bound of the
    observed proportion is >= the nominal 1-alpha; it "falls below" iff the
    upper bound is < nominal. A false rate "<= alpha_abs + tol" iff the Wilson
    lower bound is <= alpha_abs.
  * sanity anomaly (0 in L after rejection): theta = 0, out = KEEP, counted.
"""
import argparse
import csv
import gzip
import json
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import convolve1d
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
# Pre-registered constants (PROTOCOL.md)
# ----------------------------------------------------------------------------
H = 0.8
T_STEP = 0.05
NT = 481
TGRID = np.linspace(-12.0, 12.0, NT)
IDX0 = NT // 2  # t = 0
assert abs(TGRID[IDX0]) < 1e-12 and abs(TGRID[1] - TGRID[0] - T_STEP) < 1e-12
W_DIP = 1.0
SIG_EPS = 0.03
NOISE_SIGMA_T = 0.4
FRONT_Z = 6.0
BACK_WEAK = 0.3
SIG_TSTAR = 0.2
SIG_B = 0.15
GEN_PROBS = (0.30, 0.35, 0.35)
STRATA = ('flat', 'medium', 'sharp')
GEN_RANGES = {'flat': (0.02, 0.08), 'medium': (0.08, 0.25), 'sharp': (0.25, 0.80)}
SUBSETS = ('base', 'big', 'twosheet_back', 'twosheet_front', 'misfit')
MM_FRAC = 0.15
MISFIT_FRAC = 0.03
ALPHA_TEST = 0.10
ALPHA_ABS = 0.05
BH_Q = 0.10
EPS_TOL = 0.5
TAUS = (1.0, 2.0, 3.0)
D_GUARD = 0.4
VOXEL = 0.8
MIN_STRATUM = 50
F_BIGS = (0.01, 0.10)
CALIBS = ('exch', 'sfm')
ALPHA_TGTS = (0.20, 0.50)
ARM_ORDER = ('IDENTITY', 'ARGMIN', 'GATED1', 'GATED2', 'GATED3',
             'TEST_ARGMIN', 'CPR2', 'CPR2_G', 'TEST_ARGMIN_G')
Z95 = 1.959963984540054

_SIG_STEPS = NOISE_SIGMA_T / T_STEP
_HALF = int(math.ceil(4 * _SIG_STEPS))
_KERN = np.exp(-0.5 * (np.arange(-_HALF, _HALF + 1) / _SIG_STEPS) ** 2)
_KERN = _KERN / np.sqrt(np.sum(_KERN ** 2))


def scenario_key(f_big, calib, alpha_tgt):
    return 'fbig%.2f_%s_atgt%.1f' % (f_big, calib, alpha_tgt)


# ----------------------------------------------------------------------------
# Generative model
# ----------------------------------------------------------------------------
def smooth_noise(rng, n, chunk=4000):
    out = np.empty((n, NT))
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        white = rng.standard_normal((i1 - i0, NT + 2 * _HALF))
        full = convolve1d(white, _KERN, axis=1, mode='constant')
        out[i0:i1] = full[:, _HALF:_HALF + NT]
    return out


def draw_curve_params(rng, subset, mm, tstar, force_sharp=False):
    n = len(tstar)
    if force_sharp:
        gen = np.full(n, 2, dtype=int)
    else:
        gen = rng.choice(3, size=n, p=GEN_PROBS)
    A = np.empty(n)
    for k, name in enumerate(STRATA):
        lo, hi = GEN_RANGES[name]
        sel = gen == k
        A[sel] = rng.uniform(lo, hi, int(sel.sum()))
    b = rng.normal(0.0, SIG_B, n)
    mu = tstar + b
    mis = subset == 'misfit'
    mu[mis] = rng.uniform(-10.0, 10.0, int(mis.sum()))
    M = np.zeros(n)
    M[mis] = rng.uniform(0.3, 0.6, int(mis.sum()))
    Delta = np.zeros(n)
    r = np.zeros(n)
    nmm = int(mm.sum())
    Delta[mm] = rng.choice(np.array([-1.0, 1.0]), nmm) * rng.uniform(4.0, 8.0, nmm)
    r[mm] = rng.uniform(0.7, 1.05, nmm)
    return dict(gen=gen, A=A, b=b, mu=mu, M=M, Delta=Delta, r=r)


def make_curves(rng, tstar, subset, mm, prm, chunk=4000):
    n = len(tstar)
    inv = 1.0 / (2.0 * W_DIP ** 2)
    t = TGRID[None, :]
    c = np.empty((n, NT))
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        mu = prm['mu'][i0:i1]
        B = np.exp(-(t - mu[:, None]) ** 2 * inv)
        m = mm[i0:i1]
        if m.any():
            mu2 = (prm['mu'][i0:i1][m] + prm['Delta'][i0:i1][m])[:, None]
            B2 = prm['r'][i0:i1][m][:, None] * np.exp(-(t - mu2) ** 2 * inv)
            B[m] = np.maximum(B[m], B2)
        back = subset[i0:i1] == 'twosheet_back'
        if back.any():
            B1 = np.exp(-(t - (FRONT_Z + prm['b'][i0:i1][back])[:, None]) ** 2 * inv)
            B2 = BACK_WEAK * np.exp(-(t - tstar[i0:i1][back][:, None]) ** 2 * inv)
            B[back] = np.maximum(B1, B2)
        c[i0:i1] = prm['M'][i0:i1][:, None] + prm['A'][i0:i1][:, None] * (1.0 - B)
    c += SIG_EPS * smooth_noise(rng, n)
    return c


def disk_cells(ix, iy, cx, cy, r):
    dx = ix - cx
    dy = iy - cy
    return np.flatnonzero(dx * dx + dy * dy <= r * r)


def place_big_patches(rng, ix, iy, in_region, target, nx, ny):
    N = len(ix)
    big = np.zeros(N, dtype=bool)
    sign = np.zeros(N)
    pid = np.full(N, -1, dtype=int)
    n_p = 0
    n_overlap = 0
    count = 0
    full_size = {}
    for r in range(3, 7):
        g = np.arange(-r, r + 1)
        full_size[r] = int((g[:, None] ** 2 + g[None, :] ** 2 <= r * r).sum())
    while count < target:
        r = int(rng.integers(3, 7))
        if count + full_size[r] - target > target - count:
            r = 3
            if count + full_size[3] - target > target - count:
                break
        placed = False
        for attempt in range(400):
            cx = int(rng.integers(0, nx))
            cy = int(rng.integers(0, ny))
            cells = disk_cells(ix, iy, cx, cy, r)
            if in_region[cells].any():
                continue
            overlap = bool(big[cells].any())
            if overlap and attempt < 300:
                continue
            new = cells[~big[cells]]
            if len(new) == 0:
                continue
            s = 1.0 if rng.random() < 0.5 else -1.0
            big[new] = True
            sign[new] = s
            pid[new] = n_p
            n_p += 1
            n_overlap += int(overlap)
            count = int(big.sum())
            placed = True
            break
        if not placed:
            break
    return big, sign, pid, n_p, n_overlap


def gen_population(rng, nx, ny, f_big):
    N = nx * ny
    idx = np.arange(N)
    ix = idx % nx
    iy = idx // nx
    x = ix * H
    y = iy * H
    region_cols = max(1, nx // 10)
    col0 = nx // 2 - region_cols // 2
    col1 = col0 + region_cols
    in_region = (ix >= col0) & (ix < col1)
    subset = np.full(N, 'base', dtype=object)
    z0 = np.zeros(N)
    back = in_region & (((ix + iy) % 2) == 0)
    front = in_region & ~back
    subset[back] = 'twosheet_back'
    subset[front] = 'twosheet_front'
    z0[front] = FRONT_Z

    target = int(round(f_big * N))
    big, sign, pid, n_p, n_overlap = place_big_patches(rng, ix, iy, in_region, target, nx, ny)
    subset[big] = 'big'

    cand = np.flatnonzero(~in_region & ~big)
    n_mis = int(round(MISFIT_FRAC * int((~in_region).sum())))
    mis = rng.choice(cand, size=n_mis, replace=False)
    subset[mis] = 'misfit'

    tstar = rng.normal(0.0, SIG_TSTAR, N)
    bidx = np.flatnonzero(big)
    tstar[bidx] = sign[bidx] * rng.uniform(2.0, 8.0, len(bidx)) + rng.normal(0.0, SIG_TSTAR, len(bidx))

    mm = np.zeros(N, dtype=bool)
    elig = (subset == 'base') | (subset == 'big')
    mm[elig] = rng.random(int(elig.sum())) < MM_FRAC

    prm = draw_curve_params(rng, subset, mm, tstar)
    c = make_curves(rng, tstar, subset, mm, prm)
    pop = dict(N=N, nx=nx, ny=ny, ix=ix, iy=iy, x=x, y=y, z0=z0, in_region=in_region,
               region_cols=(col0, col1), subset=subset, mm=mm, tstar=tstar, patch_id=pid, c=c)
    pop.update(prm)
    pop['realized'] = dict(
        N=N, f_big_target=f_big, n_big_target=target, n_big=int(big.sum()),
        f_big_realized=float(big.sum()) / N, n_patches=n_p, n_patches_with_overlap=n_overlap,
        n_twosheet_back=int(back.sum()), n_twosheet_front=int(front.sum()),
        n_misfit=int(n_mis), n_mm=int(mm.sum()), n_base=int((subset == 'base').sum()),
        mm_frac_of_base_big=float(mm.sum()) / max(1, int(elig.sum())),
        region_x_range=[float(x[in_region].min()), float(x[in_region].max())],
        gen_stratum_counts={name: int((prm['gen'] == k).sum()) for k, name in enumerate(STRATA)})
    return pop


def gen_calset(rng, n, calib, props):
    if calib == 'exch':
        p = np.array([props[s] for s in SUBSETS])
        p = p / p.sum()
        subset = rng.choice(np.array(SUBSETS, dtype=object), size=n, p=p)
        tstar = rng.normal(0.0, SIG_TSTAR, n)
        big = subset == 'big'
        nb = int(big.sum())
        tstar[big] = rng.choice(np.array([-1.0, 1.0]), nb) * rng.uniform(2.0, 8.0, nb) + rng.normal(0.0, SIG_TSTAR, nb)
        mm = np.zeros(n, dtype=bool)
        elig = (subset == 'base') | big
        mm[elig] = rng.random(int(elig.sum())) < MM_FRAC
        prm = draw_curve_params(rng, subset, mm, tstar)
    elif calib == 'sfm':
        subset = np.full(n, 'base', dtype=object)
        tstar = rng.normal(0.0, SIG_TSTAR, n)
        mm = np.zeros(n, dtype=bool)
        prm = draw_curve_params(rng, subset, mm, tstar, force_sharp=True)
    else:
        raise ValueError(calib)
    c = make_curves(rng, tstar, subset, mm, prm)
    out = dict(n=n, subset=subset, mm=mm, tstar=tstar, c=c)
    out.update(prm)
    return out


# ----------------------------------------------------------------------------
# Scores, stratification, calibration
# ----------------------------------------------------------------------------
def scores(c):
    m = c.min(axis=1)
    s = c - m[:, None]
    amin = c.argmin(axis=1)
    Ahat = c.max(axis=1) - m
    return s, m, TGRID[amin], Ahat


def interp_rows(s, t):
    u = (t - TGRID[0]) / T_STEP
    u = np.clip(u, 0.0, NT - 1.0)
    j = np.minimum(np.floor(u).astype(int), NT - 2)
    f = u - j
    rows = np.arange(len(t))
    return s[rows, j] * (1.0 - f) + s[rows, j + 1] * f


def conf_q(sorted_vals, alpha):
    n = len(sorted_vals)
    k = int(math.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return float('inf')
    return float(sorted_vals[k - 1])


def calibrate(s_true_K, m_K, strat_K, alpha_tgt):
    cal = {}
    for k, name in enumerate(STRATA):
        sel = strat_K == k
        fallback = int(sel.sum()) < MIN_STRATUM
        use = np.ones_like(sel) if fallback else sel
        srt = np.sort(s_true_K[use])
        mrt = np.sort(m_K[use])
        cal[name] = dict(n_stratum=int(sel.sum()), n_used=int(use.sum()), fallback=bool(fallback),
                         sorted_s=srt,
                         q_test=conf_q(srt, ALPHA_TEST), q_tgt=conf_q(srt, alpha_tgt),
                         q_abs=conf_q(mrt, ALPHA_ABS))
    return cal


# ----------------------------------------------------------------------------
# Decisions (no truth of I enters here)
# ----------------------------------------------------------------------------
def bh_reject(p, q):
    n = len(p)
    order = np.argsort(p, kind='stable')
    ps = p[order]
    thr = q * np.arange(1, n + 1) / n
    ok = np.flatnonzero(ps <= thr)
    rej = np.zeros(n, dtype=bool)
    if len(ok):
        rej[order[:ok[-1] + 1]] = True
    return rej


def level_set_geometry(L, s, qtgt_i):
    n = L.shape[0]
    pad = np.zeros((n, 1), dtype=bool)
    starts = L & ~np.concatenate([pad, L[:, :-1]], axis=1)
    ends = L & ~np.concatenate([L[:, 1:], pad], axis=1)
    n_comp = starts.sum(axis=1)
    a_near = np.zeros(n, dtype=int)
    b_near = np.zeros(n, dtype=int)
    pi = np.zeros(n)
    g = np.zeros(n)
    single = n_comp == 1
    a_near[single] = starts[single].argmax(axis=1)
    b_near[single] = ends[single].argmax(axis=1)
    tau_g = np.maximum(qtgt_i / 2.0, 1e-6)
    for i in np.flatnonzero(n_comp >= 2):
        a = np.flatnonzero(starts[i])
        b = np.flatnonzero(ends[i])
        d = np.where(a > IDX0, a - IDX0, np.where(b < IDX0, IDX0 - b, 0))
        j = int(np.argmin(d))
        a_near[i] = a[j]
        b_near[i] = b[j]
        om = np.exp(-s[i] / tau_g[i]) * L[i]
        cs = np.concatenate([[0.0], np.cumsum(om)])
        Wc = cs[b + 1] - cs[a]
        tot = Wc.sum()
        others = np.ones(len(a), dtype=bool)
        others[j] = False
        Wo = Wc[others].sum()
        pi[i] = Wo / tot if tot > 0 else 0.0
        g[i] = float((Wc[others] * d[others] * T_STEP).sum() / Wo) if Wo > 0 else 0.0
    w = (b_near - a_near + 1) * T_STEP
    return n_comp, a_near, b_near, w, pi, g


def decide(s, m, strat, cal):
    n = s.shape[0]
    s0 = s[:, IDX0]
    p = np.empty(n)
    qtgt_i = np.empty(n)
    qabs_i = np.empty(n)
    qtest_i = np.empty(n)
    for k, name in enumerate(STRATA):
        sel = strat == k
        srt = cal[name]['sorted_s']
        nk = len(srt)
        cnt = nk - np.searchsorted(srt, s0[sel], side='left')
        p[sel] = (1.0 + cnt) / (nk + 1.0)
        qtgt_i[sel] = cal[name]['q_tgt']
        qabs_i[sel] = cal[name]['q_abs']
        qtest_i[sel] = cal[name]['q_test']
    rej = bh_reject(p, BH_Q)
    L = s <= qtgt_i[:, None]
    n_comp, a_near, b_near, w, pi, g = level_set_geometry(L, s, qtgt_i)
    abs_bad = m > qabs_i
    cand = (~abs_bad) & rej & (n_comp == 1)
    zero_in_L = L[:, IDX0]
    anomaly = cand & zero_in_L
    mv = cand & ~zero_in_L
    theta_end = np.where(a_near > IDX0, TGRID[a_near], TGRID[b_near])
    theta = np.zeros(n)
    theta[mv] = theta_end[mv]
    out = np.full(n, 'KEEP', dtype=object)
    out[mv] = 'MOVE'
    res = np.full(n, 'NONE', dtype=object)
    res[n_comp >= 2] = 'ACQUIRE'
    res[abs_bad] = 'RESPECIFY'
    return dict(p=p, rej=rej, L=L, n_comp=n_comp, w=w, out=out, theta=theta, res=res,
                prio=pi * g, pi=pi, g=g, anomaly=anomaly, s0=s0,
                qtgt_i=qtgt_i, qtest_i=qtest_i, qabs_i=qabs_i, a_near=a_near, b_near=b_near)


def apply_guard(theta, x, y, z0):
    pts = np.column_stack([x, y, z0 + theta])
    moved_idx = np.flatnonzero(theta != 0)
    collapse = np.zeros(len(theta), dtype=bool)
    theta_g = theta.copy()
    nn = np.full(len(moved_idx), np.nan)
    if len(moved_idx):
        tree = cKDTree(pts)
        d, _ = tree.query(pts[moved_idx], k=2)
        nn = d[:, 1]
        flag = nn < D_GUARD
        theta_g[moved_idx[flag]] = 0.0
        collapse[moved_idx[flag]] = True
    return theta_g, collapse, nn


def build_arms(t_argmin, out, theta_cpr2, x, y, z0):
    arms = {}
    arms['IDENTITY'] = np.zeros_like(t_argmin)
    arms['ARGMIN'] = t_argmin.copy()
    for tau in TAUS:
        arms['GATED%d' % int(tau)] = np.where(np.abs(t_argmin) > tau, t_argmin, 0.0)
    mv = out == 'MOVE'
    arms['TEST_ARGMIN'] = np.where(mv, t_argmin, 0.0)
    arms['CPR2'] = theta_cpr2.copy()
    guard = {}
    for base in ('CPR2', 'TEST_ARGMIN'):
        th, col, nn = apply_guard(arms[base], x, y, z0)
        arms[base + '_G'] = th
        guard[base + '_G'] = dict(collapse=col, nn=nn, base=base)
    return arms, guard


# ----------------------------------------------------------------------------
# Metrics (truth enters here)
# ----------------------------------------------------------------------------
def wilson(k, n):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    ph = k / n
    z2 = Z95 * Z95
    den = 1.0 + z2 / n
    c = (ph + z2 / (2 * n)) / den
    hw = Z95 * math.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n)) / den
    return ph, c - hw, c + hw


def wilson_dict(flags):
    flags = np.asarray(flags, dtype=bool)
    n = int(flags.size)
    k = int(flags.sum())
    rate, lo, hi = wilson(k, n)
    return dict(n=n, k=k, rate=rate, lo=lo, hi=hi)


def arm_block(theta, tstar, sel, big_mask):
    th = theta[sel]
    ts = tstar[sel]
    n = int(th.size)
    nan = float('nan')
    if n == 0:
        keys = ['n_moved', 'n_dmg', 'dmg_rate_N', 'dmg_rate_moved', 'dmg_mag_mean_moved',
                'dmg_mag_p95_moved', 'dmg_mag_max', 'dmg_sum', 'e_after_mean', 'e_after_rmse',
                'e_after_median', 'e_before_mean', 'n_big', 'big_e_after_mean', 'big_e_before_mean',
                'big_repair_cov']
        d = {k: nan for k in keys}
        d['n'] = 0
        return d
    e_b = np.abs(ts)
    e_a = np.abs(th - ts)
    moved = th != 0
    dmg = e_a > e_b + 1e-9
    d = np.maximum(e_a - e_b, 0.0)
    nm = int(moved.sum())
    nd = int(dmg.sum())
    blk = dict(n=n, n_moved=nm, n_dmg=nd,
               dmg_rate_N=nd / n,
               dmg_rate_moved=(nd / nm) if nm else nan,
               dmg_mag_mean_moved=float(d[moved].mean()) if nm else nan,
               dmg_mag_p95_moved=float(np.percentile(d[moved], 95)) if nm else nan,
               dmg_mag_max=float(d.max()),
               dmg_sum=float(d.sum()),
               e_after_mean=float(e_a.mean()),
               e_after_rmse=float(np.sqrt(np.mean(e_a ** 2))),
               e_after_median=float(np.median(e_a)),
               e_before_mean=float(e_b.mean()))
    b = big_mask[sel]
    nb = int(b.sum())
    blk['n_big'] = nb
    if nb:
        blk['big_e_after_mean'] = float(e_a[b].mean())
        blk['big_e_before_mean'] = float(e_b[b].mean())
        blk['big_repair_cov'] = float(np.mean(e_a[b] < 0.5 * e_b[b]))
    else:
        blk['big_e_after_mean'] = nan
        blk['big_e_before_mean'] = nan
        blk['big_repair_cov'] = nan
    return blk


def dmg_count(theta, tstar, sel):
    e_b = np.abs(tstar[sel])
    e_a = np.abs(theta[sel] - tstar[sel])
    n = int(sel.sum())
    nd = int((e_a > e_b + 1e-9).sum())
    return dict(n_moved=n, n_dmg=nd, dmg_rate=(nd / n) if n else float('nan'),
                dmg_sum=float(np.maximum(e_a - e_b, 0.0).sum()) if n else 0.0)


def voxel_dedup(pts):
    key = np.round(pts / VOXEL).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, pts)
    cnt = np.bincount(inv, minlength=len(uniq))
    return sums / cnt[:, None]


def set_metrics(O_pts, T_pts):
    O = voxel_dedup(O_pts)
    T = voxel_dedup(T_pts)
    acc = float(cKDTree(T).query(O)[0].mean())
    comp = float(cKDTree(O).query(T)[0].mean())
    return dict(accuracy=acc, completeness=comp, n_O=int(len(O)), n_T=int(len(T)))


def sheet_flips(theta, subset, z0):
    back = subset == 'twosheet_back'
    front = subset == 'twosheet_front'
    zb = z0[back] + theta[back]
    zf = z0[front] + theta[front]
    return dict(back_flip=wilson_dict(np.abs(zb - FRONT_Z) < np.abs(zb)),
                front_flip=wilson_dict(np.abs(zf) < np.abs(zf - FRONT_Z)),
                back_moved=int((theta[back] != 0).sum()), front_moved=int((theta[front] != 0).sum()))


EQ_DIMS = (('dmg_rate_N', 0.002, 'lower'),
           ('dmg_mag_mean_moved', 0.05, 'lower'),
           ('dmg_mag_p95_moved', 0.2, 'lower'),
           ('e_after_mean', 0.02, 'lower'),
           ('big_repair_cov', 0.02, 'higher'))


def equivalence(blkA, blkB):
    """CPR2 (A) minus TEST_ARGMIN (B) for the five pre-registered dimensions."""
    out = {}
    for key, margin, better in EQ_DIMS:
        a = blkA[key]
        b = blkB[key]
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            out[key] = dict(delta=float('nan'), margin=margin, within=None, favors='n/a')
            continue
        d = float(a - b)
        if d == 0.0:
            fav = 'tie'
        elif better == 'lower':
            fav = 'CPR2' if d < 0 else 'TEST_ARGMIN'
        else:
            fav = 'CPR2' if d > 0 else 'TEST_ARGMIN'
        out[key] = dict(delta=d, margin=margin, within=bool(abs(d) <= margin), favors=fav)
    return out


def dist_stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return dict(n=0, frac_pos=float('nan'), mean=float('nan'), median=float('nan'),
                    p90=float('nan'), max=float('nan'))
    return dict(n=int(v.size), frac_pos=float(np.mean(v > 0)), mean=float(v.mean()),
                median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))


def evaluate(pop, s_true_I, stratI, dec, arms, guard, cal, alpha_tgt, Kp):
    tstar = pop['tstar']
    subset = pop['subset']
    mm = pop['mm']
    N = pop['N']
    x, y, z0 = pop['x'], pop['y'], pop['z0']
    in_region = pop['in_region']
    covered_tgt = s_true_I <= dec['qtgt_i']
    covered_test = s_true_I <= dec['qtest_i']
    big_mask = subset == 'big'
    twosheet = (subset == 'twosheet_back') | (subset == 'twosheet_front')
    misfit = subset == 'misfit'
    R = {}

    R['calibration'] = {name: dict(n_stratum=cal[name]['n_stratum'], n_used=cal[name]['n_used'],
                                   fallback=cal[name]['fallback'], q_test=cal[name]['q_test'],
                                   q_tgt=cal[name]['q_tgt'], q_abs=cal[name]['q_abs'])
                        for name in STRATA}
    R['strata_counts_I'] = {name: int((stratI == k).sum()) for k, name in enumerate(STRATA)}
    R['strata_counts_Kp'] = {name: int((Kp['strat'] == k).sum()) for k, name in enumerate(STRATA)}

    cov = {}
    cov['true_tgt_by_stratum'] = {name: wilson_dict(covered_tgt[stratI == k]) for k, name in enumerate(STRATA)}
    cov['true_test_by_stratum'] = {name: wilson_dict(covered_test[stratI == k]) for k, name in enumerate(STRATA)}
    cov['true_tgt_by_subset'] = {sub: wilson_dict(covered_tgt[subset == sub]) for sub in SUBSETS}
    cov['true_test_by_subset'] = {sub: wilson_dict(covered_test[subset == sub]) for sub in SUBSETS}
    cov['true_tgt_overall'] = wilson_dict(covered_tgt)
    cov['true_test_overall'] = wilson_dict(covered_test)
    cov['proxy_tgt_Kp_by_stratum'] = {name: wilson_dict(Kp['s_true'][Kp['strat'] == k] <= cal[name]['q_tgt'])
                                      for k, name in enumerate(STRATA)}
    cov['proxy_test_Kp_by_stratum'] = {name: wilson_dict(Kp['s_true'][Kp['strat'] == k] <= cal[name]['q_test'])
                                       for k, name in enumerate(STRATA)}
    R['coverage'] = cov

    out = dec['out']
    res = dec['res']
    rej = dec['rej']
    D = {}
    D['frac_keep'] = float(np.mean(out == 'KEEP'))
    D['frac_move'] = float(np.mean(out == 'MOVE'))
    D['frac_none'] = float(np.mean(res == 'NONE'))
    D['frac_acquire'] = float(np.mean(res == 'ACQUIRE'))
    D['frac_respecify'] = float(np.mean(res == 'RESPECIFY'))
    D['n_rej'] = int(rej.sum())
    D['frac_rej'] = float(rej.mean())
    D['fdp_eps'] = float(np.mean(np.abs(tstar[rej]) <= EPS_TOL)) if rej.any() else float('nan')
    D['fdp_eps_label'] = 'empirical, point-null p-values'
    D['n_anomaly_zero_in_L'] = int(dec['anomaly'].sum())
    D['n_abs_bad'] = int((res == 'RESPECIFY').sum())
    D['keep_rate_by_stratum'] = {name: wilson_dict(out[stratI == k] == 'KEEP') for k, name in enumerate(STRATA)}
    D['move_rate_by_stratum'] = {name: wilson_dict(out[stratI == k] == 'MOVE') for k, name in enumerate(STRATA)}
    D['rej_rate_by_stratum'] = {name: wilson_dict(rej[stratI == k]) for k, name in enumerate(STRATA)}
    D['move_rate_by_subset'] = {sub: wilson_dict(out[subset == sub] == 'MOVE') for sub in SUBSETS}
    D['rej_rate_by_subset'] = {sub: wilson_dict(rej[subset == sub]) for sub in SUBSETS}
    D['acquire_sens_by_stratum'] = {name: wilson_dict(res[(stratI == k) & mm] == 'ACQUIRE') for k, name in enumerate(STRATA)}
    D['acquire_false_by_stratum'] = {name: wilson_dict(res[(stratI == k) & ~mm & ~twosheet] == 'ACQUIRE')
                                     for k, name in enumerate(STRATA)}
    D['acquire_sens_medium_sharp'] = wilson_dict(res[(stratI >= 1) & mm] == 'ACQUIRE')
    D['acquire_sens_all'] = wilson_dict(res[mm] == 'ACQUIRE')
    D['acquire_false_all'] = wilson_dict(res[~mm & ~twosheet] == 'ACQUIRE')
    D['respecify_sens_misfit'] = wilson_dict(res[misfit] == 'RESPECIFY')
    D['respecify_false_nonmisfit'] = wilson_dict(res[~misfit] == 'RESPECIFY')
    D['respecify_false_nonmisfit_by_stratum'] = {name: wilson_dict(res[(stratI == k) & ~misfit] == 'RESPECIFY')
                                                 for k, name in enumerate(STRATA)}
    D['ncomp_hist'] = {str(k): int((dec['n_comp'] == k).sum()) for k in range(1, 6)}
    D['ncomp_ge6'] = int((dec['n_comp'] >= 6).sum())
    R['decisions'] = D

    pr = dec['prio']
    R['priority'] = dict(mm=dist_stats(pr[mm]), not_mm=dist_stats(pr[~mm]),
                         by_subset={sub: dist_stats(pr[subset == sub]) for sub in SUBSETS},
                         mm_acquire=dist_stats(pr[mm & (res == 'ACQUIRE')]),
                         not_mm_acquire=dist_stats(pr[~mm & (res == 'ACQUIRE')]))

    allmask = np.ones(N, dtype=bool)
    T_pts = np.column_stack([x, y, z0 + tstar])
    A = {}
    for name in ARM_ORDER:
        th = arms[name]
        blk = {}
        blk['overall'] = arm_block(th, tstar, allmask, big_mask)
        blk['by_stratum'] = {sn: arm_block(th, tstar, stratI == k, big_mask) for k, sn in enumerate(STRATA)}
        blk['by_subset'] = {sub: arm_block(th, tstar, subset == sub, big_mask) for sub in SUBSETS}
        moved = th != 0
        blk['covsplit'] = dict(covered=dmg_count(th, tstar, moved & covered_tgt),
                               uncovered=dmg_count(th, tstar, moved & ~covered_tgt))
        blk['sheet'] = sheet_flips(th, subset, z0)
        O_pts = np.column_stack([x, y, z0 + th])
        blk['set'] = dict(overall=set_metrics(O_pts, T_pts),
                          region=set_metrics(O_pts[in_region], T_pts[in_region]))
        if name in guard:
            gd = guard[name]
            base_th = arms[gd['base']]
            moved_b = base_th != 0
            col = gd['collapse']
            nn = gd['nn']
            outside = ~in_region
            n_mo = int((moved_b & outside).sum())
            n_fo = int((col & outside).sum())
            blk['guard'] = dict(n_moved_before=int(moved_b.sum()), n_flagged=int(col.sum()),
                                n_moved_outside_region=n_mo, n_flagged_outside_region=n_fo,
                                fp_rate_outside_region=(n_fo / n_mo) if n_mo else float('nan'),
                                n_moved_in_region=int((moved_b & in_region).sum()),
                                n_flagged_in_region=int((col & in_region).sum()),
                                nn_min_moved=float(np.nanmin(nn)) if nn.size else float('nan'),
                                nn_p05_moved=float(np.nanpercentile(nn, 5)) if nn.size else float('nan'),
                                nn_median_moved=float(np.nanmedian(nn)) if nn.size else float('nan'))
        A[name] = blk
    R['arms'] = A

    th = arms['CPR2']
    sel = (th != 0) & covered_tgt
    if sel.sum() >= 2:
        e_a = np.abs(th[sel] - tstar[sel])
        hw = dec['w'][sel] / 2.0
        ratio = e_a / hw
        rho = spearmanr(e_a, hw).correlation if sel.sum() >= 3 else float('nan')
        R['residual_cpr2'] = dict(n=int(sel.sum()), median_ratio=float(np.median(ratio)),
                                  q25_ratio=float(np.percentile(ratio, 25)), q75_ratio=float(np.percentile(ratio, 75)),
                                  iqr_ratio=float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
                                  spearman=float(rho), mean_e_after=float(e_a.mean()), mean_half_w=float(hw.mean()),
                                  median_w=float(np.median(dec['w'][sel])))
    else:
        R['residual_cpr2'] = dict(n=int(sel.sum()), median_ratio=float('nan'), q25_ratio=float('nan'),
                                  q75_ratio=float('nan'), iqr_ratio=float('nan'), spearman=float('nan'),
                                  mean_e_after=float('nan'), mean_half_w=float('nan'), median_w=float('nan'))

    R['equivalence'] = dict(overall=equivalence(A['CPR2']['overall'], A['TEST_ARGMIN']['overall']),
                            by_stratum={sn: equivalence(A['CPR2']['by_stratum'][sn], A['TEST_ARGMIN']['by_stratum'][sn])
                                        for sn in STRATA})
    R['moved_set_identical_CPR2_TEST_ARGMIN'] = bool(np.array_equal(arms['CPR2'] != 0, arms['TEST_ARGMIN'] != 0))
    return R


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def write_decisions_csv(path, pop, stratI, dec, arms, guard, s_true_I, Ahat, targmin):
    cols = ['i', 'ix', 'iy', 'x', 'y', 'z0', 'subset', 'mm', 'gen_stratum', 'A', 'Ahat', 'obs_stratum',
            'tstar', 'm', 't_argmin', 's0', 'p', 'rej', 'n_comp', 'w', 'covered_tgt', 'out', 'res',
            'prio', 'anomaly']
    cols += ['theta_' + a for a in ARM_ORDER]
    cols += ['collapse_CPR2_G', 'collapse_TEST_ARGMIN_G']
    N = pop['N']
    covered = s_true_I <= dec['qtgt_i']
    with gzip.open(path, 'wt', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for i in range(N):
            row = [i, int(pop['ix'][i]), int(pop['iy'][i]), '%.3f' % pop['x'][i], '%.3f' % pop['y'][i],
                   '%.1f' % pop['z0'][i], pop['subset'][i], int(pop['mm'][i]), STRATA[int(pop['gen'][i])],
                   '%.5f' % pop['A'][i], '%.5f' % Ahat[i], STRATA[int(stratI[i])],
                   '%.5f' % pop['tstar'][i], '%.5f' % pop['m_'][i], '%.3f' % targmin[i], '%.5f' % dec['s0'][i],
                   '%.6f' % dec['p'][i], int(dec['rej'][i]), int(dec['n_comp'][i]), '%.3f' % dec['w'][i],
                   int(covered[i]), dec['out'][i], dec['res'][i], '%.5f' % dec['prio'][i], int(dec['anomaly'][i])]
            row += ['%.3f' % arms[a][i] for a in ARM_ORDER]
            row += [int(guard['CPR2_G']['collapse'][i]), int(guard['TEST_ARGMIN_G']['collapse'][i])]
            wr.writerow(row)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_population(seed, f_big, nx, ny, n_cal, res_dir, log):
    fb_code = int(round(f_big * 100))
    t0 = time.time()
    rng_I = np.random.default_rng([seed, fb_code, 0])
    pop = gen_population(rng_I, nx, ny, f_big)
    sI, mI, targmin, AhatI = scores(pop['c'])
    pop['m_'] = mI
    cut = np.quantile(AhatI, [1.0 / 3.0, 2.0 / 3.0])
    stratI = np.digitize(AhatI, cut)
    s_true_I = interp_rows(sI, pop['tstar'])  # evaluation channel only
    props = {sub: float(np.mean(pop['subset'] == sub)) for sub in SUBSETS}
    log('  population seed=%d f_big=%.2f: N=%d realized f_big=%.4f (%d cells, %d patches), misfit=%d, mm=%d, '
        'strata cutoffs=(%.4f, %.4f), gen time %.1fs' % (
            seed, f_big, pop['N'], pop['realized']['f_big_realized'], pop['realized']['n_big'],
            pop['realized']['n_patches'], pop['realized']['n_misfit'], pop['realized']['n_mm'],
            cut[0], cut[1], time.time() - t0))
    results = []
    for ci, calib in enumerate(CALIBS):
        rngK = np.random.default_rng([seed, fb_code, 10 + ci, 1])
        rngKp = np.random.default_rng([seed, fb_code, 10 + ci, 2])
        K = gen_calset(rngK, n_cal, calib, props)
        Kp = gen_calset(rngKp, n_cal, calib, props)
        sK, mK, _, AhK = scores(K['c'])
        stratK = np.digitize(AhK, cut)
        s_true_K = interp_rows(sK, K['tstar'])
        sKp, mKp, _, AhKp = scores(Kp['c'])
        stratKp = np.digitize(AhKp, cut)
        s_true_Kp = interp_rows(sKp, Kp['tstar'])
        Kp_info = dict(s_true=s_true_Kp, m=mKp, strat=stratKp)
        for alpha_tgt in ALPHA_TGTS:
            t1 = time.time()
            cal = calibrate(s_true_K, mK, stratK, alpha_tgt)
            dec = decide(sI, mI, stratI, cal)
            arms, guard = build_arms(targmin, dec['out'], dec['theta'], pop['x'], pop['y'], pop['z0'])
            R = evaluate(pop, s_true_I, stratI, dec, arms, guard, cal, alpha_tgt, Kp_info)
            R['scenario'] = dict(f_big=f_big, calib=calib, alpha_tgt=alpha_tgt, key=scenario_key(f_big, calib, alpha_tgt))
            R['seed'] = seed
            R['population'] = pop['realized']
            R['population']['strata_cutoffs'] = [float(cut[0]), float(cut[1])]
            R['params'] = dict(nx=nx, ny=ny, n_cal=n_cal, alpha_test=ALPHA_TEST, alpha_abs=ALPHA_ABS, bh_q=BH_Q,
                               eps_tol=EPS_TOL, taus=list(TAUS), d_guard=D_GUARD, voxel=VOXEL)
            R['wall_time_s'] = time.time() - t1
            key = R['scenario']['key']
            os.makedirs(os.path.join(res_dir, 'per_seed'), exist_ok=True)
            with open(os.path.join(res_dir, 'per_seed', '%s_seed%d.json' % (key, seed)), 'w') as fh:
                json.dump(to_jsonable(R), fh, indent=1)
            if seed == 1:
                os.makedirs(os.path.join(res_dir, 'decisions'), exist_ok=True)
                write_decisions_csv(os.path.join(res_dir, 'decisions', '%s_seed1.csv.gz' % key),
                                    pop, stratI, dec, arms, guard, s_true_I, AhatI, targmin)
            c2 = R['arms']['CPR2']['overall']
            log('    run %s seed=%d: rej=%d move=%.4f keep=%.4f | CPR2 dmg/N=%.5f dmg_cov=%d e_after=%.4f | '
                'anomalies=%d | fallback=%s | cov_tgt=%s | %.1fs' % (
                    key, seed, R['decisions']['n_rej'], R['decisions']['frac_move'], R['decisions']['frac_keep'],
                    c2['dmg_rate_N'], R['arms']['CPR2']['covsplit']['covered']['n_dmg'], c2['e_after_mean'],
                    R['decisions']['n_anomaly_zero_in_L'],
                    ','.join(sn for sn in STRATA if cal[sn]['fallback']) or '-',
                    ','.join('%.3f' % R['coverage']['true_tgt_by_stratum'][sn]['rate'] for sn in STRATA),
                    R['wall_time_s']))
            results.append(R)
    return results


# ----------------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------------
def aggregate(runs):
    first = runs[0]
    if isinstance(first, dict):
        return {k: aggregate([r[k] for r in runs]) for k in first}
    if isinstance(first, bool) or first is None or isinstance(first, str):
        vals = [r for r in runs]
        return vals[0] if all(v == vals[0] for v in vals) else vals
    if isinstance(first, (int, float)):
        vals = np.array([float(r) if r is not None else np.nan for r in runs], dtype=float)
        good = vals[~np.isnan(vals)]
        return dict(mean=float(good.mean()) if good.size else float('nan'),
                    sd=float(good.std(ddof=1)) if good.size > 1 else 0.0,
                    n=int(good.size), values=[float(v) for v in vals])
    if isinstance(first, list):
        return runs[0] if all(r == runs[0] for r in runs) else runs
    return runs


# ----------------------------------------------------------------------------
# Results document
# ----------------------------------------------------------------------------
def f4(v, nd=4):
    if v is None:
        return 'n/a'
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    try:
        if math.isnan(v):
            return 'nan'
        if math.isinf(v):
            return '+inf' if v > 0 else '-inf'
    except TypeError:
        return str(v)
    return ('%.' + str(nd) + 'f') % v


def pm(leaf, nd=4):
    if isinstance(leaf, dict) and 'mean' in leaf:
        return '%s ± %s' % (f4(leaf['mean'], nd), f4(leaf['sd'], nd))
    return f4(leaf, nd)


def md_table(header, rows):
    lines = ['| ' + ' | '.join(str(h) for h in header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(lines) + '\n'


def run_index(all_runs):
    idx = {}
    for R in all_runs:
        sc = R['scenario']
        idx[(sc['f_big'], sc['calib'], sc['alpha_tgt'], R['seed'])] = R
    return idx


def status_of(n_pass, n_total):
    if n_total == 0:
        return 'N/A'
    if n_pass == n_total:
        return 'PASS'
    if n_pass == 0:
        return 'FAIL'
    return 'PARTIAL'


def check_predictions(all_runs, aggs):
    idx = run_index(all_runs)
    lines = []

    # ---------------- P1 ----------------
    lines.append('### P1 (exch)\n')
    lines.append('Criteria per exch run: (i) for every obs_stratum, Wilson-95% upper bound of true coverage of '
                 'L(α_tgt) on I ≥ 1−α_tgt; (ii) CPR2 #Dmg/N ≤ α_tgt; (iii) CPR2 damage count in the '
                 'coverage-satisfied moved split == 0.\n')
    rows = []
    n_pass = n_tot = 0
    sub_pass = {'i': 0, 'ii': 0, 'iii': 0}
    for R in all_runs:
        sc = R['scenario']
        if sc['calib'] != 'exch':
            continue
        n_tot += 1
        nominal = 1 - sc['alpha_tgt']
        cov = R['coverage']['true_tgt_by_stratum']
        ok_i = all(cov[sn]['hi'] >= nominal - 1e-12 for sn in STRATA)
        dmgN = R['arms']['CPR2']['overall']['dmg_rate_N']
        ok_ii = dmgN <= sc['alpha_tgt']
        ndc = R['arms']['CPR2']['covsplit']['covered']['n_dmg']
        ok_iii = ndc == 0
        sub_pass['i'] += ok_i
        sub_pass['ii'] += ok_ii
        sub_pass['iii'] += ok_iii
        ok = ok_i and ok_ii and ok_iii
        n_pass += ok
        rows.append([sc['key'], R['seed']] +
                    ['%.4f [%.4f, %.4f]' % (cov[sn]['rate'], cov[sn]['lo'], cov[sn]['hi']) for sn in STRATA] +
                    ['%.4f' % nominal, 'yes' if ok_i else 'no', '%.5f' % dmgN, 'yes' if ok_ii else 'no',
                     str(ndc), 'yes' if ok_iii else 'no', 'PASS' if ok else 'FAIL'])
    lines.append(md_table(['scenario', 'seed', 'cov flat [Wilson]', 'cov medium', 'cov sharp', '1−α_tgt', '(i)',
                           'CPR2 #Dmg/N', '(ii)', 'CPR2 #Dmg covered', '(iii)', 'run'], rows))
    lines.append('**P1: %s** — %d/%d exch runs pass all three criteria (i: %d/%d, ii: %d/%d, iii: %d/%d).\n' % (
        status_of(n_pass, n_tot), n_pass, n_tot, sub_pass['i'], n_tot, sub_pass['ii'], n_tot, sub_pass['iii'], n_tot))

    # ---------------- P2 ----------------
    lines.append('### P2 (sfm)\n')
    lines.append('Criteria per sfm run: (i) K\' proxy coverage of L(α_tgt) passes (Wilson upper ≥ 1−α_tgt) in all '
                 'strata with n_K\'σ > 0; (ii) true coverage on I falls below 1−α_tgt−tol (Wilson upper < 1−α_tgt) '
                 'in flat and/or medium; (iii) in every stratum identified in (ii), CPR2 #Dmg/N_σ (sfm) > CPR2 '
                 '#Dmg/N_σ (exch counterpart: same f_big, α_tgt, seed).\n')
    rows = []
    n_pass = n_tot = 0
    sub_pass = {'i': 0, 'ii': 0, 'iii': 0}
    for R in all_runs:
        sc = R['scenario']
        if sc['calib'] != 'sfm':
            continue
        n_tot += 1
        nominal = 1 - sc['alpha_tgt']
        prox = R['coverage']['proxy_tgt_Kp_by_stratum']
        ok_i = all((prox[sn]['n'] == 0) or (prox[sn]['hi'] >= nominal - 1e-12) for sn in STRATA)
        cov = R['coverage']['true_tgt_by_stratum']
        low = [sn for sn in ('flat', 'medium') if cov[sn]['hi'] < nominal]
        ok_ii = len(low) > 0
        Rex = idx[(sc['f_big'], 'exch', sc['alpha_tgt'], R['seed'])]
        cmp_txt = []
        ok_iii = ok_ii
        for sn in low:
            a = R['arms']['CPR2']['by_stratum'][sn]['dmg_rate_N']
            b = Rex['arms']['CPR2']['by_stratum'][sn]['dmg_rate_N']
            cmp_txt.append('%s: %.5f vs %.5f' % (sn, a, b))
            if not (a > b):
                ok_iii = False
        sub_pass['i'] += ok_i
        sub_pass['ii'] += ok_ii
        sub_pass['iii'] += ok_iii
        ok = ok_i and ok_ii and ok_iii
        n_pass += ok
        rows.append([sc['key'], R['seed']] +
                    ['%.4f (n=%d)' % (prox[sn]['rate'], prox[sn]['n']) if prox[sn]['n'] else 'n=0' for sn in STRATA] +
                    ['yes' if ok_i else 'no'] +
                    ['%.4f [%.4f, %.4f]' % (cov[sn]['rate'], cov[sn]['lo'], cov[sn]['hi']) for sn in ('flat', 'medium')] +
                    [','.join(low) or '-', 'yes' if ok_ii else 'no', '; '.join(cmp_txt) or '-',
                     'yes' if ok_iii else 'no', 'PASS' if ok else 'FAIL'])
    lines.append(md_table(['scenario', 'seed', 'K\' proxy flat', 'K\' proxy medium', 'K\' proxy sharp', '(i)',
                           'true cov flat', 'true cov medium', 'strata below', '(ii)',
                           'CPR2 #Dmg/N_σ sfm vs exch', '(iii)', 'run'], rows))
    lines.append('**P2: %s** — %d/%d sfm runs pass all three criteria (i: %d/%d, ii: %d/%d, iii: %d/%d).\n' % (
        status_of(n_pass, n_tot), n_pass, n_tot, sub_pass['i'], n_tot, sub_pass['ii'], n_tot, sub_pass['iii'], n_tot))

    # ---------------- P3 ----------------
    lines.append('### P3 (flat stratum, exch)\n')
    lines.append('Criteria per exch run, flat stratum: (i) CPR2 KEEP rate ≥ 0.95; (ii) CPR2 #Dmg/N_flat ≤ α_tgt; '
                 '(iii) ARGMIN #Dmg/N_flat ≥ 0.5; (iv) every GATED(τ) has #Dmg/N_flat > CPR2.\n')
    rows = []
    n_pass = n_tot = 0
    sub_pass = {'i': 0, 'ii': 0, 'iii': 0, 'iv': 0}
    for R in all_runs:
        sc = R['scenario']
        if sc['calib'] != 'exch':
            continue
        n_tot += 1
        keep = R['decisions']['keep_rate_by_stratum']['flat']['rate']
        c2 = R['arms']['CPR2']['by_stratum']['flat']['dmg_rate_N']
        am = R['arms']['ARGMIN']['by_stratum']['flat']['dmg_rate_N']
        am_m = R['arms']['ARGMIN']['by_stratum']['flat']['dmg_rate_moved']
        gs = [R['arms']['GATED%d' % int(t)]['by_stratum']['flat']['dmg_rate_N'] for t in TAUS]
        ok_i = keep >= 0.95
        ok_ii = c2 <= sc['alpha_tgt']
        ok_iii = am >= 0.5
        ok_iv = all(gv > c2 for gv in gs)
        sub_pass['i'] += ok_i
        sub_pass['ii'] += ok_ii
        sub_pass['iii'] += ok_iii
        sub_pass['iv'] += ok_iv
        ok = ok_i and ok_ii and ok_iii and ok_iv
        n_pass += ok
        rows.append([sc['key'], R['seed'], '%.4f' % keep, 'yes' if ok_i else 'no', '%.5f' % c2, 'yes' if ok_ii else 'no',
                     '%.4f (over moved %.4f)' % (am, am_m), 'yes' if ok_iii else 'no'] +
                    ['%.5f' % gv for gv in gs] + ['yes' if ok_iv else 'no', 'PASS' if ok else 'FAIL'])
    lines.append(md_table(['scenario', 'seed', 'CPR2 KEEP flat', '(i)', 'CPR2 #Dmg/N flat', '(ii)', 'ARGMIN #Dmg/N flat',
                           '(iii)', 'GATED1', 'GATED2', 'GATED3', '(iv)', 'run'], rows))
    lines.append('**P3: %s** — %d/%d exch runs pass all four criteria (i: %d/%d, ii: %d/%d, iii: %d/%d, iv: %d/%d).\n' % (
        status_of(n_pass, n_tot), n_pass, n_tot, sub_pass['i'], n_tot, sub_pass['ii'], n_tot, sub_pass['iii'], n_tot,
        sub_pass['iv'], n_tot))

    # ---------------- P4 ----------------
    lines.append('### P4 (projection vs argmin, same movable set)\n')
    lines.append('Five dimensions, Δ = CPR2 − TEST_ARGMIN on the seed-mean of each scenario, overall and per obs_stratum. '
                 '"within" = |Δ| ≤ margin. Margins: #Dmg/N 0.002; mean damage magnitude (over moved) 0.05; p95 damage '
                 '(over moved) 0.2; mean e_after 0.02; repair coverage (big) 0.02. Lower is better for the first four, '
                 'higher for repair coverage.\n')
    all_within = True
    n_cells = 0
    n_within_cells = 0
    fav_count = {'CPR2': 0, 'TEST_ARGMIN': 0, 'tie': 0, 'n/a': 0}
    fav_by_dim = {k: {'CPR2': 0, 'TEST_ARGMIN': 0, 'tie': 0, 'n/a': 0, 'within': 0, 'outside': 0} for k, _, _ in EQ_DIMS}
    rows = []
    for key in sorted(aggs):
        ag = aggs[key]
        for scope in ['overall'] + list(STRATA):
            eq = ag['equivalence']['overall'] if scope == 'overall' else ag['equivalence']['by_stratum'][scope]
            row = [key, scope]
            cell_within = True
            for k, margin, better in EQ_DIMS:
                d = eq[k]['delta']['mean']
                if math.isnan(d):
                    row.append('n/a')
                    fav_by_dim[k]['n/a'] += 1
                    continue
                w_in = abs(d) <= margin
                fav = 'tie' if d == 0 else (('CPR2' if d < 0 else 'TEST_ARGMIN') if better == 'lower'
                                            else ('CPR2' if d > 0 else 'TEST_ARGMIN'))
                fav_by_dim[k][fav] += 1
                fav_by_dim[k]['within' if w_in else 'outside'] += 1
                if not w_in:
                    cell_within = False
                    all_within = False
                row.append('%+.5f %s (%s)' % (d, 'in' if w_in else 'OUT', fav))
            n_cells += 1
            n_within_cells += cell_within
            row.append('yes' if cell_within else 'no')
            rows.append(row)
    lines.append(md_table(['scenario', 'scope', 'Δ #Dmg/N', 'Δ mean dmg mag', 'Δ p95 dmg', 'Δ mean e_after',
                           'Δ repair cov (big)', 'all five within'], rows))
    lines.append('Per-dimension tally over %d (scenario × scope) cells: ' % n_cells +
                 '; '.join('%s: favors CPR2 %d, favors TEST_ARGMIN %d, tie %d, n/a %d, within %d, outside %d' % (
                     k, v['CPR2'], v['TEST_ARGMIN'], v['tie'], v['n/a'], v['within'], v['outside'])
                           for k, v in fav_by_dim.items()) + '.\n')
    # expectations
    exp_rows = []
    e1 = e2 = e1n = 0
    for key in sorted(aggs):
        ag = aggs[key]
        a = ag['arms']['TEST_ARGMIN']['by_stratum']['sharp']['big_e_after_mean']['mean']
        b = ag['arms']['CPR2']['by_stratum']['sharp']['big_e_after_mean']['mean']
        dc = ag['arms']['CPR2']['covsplit']['covered']['n_dmg']
        dt = ag['arms']['TEST_ARGMIN']['covsplit']['covered']['n_dmg']
        sc = ag['arms']['CPR2']['overall']['dmg_sum']
        st = ag['arms']['TEST_ARGMIN']['overall']['dmg_sum']
        ok1 = (not math.isnan(a)) and (not math.isnan(b)) and a < b
        ok2 = max(dc['values']) == 0 and min(dt['values']) > 0
        e1 += ok1
        e1n += 1
        e2 += ok2
        exp_rows.append([key, f4(a), f4(b), 'yes' if ok1 else 'no',
                         '/'.join(str(int(v)) for v in dc['values']), '/'.join(str(int(v)) for v in dt['values']),
                         'yes' if ok2 else 'no', pm(sc, 3), pm(st, 3),
                         'CPR2 lower' if sc['mean'] < st['mean'] else ('TEST_ARGMIN lower' if st['mean'] < sc['mean'] else 'tie')])
    lines.append('Expectations (not required to pass), seed-mean per scenario:\n')
    lines.append(md_table(['scenario', 'TEST_ARGMIN big e_after (sharp)', 'CPR2 big e_after (sharp)', 'TEST_ARGMIN lower?',
                           'CPR2 #Dmg covered (seeds)', 'TEST_ARGMIN #Dmg covered (seeds)', 'CPR2==0 & TA>0?',
                           'CPR2 Σd', 'TEST_ARGMIN Σd', 'lower Σd'], exp_rows))
    decision = ('all five margins met in all scenarios and strata → the pre-registered rule would allow dropping projection'
                if all_within else
                'not all margins met in all scenarios and strata → pre-registered rule: RETAIN projection')
    lines.append('**P4: REPORTED** — %d/%d (scenario × scope) cells have all five within margins; %s. Expectation "TEST_ARGMIN '
                 'lower big e_after in sharp": %d/%d scenarios; expectation "CPR2 covered damage == 0 and TEST_ARGMIN > 0 in '
                 'all seeds": %d/%d scenarios.\n' % (n_within_cells, n_cells, decision, e1, e1n, e2, e1n))

    # ---------------- P5 ----------------
    lines.append('### P5 (two-sheet)\n')
    lines.append('Criteria per run: without guard, for BOTH CPR2 and TEST_ARGMIN: (i) BACK flip rate ≥ 0.50 and (ii) region '
                 'completeness > IDENTITY region completeness; with guard, for BOTH CPR2_G and TEST_ARGMIN_G: (iii) BACK flip '
                 'rate < 0.10 and (iv) |region accuracy − IDENTITY| ≤ 0.1 and |region completeness − IDENTITY| ≤ 0.1; '
                 '(v) guard false-positive rate (outside two-sheet region) ≤ 0.01 for both guard arms.\n')
    rows = []
    n_pass = n_tot = 0
    sub_pass = {'i': 0, 'ii': 0, 'iii': 0, 'iv': 0, 'v': 0}
    for R in all_runs:
        sc = R['scenario']
        n_tot += 1
        idr = R['arms']['IDENTITY']['set']['region']
        fl = {a: R['arms'][a]['sheet']['back_flip']['rate'] for a in ('CPR2', 'TEST_ARGMIN', 'CPR2_G', 'TEST_ARGMIN_G')}
        cm = {a: R['arms'][a]['set']['region']['completeness'] for a in ('CPR2', 'TEST_ARGMIN', 'CPR2_G', 'TEST_ARGMIN_G')}
        ac = {a: R['arms'][a]['set']['region']['accuracy'] for a in ('CPR2_G', 'TEST_ARGMIN_G')}
        fp = {a: R['arms'][a]['guard']['fp_rate_outside_region'] for a in ('CPR2_G', 'TEST_ARGMIN_G')}
        ok_i = fl['CPR2'] >= 0.5 and fl['TEST_ARGMIN'] >= 0.5
        ok_ii = cm['CPR2'] > idr['completeness'] and cm['TEST_ARGMIN'] > idr['completeness']
        ok_iii = fl['CPR2_G'] < 0.10 and fl['TEST_ARGMIN_G'] < 0.10
        ok_iv = all(abs(ac[a] - idr['accuracy']) <= 0.1 and abs(cm[a] - idr['completeness']) <= 0.1 for a in ac)
        ok_v = all((math.isnan(fp[a]) or fp[a] <= 0.01) for a in fp)
        for kk, vv in zip(('i', 'ii', 'iii', 'iv', 'v'), (ok_i, ok_ii, ok_iii, ok_iv, ok_v)):
            sub_pass[kk] += vv
        ok = ok_i and ok_ii and ok_iii and ok_iv and ok_v
        n_pass += ok
        rows.append([sc['key'], R['seed'], '%.4f/%.4f' % (fl['CPR2'], fl['TEST_ARGMIN']), 'yes' if ok_i else 'no',
                     '%.4f | %.4f/%.4f' % (idr['completeness'], cm['CPR2'], cm['TEST_ARGMIN']), 'yes' if ok_ii else 'no',
                     '%.4f/%.4f' % (fl['CPR2_G'], fl['TEST_ARGMIN_G']), 'yes' if ok_iii else 'no',
                     'acc %.4f | %.4f/%.4f; comp %.4f | %.4f/%.4f' % (idr['accuracy'], ac['CPR2_G'], ac['TEST_ARGMIN_G'],
                                                                      idr['completeness'], cm['CPR2_G'], cm['TEST_ARGMIN_G']),
                     'yes' if ok_iv else 'no', '%s/%s' % (f4(fp['CPR2_G']), f4(fp['TEST_ARGMIN_G'])), 'yes' if ok_v else 'no',
                     'PASS' if ok else 'FAIL'])
    lines.append(md_table(['scenario', 'seed', 'BACK flip CPR2/TEST_ARGMIN', '(i)', 'region compl. IDENTITY | CPR2/TA', '(ii)',
                           'BACK flip CPR2_G/TA_G', '(iii)', 'region acc/compl IDENTITY | guard arms', '(iv)',
                           'guard FP CPR2_G/TA_G', '(v)', 'run'], rows))
    lines.append('**P5: %s** — %d/%d runs pass all five criteria (i: %d/%d, ii: %d/%d, iii: %d/%d, iv: %d/%d, v: %d/%d).\n' % (
        status_of(n_pass, n_tot), n_pass, n_tot, sub_pass['i'], n_tot, sub_pass['ii'], n_tot, sub_pass['iii'], n_tot,
        sub_pass['iv'], n_tot, sub_pass['v'], n_tot))

    # ---------------- P6 ----------------
    lines.append('### P6 (residual ≈ w/2)\n')
    lines.append('Criterion per run: median of e_after/(w_i/2) over CPR2 coverage-satisfied moves lies in (0.5, 1.5).\n')
    rows = []
    n_pass = n_tot = 0
    for R in all_runs:
        rr = R['residual_cpr2']
        if rr['n'] < 2:
            rows.append([R['scenario']['key'], R['seed'], rr['n'], 'n/a', 'n/a', 'n/a', 'N/A (n<2)'])
            continue
        n_tot += 1
        ok = 0.5 < rr['median_ratio'] < 1.5
        n_pass += ok
        rows.append([R['scenario']['key'], R['seed'], rr['n'], f4(rr['median_ratio']),
                     '[%s, %s]' % (f4(rr['q25_ratio']), f4(rr['q75_ratio'])), f4(rr['spearman']), 'PASS' if ok else 'FAIL'])
    lines.append(md_table(['scenario', 'seed', 'n (moved & covered)', 'median ratio', 'IQR [q25, q75]', 'Spearman(e_after, w/2)',
                           'run'], rows))
    lines.append('**P6: %s** — %d/%d runs with n ≥ 2 have the median ratio in (0.5, 1.5).\n' % (status_of(n_pass, n_tot), n_pass, n_tot))

    # ---------------- P7 ----------------
    lines.append('### P7 (detection)\n')
    lines.append('Criteria per run: (i) ACQUIRE sensitivity on mm components in medium+sharp ≥ 0.6; (ii) RESPECIFY sensitivity on '
                 'misfit ≥ 0.8; (iii) RESPECIFY false rate on non-misfit ≤ α_abs + tol (Wilson lower bound ≤ 0.05).\n')
    rows = []
    n_pass = n_tot = 0
    sub_pass = {'i': 0, 'ii': 0, 'iii': 0}
    for R in all_runs:
        n_tot += 1
        D = R['decisions']
        a = D['acquire_sens_medium_sharp']
        b = D['respecify_sens_misfit']
        c = D['respecify_false_nonmisfit']
        ok_i = a['rate'] >= 0.6
        ok_ii = b['rate'] >= 0.8
        ok_iii = c['lo'] <= ALPHA_ABS
        sub_pass['i'] += ok_i
        sub_pass['ii'] += ok_ii
        sub_pass['iii'] += ok_iii
        ok = ok_i and ok_ii and ok_iii
        n_pass += ok
        rows.append([R['scenario']['key'], R['seed'], '%.4f (n=%d)' % (a['rate'], a['n']), 'yes' if ok_i else 'no',
                     '%.4f (n=%d)' % (b['rate'], b['n']), 'yes' if ok_ii else 'no',
                     '%.4f [%.4f, %.4f] (n=%d)' % (c['rate'], c['lo'], c['hi'], c['n']), 'yes' if ok_iii else 'no',
                     'PASS' if ok else 'FAIL'])
    lines.append(md_table(['scenario', 'seed', 'ACQUIRE sens mm (med+sharp)', '(i)', 'RESPECIFY sens misfit', '(ii)',
                           'RESPECIFY false non-misfit [Wilson]', '(iii)', 'run'], rows))
    lines.append('**P7: %s** — %d/%d runs pass all three criteria (i: %d/%d, ii: %d/%d, iii: %d/%d).\n' % (
        status_of(n_pass, n_tot), n_pass, n_tot, sub_pass['i'], n_tot, sub_pass['ii'], n_tot, sub_pass['iii'], n_tot))
    return '\n'.join(lines)


def write_results_md(path, all_runs, aggs, seeds, meta):
    L = []
    L.append('# E0 RESULTS_RAW\n')
    L.append('Generated by `e0_mechanism_test.py` from the per-seed JSON files in `%s`. Tables are seed-mean ± sd '
             '(sd with ddof=1) over seeds %s unless a table lists seeds explicitly. All lengths in mm. '
             'No interpretation beyond the pre-registered PASS/FAIL/PARTIAL rules.\n' % (meta['res_dir'], seeds))
    L.append('Run parameters: Nx=%d, Ny=%d, N=%d, |K|=|K\'|=%d, α_test=%.2f, α_abs=%.2f, BH q=%.2f, ε_tol=%.1f, '
             'd_g=%.1f, voxel=%.1f. Scenarios: f_big ∈ %s × calibration ∈ %s × α_tgt ∈ %s; seeds %s → %d runs. '
             'Arms: %s.\n' % (meta['nx'], meta['ny'], meta['nx'] * meta['ny'], meta['n_cal'], ALPHA_TEST, ALPHA_ABS, BH_Q,
                             EPS_TOL, D_GUARD, VOXEL, list(F_BIGS), list(CALIBS), list(ALPHA_TGTS), seeds, len(all_runs),
                             ', '.join(ARM_ORDER)))
    L.append('Definitions used (fixed before running): Wilson tolerance = 95% Wilson interval of the observed proportion; '
             '"≥ nominal − tol" ⇔ Wilson upper ≥ nominal; "< nominal − tol" ⇔ Wilson upper < nominal; '
             '"≤ α_abs + tol" ⇔ Wilson lower ≤ α_abs. Level-set component width w = (#grid points)·0.05. '
             'Δ in the equivalence tables = CPR2 − TEST_ARGMIN.\n')
    keys = sorted(aggs)

    # 0. sanity counters (straight from the saved runs / seed-1 decisions files)
    L.append('## 0. Sanity counters (all runs)\n')
    n_anom = sum(int(R['decisions']['n_anomaly_zero_in_L']) for R in all_runs)
    L.append('Anomalies "0 ∈ L_i after rejection with n_comp=1" (θ̂ set to 0, counted): total over %d runs = %d; per-run max = %d.\n' % (
        len(all_runs), n_anom, max(int(R['decisions']['n_anomaly_zero_in_L']) for R in all_runs)))
    for a in ('CPR2_G', 'TEST_ARGMIN_G'):
        gs = [R['arms'][a]['guard'] for R in all_runs]
        nn_mins = [g['nn_min_moved'] for g in gs if not math.isnan(g['nn_min_moved'])]
        L.append('Guard %s: Σ#moved before guard = %d, Σ#flagged = %d; min over runs of min NN distance among moved points = %s '
                 '(grid spacing h = %.1f, d_g = %.1f).\n' % (
                     a, sum(g['n_moved_before'] for g in gs), sum(g['n_flagged'] for g in gs),
                     ('%.4f' % min(nn_mins)) if nn_mins else 'n/a (no moved points)', H, D_GUARD))
    rows = []
    for key in keys:
        dpath = os.path.join(meta['res_dir'], 'decisions', '%s_seed1.csv.gz' % key)
        if not os.path.exists(dpath):
            continue
        pv = []
        nrej = 0
        with gzip.open(dpath, 'rt') as fh:
            for r in csv.DictReader(fh):
                pv.append(float(r['p']))
                nrej += int(r['rej'])
        pv = np.asarray(pv)
        pmin = float(pv.min())
        k_floor = int((pv <= pmin * (1 + 1e-9)).sum())
        rows.append([key, len(pv), '%.5f' % pmin, k_floor, '%.5f' % (BH_Q * k_floor / len(pv)),
                     '%.5f' % (BH_Q * math.ceil(pmin * len(pv) / BH_Q) / len(pv)), math.ceil(pmin * len(pv) / BH_Q), nrej])
    L.append('Conformal p-value floor (seed 1 of each scenario, from decisions.csv): min p = 1/(n_σ+1) of the largest-n stratum; '
             '"#at floor" = components with p equal to the min; BH rejects the floor components only if min p ≤ q·k/N for some '
             'k ≥ #at floor, i.e. if #at floor ≥ ⌈min p·N/q⌉.\n')
    L.append(md_table(['scenario', 'N', 'min p', '#at floor', 'q·(#at floor)/N', 'q·⌈min p·N/q⌉/N', '⌈min p·N/q⌉ (needed at floor)',
                       '#Rej (seed 1)'], rows))

    # 1. population
    L.append('## 1. Realized population (I) per (f_big, seed)\n')
    rows = []
    seen = set()
    for R in all_runs:
        k = (R['scenario']['f_big'], R['seed'])
        if k in seen:
            continue
        seen.add(k)
        P = R['population']
        rows.append(['%.2f' % k[0], k[1], P['N'], P['n_big_target'], P['n_big'], '%.5f' % P['f_big_realized'], P['n_patches'],
                     P['n_patches_with_overlap'], P['n_twosheet_back'], P['n_twosheet_front'], P['n_misfit'], P['n_base'],
                     P['n_mm'], '%.4f' % P['mm_frac_of_base_big'],
                     '(%.4f, %.4f)' % tuple(P['strata_cutoffs']),
                     '/'.join(str(P['gen_stratum_counts'][s]) for s in STRATA),
                     '/'.join(str(R['strata_counts_I'][s]) for s in STRATA)])
    L.append(md_table(['f_big', 'seed', 'N', 'big target', 'big realized', 'f_big realized', 'patches', 'patches w/ overlap',
                       'BACK', 'FRONT', 'misfit', 'base', 'mm', 'mm frac (base+big)', 'Â tertile cutoffs',
                       'gen strata flat/med/sharp', 'obs strata flat/med/sharp'], rows))

    # 2. calibration
    L.append('## 2. Calibration quantiles per scenario and obs_stratum (seed-mean ± sd; n_K,σ and fallback listed per seed)\n')
    rows = []
    for key in keys:
        ag = aggs[key]
        for sn in STRATA:
            c = ag['calibration'][sn]
            fb = c['fallback']
            fb_txt = str(fb) if not isinstance(fb, list) else '/'.join(str(v) for v in fb)
            rows.append([key, sn, '/'.join(str(int(v)) for v in c['n_stratum']['values']),
                         '/'.join(str(int(v)) for v in c['n_used']['values']), fb_txt, pm(c['q_test'], 5), pm(c['q_tgt'], 5),
                         pm(c['q_abs'], 5), '/'.join(str(int(v)) for v in ag['strata_counts_Kp'][sn]['values'])])
    L.append(md_table(['scenario', 'stratum', 'n_K,σ (seeds)', 'n used', 'fallback pooled', 'q_test', 'q_tgt', 'q_abs',
                       'n_K\',σ (seeds)'], rows))

    # 3. coverage
    L.append('## 3. Coverage\n')
    L.append('### 3a. True coverage on I of L(α_tgt) and of {s ≤ q_test}, and proxy coverage on K\', per obs_stratum '
             '(seed-mean ± sd of the rate; per-seed Wilson 95% intervals in brackets)\n')
    rows = []
    for key in keys:
        ag = aggs[key]
        a_t = aggs[key]['scenario']['alpha_tgt']['mean']
        for sn in STRATA:
            ct = ag['coverage']['true_tgt_by_stratum'][sn]
            cq = ag['coverage']['true_test_by_stratum'][sn]
            pt = ag['coverage']['proxy_tgt_Kp_by_stratum'][sn]
            pq = ag['coverage']['proxy_test_Kp_by_stratum'][sn]
            wil = '; '.join('[%.3f,%.3f]' % (lo, hi) for lo, hi in zip(ct['lo']['values'], ct['hi']['values']))
            rows.append([key, sn, '%.2f' % (1 - a_t), pm(ct['rate']), wil, '/'.join(str(int(v)) for v in ct['n']['values']),
                         pm(cq['rate']), pm(pt['rate']), '/'.join(str(int(v)) for v in pt['n']['values']), pm(pq['rate'])])
    L.append(md_table(['scenario', 'stratum', '1−α_tgt', 'true cov L(α_tgt)', 'Wilson per seed', 'n_I,σ', 'true cov q_test (nominal 0.90)',
                       'proxy cov K\' L(α_tgt)', 'n_K\',σ', 'proxy cov K\' q_test'], rows))
    L.append('### 3b. True coverage on I per subset label\n')
    rows = []
    for key in keys:
        ag = aggs[key]
        for sub in SUBSETS:
            ct = ag['coverage']['true_tgt_by_subset'][sub]
            cq = ag['coverage']['true_test_by_subset'][sub]
            rows.append([key, sub, '/'.join(str(int(v)) for v in ct['n']['values']), pm(ct['rate']), pm(cq['rate'])])
        ct = ag['coverage']['true_tgt_overall']
        cq = ag['coverage']['true_test_overall']
        rows.append([key, 'ALL', '/'.join(str(int(v)) for v in ct['n']['values']), pm(ct['rate']), pm(cq['rate'])])
    L.append(md_table(['scenario', 'subset', 'n (seeds)', 'true cov L(α_tgt)', 'true cov q_test'], rows))

    # 4. arms overall
    L.append('## 4. Arm metrics, overall (per scenario; seed-mean ± sd)\n')
    for key in keys:
        ag = aggs[key]
        L.append('### 4.%s\n' % key)
        rows = []
        for a in ARM_ORDER:
            b = ag['arms'][a]['overall']
            rows.append([a, pm(b['n_moved'], 1), pm(b['n_dmg'], 1), pm(b['dmg_rate_N'], 5), pm(b['dmg_rate_moved']),
                         pm(b['dmg_mag_mean_moved']), pm(b['dmg_mag_p95_moved']), pm(b['dmg_mag_max']), pm(b['dmg_sum'], 2),
                         pm(b['e_after_mean']), pm(b['e_after_rmse']), pm(b['e_after_median']), pm(b['big_e_after_mean']),
                         pm(b['big_repair_cov'])])
        L.append(md_table(['arm', '#moved', '#Dmg', '#Dmg/N', '#Dmg/#M', 'dmg mean (moved)', 'dmg p95 (moved)', 'dmg max', 'Σd',
                           'mean e_after', 'RMSE e_after', 'median e_after', 'big: mean e_after', 'big: repair cov'], rows))
        eb = ag['arms']['IDENTITY']['overall']['e_before_mean']
        L.append('mean e_before (all) = %s; big: mean e_before = %s; n_big = %s\n' % (
            pm(eb), pm(ag['arms']['IDENTITY']['overall']['big_e_before_mean']), pm(ag['arms']['IDENTITY']['overall']['n_big'], 1)))

    # 5. arms by stratum
    L.append('## 5. Arm metrics by obs_stratum (per scenario; seed-mean ± sd)\n')
    for key in keys:
        ag = aggs[key]
        L.append('### 5.%s\n' % key)
        rows = []
        for a in ARM_ORDER:
            row = [a]
            for sn in STRATA:
                b = ag['arms'][a]['by_stratum'][sn]
                row += [pm(b['n_moved'], 1), pm(b['dmg_rate_N'], 5), pm(b['dmg_rate_moved']), pm(b['e_after_mean']),
                        pm(b['big_e_after_mean']), pm(b['big_repair_cov'])]
            rows.append(row)
        hdr = ['arm']
        for sn in STRATA:
            hdr += ['%s #moved' % sn, '%s #Dmg/N' % sn, '%s #Dmg/#M' % sn, '%s mean e_after' % sn, '%s big e_after' % sn,
                    '%s big repair' % sn]
        L.append(md_table(hdr, rows))
        L.append('n_I,σ (seeds): ' + '; '.join('%s %s' % (sn, '/'.join(str(int(v)) for v in ag['strata_counts_I'][sn]['values']))
                                                for sn in STRATA) + '; n_big per stratum (seed-mean): ' +
                 '; '.join('%s %s' % (sn, pm(ag['arms']['IDENTITY']['by_stratum'][sn]['n_big'], 1)) for sn in STRATA) + '\n')

    # 6. arms by subset
    L.append('## 6. Arm metrics by subset label (per scenario; seed-mean ± sd)\n')
    for key in keys:
        ag = aggs[key]
        L.append('### 6.%s\n' % key)
        rows = []
        for a in ARM_ORDER:
            row = [a]
            for sub in SUBSETS:
                b = ag['arms'][a]['by_subset'][sub]
                row += [pm(b['n_moved'], 1), pm(b['dmg_rate_N'], 4), pm(b['e_after_mean'])]
            rows.append(row)
        hdr = ['arm']
        for sub in SUBSETS:
            hdr += ['%s #moved' % sub, '%s #Dmg/N' % sub, '%s mean e_after' % sub]
        L.append(md_table(hdr, rows))

    # 7. coverage split
    L.append('## 7. Damage among MOVED components split by coverage satisfaction (t* ∈ L(α_tgt), interpolated); '
             'per-seed counts listed as s1/s2/s3\n')
    rows = []
    for key in keys:
        ag = aggs[key]
        for a in ARM_ORDER:
            cs = ag['arms'][a]['covsplit']
            rows.append([key, a,
                         '/'.join(str(int(v)) for v in cs['covered']['n_moved']['values']),
                         '/'.join(str(int(v)) for v in cs['covered']['n_dmg']['values']), pm(cs['covered']['dmg_rate']),
                         '/'.join(str(int(v)) for v in cs['uncovered']['n_moved']['values']),
                         '/'.join(str(int(v)) for v in cs['uncovered']['n_dmg']['values']), pm(cs['uncovered']['dmg_rate'])])
    L.append(md_table(['scenario', 'arm', 'covered: #moved', 'covered: #Dmg', 'covered: rate', 'uncovered: #moved',
                       'uncovered: #Dmg', 'uncovered: rate'], rows))

    # 8. residual
    L.append('## 8. CPR2 residual check (MOVED & coverage-satisfied): ratio e_after/(w_i/2)\n')
    rows = []
    for key in keys:
        r = aggs[key]['residual_cpr2']
        rows.append([key, '/'.join(str(int(v)) for v in r['n']['values']), pm(r['median_ratio']), pm(r['q25_ratio']),
                     pm(r['q75_ratio']), pm(r['iqr_ratio']), pm(r['spearman']), pm(r['mean_e_after']), pm(r['mean_half_w']),
                     pm(r['median_w'])])
    L.append(md_table(['scenario', 'n (seeds)', 'median ratio', 'q25', 'q75', 'IQR', 'Spearman(e_after, w/2)', 'mean e_after',
                       'mean w/2', 'median w'], rows))

    # 9. decisions
    L.append('## 9. Decision maps\n')
    rows = []
    for key in keys:
        D = aggs[key]['decisions']
        rows.append([key, pm(D['frac_keep']), pm(D['frac_move']), pm(D['frac_none']), pm(D['frac_acquire']),
                     pm(D['frac_respecify']), pm(D['n_rej'], 1), pm(D['fdp_eps']), '/'.join(str(int(v)) for v in D['n_anomaly_zero_in_L']['values']),
                     '/'.join(str(int(v)) for v in D['ncomp_hist']['1']['values']), '/'.join(str(int(v)) for v in D['ncomp_hist']['2']['values']),
                     '/'.join(str(int(v)) for v in D['ncomp_hist']['3']['values'])])
    L.append(md_table(['scenario', 'KEEP', 'MOVE', 'NONE', 'ACQUIRE', 'RESPECIFY', '#Rej', 'FDP_ε (empirical, point-null p-values)',
                       'anomalies 0∈L (seeds)', 'n_comp=1 (seeds)', 'n_comp=2', 'n_comp=3'], rows))
    L.append('### 9b. KEEP / MOVE / Rej rates by obs_stratum and MOVE / Rej by subset\n')
    rows = []
    for key in keys:
        D = aggs[key]['decisions']
        row = [key]
        for sn in STRATA:
            row += [pm(D['keep_rate_by_stratum'][sn]['rate']), pm(D['move_rate_by_stratum'][sn]['rate']),
                    pm(D['rej_rate_by_stratum'][sn]['rate'])]
        rows.append(row)
    hdr = ['scenario']
    for sn in STRATA:
        hdr += ['%s KEEP' % sn, '%s MOVE' % sn, '%s Rej' % sn]
    L.append(md_table(hdr, rows))
    rows = []
    for key in keys:
        D = aggs[key]['decisions']
        row = [key]
        for sub in SUBSETS:
            row += [pm(D['move_rate_by_subset'][sub]['rate']), pm(D['rej_rate_by_subset'][sub]['rate'])]
        rows.append(row)
    hdr = ['scenario']
    for sub in SUBSETS:
        hdr += ['%s MOVE' % sub, '%s Rej' % sub]
    L.append(md_table(hdr, rows))
    L.append('### 9c. ACQUIRE sensitivity P(ACQUIRE | mm) and false rate P(ACQUIRE | not mm & not twosheet) by obs_stratum; '
             'RESPECIFY sensitivity on misfit and false rate on non-misfit\n')
    rows = []
    for key in keys:
        D = aggs[key]['decisions']
        row = [key]
        for sn in STRATA:
            row += ['%s (n=%s)' % (pm(D['acquire_sens_by_stratum'][sn]['rate']), pm(D['acquire_sens_by_stratum'][sn]['n'], 0)),
                    '%s (n=%s)' % (pm(D['acquire_false_by_stratum'][sn]['rate']), pm(D['acquire_false_by_stratum'][sn]['n'], 0))]
        row += [pm(D['acquire_sens_medium_sharp']['rate']), pm(D['acquire_sens_all']['rate']), pm(D['acquire_false_all']['rate']),
                pm(D['respecify_sens_misfit']['rate']), pm(D['respecify_false_nonmisfit']['rate']),
                '; '.join('[%.4f,%.4f]' % (lo, hi) for lo, hi in zip(D['respecify_false_nonmisfit']['lo']['values'],
                                                                     D['respecify_false_nonmisfit']['hi']['values']))]
        rows.append(row)
    hdr = ['scenario']
    for sn in STRATA:
        hdr += ['%s ACQ sens' % sn, '%s ACQ false' % sn]
    hdr += ['ACQ sens med+sharp', 'ACQ sens all', 'ACQ false all', 'RESPEC sens misfit', 'RESPEC false non-misfit',
            'RESPEC false Wilson per seed']
    L.append(md_table(hdr, rows))
    rows = []
    for key in keys:
        D = aggs[key]['decisions']
        rows.append([key] + [pm(D['respecify_false_nonmisfit_by_stratum'][sn]['rate']) for sn in STRATA])
    L.append(md_table(['scenario', 'RESPEC false flat', 'RESPEC false medium', 'RESPEC false sharp'], rows))

    # 10. priority
    L.append('## 10. Acquisition priority prio_i = π_i·g_i (report only; not VOI) — distribution\n')
    rows = []
    for key in keys:
        P = aggs[key]['priority']
        for grp in ('mm', 'not_mm', 'mm_acquire', 'not_mm_acquire'):
            d = P[grp]
            rows.append([key, grp, pm(d['n'], 0), pm(d['frac_pos']), pm(d['mean']), pm(d['median']), pm(d['p90']), pm(d['max'])])
        for sub in SUBSETS:
            d = P['by_subset'][sub]
            rows.append([key, 'subset=' + sub, pm(d['n'], 0), pm(d['frac_pos']), pm(d['mean']), pm(d['median']), pm(d['p90']), pm(d['max'])])
    L.append(md_table(['scenario', 'group', 'n', 'frac prio>0', 'mean', 'median', 'p90', 'max'], rows))

    # 11. set metrics & sheet
    L.append('## 11. Set metrics (voxel 0.8 dedup) and sheet identity (two-sheet region)\n')
    rows = []
    for key in keys:
        ag = aggs[key]
        for a in ARM_ORDER:
            b = ag['arms'][a]
            rows.append([key, a, pm(b['set']['overall']['accuracy']), pm(b['set']['overall']['completeness']),
                         pm(b['set']['overall']['n_O'], 1), pm(b['set']['overall']['n_T'], 1),
                         pm(b['set']['region']['accuracy']), pm(b['set']['region']['completeness']),
                         pm(b['set']['region']['n_O'], 1), pm(b['set']['region']['n_T'], 1),
                         pm(b['sheet']['back_flip']['rate']), pm(b['sheet']['front_flip']['rate']),
                         pm(b['sheet']['back_moved'], 1), pm(b['sheet']['front_moved'], 1)])
    L.append(md_table(['scenario', 'arm', 'acc overall', 'compl overall', '#O overall', '#T overall', 'acc region', 'compl region',
                       '#O region', '#T region', 'BACK flip', 'FRONT flip', 'BACK moved', 'FRONT moved'], rows))
    L.append('### 11b. Guard statistics\n')
    rows = []
    for key in keys:
        ag = aggs[key]
        for a in ('CPR2_G', 'TEST_ARGMIN_G'):
            g = ag['arms'][a]['guard']
            rows.append([key, a, pm(g['n_moved_before'], 1), pm(g['n_flagged'], 1), pm(g['n_moved_outside_region'], 1),
                         pm(g['n_flagged_outside_region'], 1), pm(g['fp_rate_outside_region']), pm(g['n_moved_in_region'], 1),
                         pm(g['n_flagged_in_region'], 1), pm(g['nn_min_moved']), pm(g['nn_p05_moved']), pm(g['nn_median_moved'])])
    L.append(md_table(['scenario', 'arm', '#moved before guard', '#flagged', '#moved outside region', '#flagged outside region',
                       'guard FP rate (outside region)', '#moved in region', '#flagged in region', 'min NN (moved)', 'p05 NN (moved)',
                       'median NN (moved)'], rows))

    # 12. equivalence per seed
    L.append('## 12. Practical-equivalence dimensions per run (Δ = CPR2 − TEST_ARGMIN)\n')
    rows = []
    for R in all_runs:
        for scope in ['overall'] + list(STRATA):
            eq = R['equivalence']['overall'] if scope == 'overall' else R['equivalence']['by_stratum'][scope]
            rows.append([R['scenario']['key'], R['seed'], scope] +
                        ['%s %s' % (('%+.5f' % eq[k]['delta']) if not math.isnan(eq[k]['delta']) else 'n/a',
                                    ('(in, %s)' % eq[k]['favors']) if eq[k]['within'] else
                                    (('(OUT, %s)' % eq[k]['favors']) if eq[k]['within'] is not None else ''))
                         for k, _, _ in EQ_DIMS])
    L.append(md_table(['scenario', 'seed', 'scope', 'Δ #Dmg/N (±0.002)', 'Δ mean dmg mag (±0.05)', 'Δ p95 dmg (±0.2)',
                       'Δ mean e_after (±0.02)', 'Δ repair cov big (±0.02)'], rows))
    L.append('Moved sets identical (CPR2 vs TEST_ARGMIN) in all runs: %s\n' % all(R['moved_set_identical_CPR2_TEST_ARGMIN'] for R in all_runs))

    # 13. predictions
    L.append('## 13. Pre-registered predictions\n')
    L.append(check_predictions(all_runs, aggs))

    # 14. timing
    L.append('\n## 14. Wall time per run (s)\n')
    rows = [[R['scenario']['key'], R['seed'], '%.1f' % R['wall_time_s']] for R in all_runs]
    L.append(md_table(['scenario', 'seed', 'decision+evaluation wall time (s)'], rows))
    with open(path, 'w') as fh:
        fh.write('\n'.join(L))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(res_dir, fig_dir, aggs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    kx = scenario_key(0.10, 'exch', 0.5)
    ks = scenario_key(0.10, 'sfm', 0.5)

    # fig1
    if kx in aggs:
        ag = aggs[kx]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        xpos = np.arange(len(ARM_ORDER))
        wd = 0.27
        for j, (metric, lab) in enumerate((('dmg_rate_N', 'damage rate #Dmg/N'), ('e_after_mean', 'mean e_after (mm)'))):
            ax = axes[j]
            for k, sn in enumerate(STRATA):
                m = [ag['arms'][a]['by_stratum'][sn][metric]['mean'] for a in ARM_ORDER]
                s = [ag['arms'][a]['by_stratum'][sn][metric]['sd'] for a in ARM_ORDER]
                ax.bar(xpos + (k - 1) * wd, m, wd, yerr=s, capsize=2, label=sn)
            ax.set_xticks(xpos)
            ax.set_xticklabels(ARM_ORDER, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title('%s: %s per arm per obs_stratum (seed mean ± sd)' % (kx, lab), fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig1_damage_eafter_by_arm_stratum.png'), dpi=130)
        plt.close(fig)

    # fig2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for j, fb in enumerate(F_BIGS):
        ax = axes[j]
        xpos = np.arange(len(STRATA))
        wd = 0.2
        series = []
        for calib in CALIBS:
            key = scenario_key(fb, calib, 0.5)
            if key not in aggs:
                continue
            ag = aggs[key]
            series.append(('%s true (I)' % calib, [ag['coverage']['true_tgt_by_stratum'][sn]['rate']['mean'] for sn in STRATA],
                           [ag['coverage']['true_tgt_by_stratum'][sn]['rate']['sd'] for sn in STRATA]))
            series.append(('%s proxy (K\')' % calib, [ag['coverage']['proxy_tgt_Kp_by_stratum'][sn]['rate']['mean'] for sn in STRATA],
                           [ag['coverage']['proxy_tgt_Kp_by_stratum'][sn]['rate']['sd'] for sn in STRATA]))
        for k, (lab, m, s) in enumerate(series):
            ax.bar(xpos + (k - 1.5) * wd, m, wd, yerr=s, capsize=2, label=lab)
        ax.axhline(0.5, color='k', ls='--', lw=1, label='1−α_tgt = 0.5')
        ax.set_xticks(xpos)
        ax.set_xticklabels(STRATA)
        ax.set_ylim(0, 1.05)
        ax.set_title('f_big=%.2f, α_tgt=0.5: coverage of L(α_tgt)' % fb, fontsize=10)
        ax.set_ylabel('coverage')
        if series:
            ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig2_true_vs_proxy_coverage.png'), dpi=130)
    plt.close(fig)

    # fig3
    arms5 = ('IDENTITY', 'CPR2', 'CPR2_G', 'TEST_ARGMIN', 'TEST_ARGMIN_G')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, key in enumerate((kx, ks)):
        if key not in aggs:
            continue
        ag = aggs[key]
        xpos = np.arange(len(arms5))
        for j, (getter, lab) in enumerate((
                (lambda a: ag['arms'][a]['set']['region']['accuracy'], 'region accuracy O→T (mm)'),
                (lambda a: ag['arms'][a]['set']['region']['completeness'], 'region completeness T→O (mm)'),
                (lambda a: ag['arms'][a]['sheet']['back_flip']['rate'], 'BACK flip rate'))):
            ax = axes[r, j]
            m = [getter(a)['mean'] for a in arms5]
            s = [getter(a)['sd'] for a in arms5]
            ax.bar(xpos, m, 0.6, yerr=s, capsize=3, color=['gray', 'C0', 'C1', 'C2', 'C3'])
            ax.set_xticks(xpos)
            ax.set_xticklabels(arms5, rotation=30, ha='right', fontsize=8)
            ax.set_title('%s\n%s' % (key, lab), fontsize=9)
            ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig3_twosheet_region.png'), dpi=130)
    plt.close(fig)

    # fig4 (single runs: seed 1 of the exch and of the sfm scenario with f_big=0.10, alpha_tgt=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key in zip(axes, (kx, ks)):
        dpath = os.path.join(res_dir, 'decisions', '%s_seed1.csv.gz' % key)
        if not os.path.exists(dpath):
            continue
        e_a = []
        hw = []
        strat = []
        with gzip.open(dpath, 'rt') as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                th = float(row['theta_CPR2'])
                if th != 0.0 and int(row['covered_tgt']) == 1:
                    e_a.append(abs(th - float(row['tstar'])))
                    hw.append(float(row['w']) / 2.0)
                    strat.append(row['obs_stratum'])
        e_a = np.array(e_a)
        hw = np.array(hw)
        strat = np.array(strat)
        for sn, col in zip(STRATA, ('C0', 'C1', 'C2')):
            sel = strat == sn
            if sel.any():
                ax.scatter(hw[sel], e_a[sel], s=10, alpha=0.6, label='%s (n=%d)' % (sn, sel.sum()), color=col)
        if hw.size:
            lim = max(hw.max(), e_a.max()) * 1.05
            ax.plot([0, lim], [0, lim], 'k--', lw=1, label='e_after = w/2')
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.legend(fontsize=8)
        ax.set_xlabel('w_i / 2 (mm)')
        ax.set_ylabel('e_after = |θ̂ − t*| (mm)')
        ax.set_title('%s seed 1: CPR2 coverage-satisfied moves (n=%d)' % (key, e_a.size), fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig4_residual_vs_halfwidth.png'), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def load_runs(res_dir):
    runs = []
    d = os.path.join(res_dir, 'per_seed')
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.json'):
            with open(os.path.join(d, fn)) as fh:
                runs.append(json.load(fh))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=200)
    ap.add_argument('--ny', type=int, default=100)
    ap.add_argument('--n-cal', type=int, default=2000)
    ap.add_argument('--seeds', type=str, default='1,2,3')
    ap.add_argument('--res-dir', type=str, default='results')
    ap.add_argument('--fig-dir', type=str, default='figures')
    ap.add_argument('--smoke', action='store_true', help='small grid preset: nx=40, ny=20, n_cal=200, seeds=1,2')
    ap.add_argument('--report-only', action='store_true', help='rebuild RESULTS_RAW.md / aggregate / figures from saved JSON')
    args = ap.parse_args()
    if args.smoke:
        args.nx, args.ny, args.n_cal = 40, 20, 200
        if args.seeds == '1,2,3':
            args.seeds = '1,2'
        if args.res_dir == 'results':
            args.res_dir = 'smoke/results'
        if args.fig_dir == 'figures':
            args.fig_dir = 'smoke/figures'
    seeds = [int(s) for s in args.seeds.split(',')]
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(here, args.res_dir)
    fig_dir = os.path.join(here, args.fig_dir)
    os.makedirs(res_dir, exist_ok=True)
    T0 = time.time()

    def log(msg):
        print('[%7.1fs] %s' % (time.time() - T0, msg))
        sys.stdout.flush()

    log('E0 start: nx=%d ny=%d N=%d n_cal=%d seeds=%s res_dir=%s' % (args.nx, args.ny, args.nx * args.ny, args.n_cal, seeds, res_dir))
    log('numpy %s scipy %s' % (np.__version__, __import__('scipy').__version__))
    if not args.report_only:
        all_runs = []
        for seed in seeds:
            for f_big in F_BIGS:
                all_runs.extend(run_population(seed, f_big, args.nx, args.ny, args.n_cal, res_dir, log))
        all_runs = [to_jsonable(R) for R in all_runs]
    else:
        all_runs = load_runs(res_dir)
        seeds = sorted(set(R['seed'] for R in all_runs))
    by_key = {}
    for R in all_runs:
        by_key.setdefault(R['scenario']['key'], []).append(R)
    aggs = {k: aggregate(sorted(v, key=lambda r: r['seed'])) for k, v in by_key.items()}
    with open(os.path.join(res_dir, 'aggregate.json'), 'w') as fh:
        json.dump(to_jsonable(aggs), fh, indent=1)
    meta = dict(res_dir=os.path.relpath(res_dir, here), nx=args.nx, ny=args.ny, n_cal=args.n_cal)
    md_path = os.path.join(here, 'RESULTS_RAW.md') if not args.smoke else os.path.join(here, 'smoke', 'RESULTS_RAW_smoke.md')
    write_results_md(md_path, all_runs, aggs, seeds, meta)
    log('wrote %s' % md_path)
    make_figures(res_dir, fig_dir, aggs)
    log('wrote figures to %s' % fig_dir)
    log('E0 done, total wall time %.1fs' % (time.time() - T0))


if __name__ == '__main__':
    main()
