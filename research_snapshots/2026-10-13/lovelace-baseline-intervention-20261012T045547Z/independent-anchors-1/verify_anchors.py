#!/usr/bin/env python3
"""Independent exact A8 verifier; no new solver, generator or runner imports.

The source imports frozen independently authored A6/A5 verification arithmetic
and point geometry only. Importing it parses no data or private fact files.
Positive evidence is complete single-clip forward agreement; negative evidence
is a rational contradiction over the full varying-camera necessary outer set.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from contextlib import contextmanager
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import random
import sys
import time

sys.dont_write_bytecode = True
for _anchor_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_anchor_verifier_thread] = '1'
ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/independent-anchors-1')
PARENT = ROOT.parent
PYTHON = Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
_spec = importlib.util.spec_from_file_location('_anchors_frozen_independent_point', PARENT / 'point-feasibility-1/verify_point.py')
_point = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_point)
_old, _proof = _point._old, _point._a5
_geometry = _old._baseline
VerificationError = _point.VerificationError
require, same, exact, encoded = _point.require, _point.same, _point.exact, _point.encoded
load, indexed, finite_seconds = _point.load, _point.indexed, _point.finite_seconds
rational = _proof.rational
verify_certificate = _proof.verify_certificate
BRANCHES = ('none', 'partial', 'full')
ARMS = ('none', 'camera', 'appearance', 'both')
STATES = ('FORWARD_WITNESS', 'UNKNOWN', 'INHERITED_INFEASIBLE', 'CERTIFIED_INFEASIBLE')
RADIUS, BETA, EPSILON, DRIFT = Q(1, 5), Q(1, 100), Q(1, 100), Q(1, 10)
ANCHOR_NOISE, TRANSFER, SURVEY, LOCALIZATION = Q(1, 200), Q(1, 200), Q(1, 100), Q(1, 100)
CENTERS = (-Q(80), Q(0), Q(80))
WITNESS_FIELDS = {'offsets', 'darkness', 't', 'noise', 'tA', 'camera_k', 'camera_noise',
                  'coupon_transfer', 'coupon_noise', 'reference_noise'}


def enabled(arm):
    require(arm in ARMS, 'anchor arm: registered two-by-two cell required')
    return arm in ('camera', 'both'), arm in ('appearance', 'both')


def _sequence(value, length, where, canonical=False):
    require(isinstance(value, (list, tuple)) and len(value) == length, where + ': exact array length required')
    return tuple((rational(v, where) if canonical else exact(v)) for v in value)


def validate_anchor(anchor, observation_id=None):
    require(isinstance(anchor, dict) and set(anchor) == {'observation_id', 'camera', 'appearance'}, 'anchors: exact public measurement schema')
    require(isinstance(anchor['observation_id'], str), 'anchors: observation identity')
    if observation_id is not None:
        same(anchor['observation_id'], observation_id, 'anchors: observation identity mismatch')
    camera, appearance = anchor['camera'], anchor['appearance']
    require(isinstance(camera, dict) and set(camera) == {'u', 'Xf', 'Zf', 'survey_bound', 'localization_bound'}, 'camera: public schema')
    require(tuple(exact(camera[k]) for k in ('Xf', 'Zf', 'survey_bound', 'localization_bound')) ==
            (Q(0), Q(1000), SURVEY, LOCALIZATION), 'camera: fixed independent survey/localization contract')
    _sequence(camera['u'], 3, 'camera u')
    require(isinstance(appearance, dict) and set(appearance) == {'coupons', 'references', 'bias_bound', 'transfer_bound', 'noise_bound'}, 'appearance: public schema')
    require(tuple(exact(appearance[k]) for k in ('bias_bound', 'transfer_bound', 'noise_bound')) ==
            (BETA, TRANSFER, ANCHOR_NOISE), 'appearance: independent bias/transfer/noise bounds')
    for key, size in (('coupons', 9), ('references', 3)):
        require(all(0 <= v <= 1 for v in _sequence(appearance[key], size, key)), 'appearance: exact clipped readings required')
    return anchor


def camera_intervals(anchor, arm):
    """Necessary projections of e_v+k+nu_v=r_v; preserve shared k in forward."""
    use_camera, _ = enabled(arm)
    validate_anchor(anchor)
    if not use_camera:
        return None, ((-RADIUS, RADIUS), (Q(0), Q(0)), (-RADIUS, RADIUS))
    residual = tuple(-center - 1000 * u for center, u in zip(CENTERS, map(exact, anchor['camera']['u'])))
    shared = (max(-SURVEY, residual[1] - LOCALIZATION), min(SURVEY, residual[1] + LOCALIZATION))
    require(shared[0] <= shared[1], 'camera: empty reference-gauge survey interval')
    intervals = []
    for view, value in enumerate(residual):
        interval = (max(-RADIUS, value - shared[1] - LOCALIZATION), min(RADIUS, value - shared[0] + LOCALIZATION))
        if view == 1:
            require(interval[0] <= 0 <= interval[1], 'camera: impossible reference gauge')
            interval = (Q(0), Q(0))
        require(interval[0] <= interval[1], 'camera: empty projected offset interval')
        intervals.append(interval)
    return shared, tuple(intervals)


def public_inputs(public, frozen_arm, depth, branch):
    depth = exact(depth)
    require(branch in BRANCHES and public.get('clipped') is True and exact(public['baseline']) == 80 and
            exact(public['radius']) == RADIUS and _old.DOMAIN[0] <= depth <= _old.DOMAIN[1], 'forward: frozen public geometry')
    require(frozen_arm.get('appearance_model') == 'bounded' and exact(frozen_arm['drift_bound']) == DRIFT, 'forward: frozen bounded model')
    image = _sequence(public['observation'], 432, 'photographic channels')
    require(all(0 <= y <= 1 for y in image), 'forward: clipped photograph required')
    calibration = tuple(_point.pair(p, 'frozen calibration') for p in frozen_arm['calibration_intervals'])
    require(len(calibration) == 3 and all(0 <= lo <= hi <= 2 * BETA for lo, hi in calibration), 'forward: original bias bounds')
    return depth, image, calibration


@lru_cache(maxsize=8192)
def point_areas(branch, depth, offsets):
    return _geometry.independent_point_areas(branch, depth, offsets, Q(80))


def area_breakpoints(branch, depth, view, interval):
    """Every horizontal edge-order change; vertical overlaps are constant."""
    require(branch in BRANCHES and type(view) is int and view in (0, 1, 2), 'area: branch/view identity')
    depth = exact(depth)
    lo, hi = map(exact, interval)
    require(_old.DOMAIN[0] <= depth <= _old.DOMAIN[1] and -RADIUS <= lo <= hi <= RADIUS and
            (view != 1 or lo == hi == 0), 'area: legal depth/offset interval and gauge')
    rectangles = (_geometry.LEFT, _geometry.RIGHT, _geometry.FRONT)
    edges = {(Q(-1) / depth, (rect[index] - CENTERS[view]) / depth) for rect in rectangles for index in (0, 1)}
    if branch != 'none':
        rear = _geometry.REAR if branch == 'partial' else _geometry.HIDDEN
        edges.update((Q(-1) / _geometry.REAR_DEPTH, (x - CENTERS[view]) / _geometry.REAR_DEPTH) for x in rear[:2])
    edges.update((Q(0), pixel[index]) for pixel in _geometry.PIXELS for index in (0, 1))
    lines, cuts = sorted(edges), {lo, hi}
    for index, (slope, intercept) in enumerate(lines):
        for other_slope, other_intercept in lines[:index]:
            if slope != other_slope:
                crossing = (other_intercept - intercept) / (slope - other_slope)
                if lo <= crossing <= hi:
                    cuts.add(crossing)
    return tuple(sorted(cuts))


@lru_cache(maxsize=8192)
def area_enclosure(branch, depth, intervals):
    """Exact component extrema at all piecewise-affine cell endpoints."""
    require(len(intervals) == 3, 'area: three camera intervals required')
    result = []
    for view, interval in enumerate(intervals):
        samples = []
        for offset in area_breakpoints(branch, depth, view, interval):
            offsets = tuple(offset if k == view else Q(0) for k in range(3))
            samples.append(point_areas(branch, exact(depth), offsets)[48 * view:48 * (view + 1)])
        for pixel in range(48):
            result.append(tuple(value for material in range(3) for value in
                                (min(s[pixel][material] for s in samples), max(s[pixel][material] for s in samples))))
    return tuple(result)


def _bands(reading, epsilon):
    y = exact(reading)
    require(0 <= y <= 1, 'anchor bands: clipped scalar required')
    return (None if y == 0 else 1 + BETA - y + epsilon,
            None if y == 1 else 1 + BETA - y - epsilon)


def necessary_problem(public, frozen_arm, anchor, depth, branch, arm):
    """Fifteen variables d9,t3,tA3 over every pose allowed by the active arm."""
    depth, image, calibration = public_inputs(public, frozen_arm, depth, branch)
    validate_anchor(anchor, public['observation_id'])
    _, intervals = camera_intervals(anchor, arm)
    geometry = area_enclosure(branch, depth, intervals)
    original = _old.necessary_rows(image, geometry, True, 'joint', frozen_arm['calibration_intervals'], DRIFT)
    rows = [(side, tuple(coefficients) + (Q(0),) * 3, bound) for side, coefficients, bound in original]
    if enabled(arm)[1]:
        for index, y in enumerate(anchor['appearance']['coupons']):
            coefficients = [Q(0)] * 15
            coefficients[index] = coefficients[12 + index // 3] = Q(1)
            for side, bound in zip(('upper', 'lower'), _bands(y, TRANSFER + ANCHOR_NOISE)):
                if bound is not None:
                    rows.append((side, tuple(coefficients), bound))
        for view, y in enumerate(anchor['appearance']['references']):
            coefficients = [Q(0)] * 15
            coefficients[12 + view] = Q(1)
            for side, bound in zip(('upper', 'lower'), _bands(y, ANCHOR_NOISE)):
                if bound is not None:
                    rows.append((side, tuple(coefficients), bound - Q(1, 2)))
    bounds = ((Q(0), Q(1)),) * 9 + calibration + ((Q(0), 2 * BETA),) * 3
    return tuple(rows), bounds


def verify_forward(public, frozen_arm, anchor, depth, branch, arm, witness, where='forward'):
    depth, image, calibration = public_inputs(public, frozen_arm, depth, branch)
    validate_anchor(anchor, public['observation_id'])
    use_camera, use_appearance = enabled(arm)
    require(isinstance(witness, dict) and set(witness) == WITNESS_FIELDS, where + ': exact full witness schema')
    offsets, darkness, ts, noise = (_sequence(witness[k], size, where + ':' + k, canonical=True)
                                   for k, size in (('offsets', 3), ('darkness', 9), ('t', 3), ('noise', 432)))
    require(offsets[1] == 0 and all(abs(e) <= RADIUS for e in offsets), where + ': common pose/gauge')
    require(all(0 <= d <= 1 for d in darkness) and all(lo <= t <= hi for t, (lo, hi) in zip(ts, calibration)), where + ': materials or photo bias out of bounds')
    require(all(abs(darkness[3 * v + k] - darkness[3 * w + k]) <= DRIFT for v in range(3) for w in range(v + 1, 3) for k in range(3)), where + ': material drift')
    require(all(abs(n) <= EPSILON for n in noise), where + ': photographic noise bound')
    tas = _sequence(witness['tA'], 3, where + ': independent appearance bias', canonical=True)
    require(all(0 <= t <= 2 * BETA for t in tas), where + ': independent appearance bias box')
    clip = lambda value: min(Q(1), max(Q(0), value))
    for pixel, areas in enumerate(point_areas(branch, depth, offsets)):
        view = pixel // 48
        gray = 1 - sum((a * d for a, d in zip(areas, darkness[3 * view:3 * view + 3])), Q(0))
        require(0 <= gray <= 1, where + ': raw physical gray')
        for channel in range(3):
            index = pixel * 3 + channel
            require(clip(gray + BETA - ts[view] + noise[index]) == image[index], f'{where}: single-clip photo mismatch {index}')
    if use_camera:
        k = rational(witness['camera_k'], where + ': shared camera survey')
        camera_noise = _sequence(witness['camera_noise'], 3, where + ': camera noise', canonical=True)
        require(abs(k) <= SURVEY and all(abs(n) <= LOCALIZATION for n in camera_noise), where + ': survey/localization bounds')
        for view, u in enumerate(map(exact, anchor['camera']['u'])):
            require((-CENTERS[view] - offsets[view] - k - camera_noise[view]) / 1000 == u, where + ': shared survey full camera equation')
    else:
        require(witness['camera_k'] is None and witness['camera_noise'] is None, where + ': inactive camera fields must be null')
    appearance_keys = ('coupon_transfer', 'coupon_noise', 'reference_noise')
    if use_appearance:
        transfer, cn, rn = (_sequence(witness[k], size, where + ':' + k, canonical=True)
                            for k, size in zip(appearance_keys, (9, 9, 3)))
        require(all(abs(q) <= TRANSFER for q in transfer) and
                all(abs(n) <= ANCHOR_NOISE for n in cn + rn), where + ': independent appearance error bounds')
        for index, y in enumerate(map(exact, anchor['appearance']['coupons'])):
            require(clip(1 - darkness[index] + BETA - tas[index // 3] + transfer[index] + cn[index]) == y, where + ': coupon single-clip mismatch')
        for view, y in enumerate(map(exact, anchor['appearance']['references'])):
            require(clip(Q(1, 2) + BETA - tas[view] + rn[view]) == y, where + ': reference single-clip mismatch')
    else:
        require(all(witness[k] is None for k in appearance_keys), where + ': inactive appearance fields must be null')
    return True


def aggregate(statuses):
    statuses = tuple(statuses)
    require(len(statuses) == 3 and all(s in STATES for s in statuses), 'union: exact three branch states')
    if 'FORWARD_WITNESS' in statuses:
        return 'FORWARD_WITNESS'
    if all(s in ('INHERITED_INFEASIBLE', 'CERTIFIED_INFEASIBLE') for s in statuses):
        return 'CERTIFIED_INFEASIBLE'
    return 'UNKNOWN'


def extend_saved_witness(public, frozen_arm, anchor, depth, branch, arm, old_witness):
    """Decide existence of NEW measurement errors for an unchanged old world.

    The old offsets/materials/photo-bias/noise are fixed. Shared camera survey
    and each view's independent appearance bias are closed interval feasibility
    problems. Rejection concerns this world only, never the whole depth.
    """
    require(isinstance(old_witness, dict) and set(old_witness) == {'offsets', 'darkness', 't', 'noise'}, 'saved: old four-field world required')
    candidate = {**old_witness, **{key: None for key in WITNESS_FIELDS - set(old_witness)}}
    candidate['tA'] = [encoded(BETA)] * 3
    verify_forward(public, frozen_arm, anchor, depth, branch, 'none', candidate, 'saved original photograph')
    use_camera, use_appearance = enabled(arm)
    if use_camera:
        shared, _ = camera_intervals(anchor, arm)
        lo, hi = shared
        residual = tuple(-c - 1000 * exact(u) - exact(e) for c, u, e in
                         zip(CENTERS, anchor['camera']['u'], old_witness['offsets']))
        for value in residual:
            lo, hi = max(lo, value - LOCALIZATION), min(hi, value + LOCALIZATION)
        if lo > hi:
            return None
        k = (lo + hi) / 2
        candidate['camera_k'] = encoded(k)
        candidate['camera_noise'] = [encoded(value - k) for value in residual]
    if use_appearance:
        darkness = tuple(map(exact, old_witness['darkness']))
        tas, transfer, noises, reference_noises = [], [], [], []
        readings = anchor['appearance']
        for view in range(3):
            lo, hi = Q(0), 2 * BETA
            for index, y in [(index, readings['coupons'][index]) for index in range(3 * view, 3 * view + 3)]:
                upper, lower = _bands(y, TRANSFER + ANCHOR_NOISE)
                if upper is not None:
                    hi = min(hi, upper - darkness[index])
                if lower is not None:
                    lo = max(lo, lower - darkness[index])
            upper, lower = _bands(readings['references'][view], ANCHOR_NOISE)
            if upper is not None:
                hi = min(hi, upper - Q(1, 2))
            if lower is not None:
                lo = max(lo, lower - Q(1, 2))
            if lo > hi:
                return None
            t = (lo + hi) / 2
            tas.append(encoded(t))
            for index in range(3 * view, 3 * view + 3):
                y = exact(readings['coupons'][index])
                residual = -EPSILON if y == 0 else EPSILON if y == 1 else y - (1 - darkness[index] + BETA - t)
                q = min(TRANSFER, max(-TRANSFER, residual))
                transfer.append(encoded(q))
                noises.append(encoded(residual - q))
            y = exact(readings['references'][view])
            n = -ANCHOR_NOISE if y == 0 else ANCHOR_NOISE if y == 1 else y - (Q(1, 2) + BETA - t)
            reference_noises.append(encoded(n))
        candidate.update(tA=tas, coupon_transfer=transfer, coupon_noise=noises, reference_noise=reference_noises)
    verify_forward(public, frozen_arm, anchor, depth, branch, arm, candidate, 'saved extended world')
    return candidate


SEARCH_LIMITS = {'lp_calls': 200, 'exact_attempts': 20, 'branch_seconds': 3.0,
                 'powell_starts': 2, 'powell_maxfev': 60, 'offset_denominator': 1000000000}
SEARCH_FIELDS = {'status', 'witness', 'center_status', 'center_witness', 'numeric_evaluations', 'classifier_lp_calls',
                 'numeric_lp_calls', 'skipped_camera_candidates', 'exact_attempts', 'numeric_failures', 'powell_starts', 'powell_evaluations',
                 'best_phase_slack', 'seconds', 'termination', 'search_budget_seconds', 'limits'}
BRANCH_FIELDS = {'branch', 'status', 'source_status', 'inherited_classification', 'outer_classification',
                 'saved_witness_status', 'saved_extended_witness', 'search', 'witness', 'termination'}
RECORD_FIELDS = {'arm', 'task_id', 'observation_id', 'incumbent', 'a6_status', 'a7_status', 'status', 'branches'}


def verify_search(public, frozen_arm, anchor, depth, branch, arm, record):
    require(isinstance(record, dict) and set(record) == SEARCH_FIELDS, 'search: exact accounting schema')
    same(record['limits'], SEARCH_LIMITS, 'search: fixed proposal caps')
    for key in ('numeric_evaluations', 'classifier_lp_calls', 'numeric_lp_calls', 'skipped_camera_candidates', 'exact_attempts', 'numeric_failures', 'powell_starts'):
        require(type(record[key]) is int and record[key] >= 0, 'search: nonnegative integer ' + key)
    calls, attempts, starts = record['numeric_lp_calls'], record['exact_attempts'], record['powell_starts']
    skipped = record['skipped_camera_candidates']
    require(calls <= 200 and attempts <= 20 and starts <= 2 and record['classifier_lp_calls'] <= attempts and
            skipped <= record['numeric_evaluations'] and
            calls == record['numeric_evaluations'] - skipped + record['classifier_lp_calls'], 'search: exact LP/recovery/camera-skip accounting')
    evaluations = record['powell_evaluations']
    require(isinstance(evaluations, list) and len(evaluations) == starts and all(type(n) is int and 0 <= n <= 60 for n in evaluations), 'search: per-start limits')
    require(sum(evaluations) <= record['numeric_evaluations'] <= 9 + sum(evaluations), 'search: center/grid/Powell accounting')
    finite_seconds(record['seconds'], 'search elapsed')
    require(finite_seconds(record['search_budget_seconds'], 'search budget') <= 3, 'search: three-second cooperative proposal cap')
    if record['best_phase_slack'] is not None:
        finite_seconds(record['best_phase_slack'], 'search slack')
    require(record['status'] in ('FORWARD_WITNESS', 'UNKNOWN') and record['center_status'] in ('FORWARD_WITNESS', 'UNKNOWN'), 'search: numerical results cannot exclude depth')
    require(record['termination'] in ('first_exact_witness', 'branch_timeout', 'whole_deadline', 'lp_budget', 'search_exhausted', 'numeric_failure', 'no_exact_witness'), 'search: registered termination')
    _, intervals = camera_intervals(anchor, arm)
    for status_key, witness_key in (('status', 'witness'), ('center_status', 'center_witness')):
        if record[status_key] == 'FORWARD_WITNESS':
            witness = record[witness_key]
            verify_forward(public, frozen_arm, anchor, depth, branch, arm, witness, 'search ' + witness_key)
            offsets = tuple(map(exact, witness['offsets']))
            require(all(e.denominator <= 1000000000 and lo <= e <= hi for e, (lo, hi) in zip(offsets, intervals)), 'search: candidate offsets within allowed intervals and rationalization cap')
        else:
            require(record[witness_key] is None, 'search: UNKNOWN carries no witness')
    if record['center_status'] == 'FORWARD_WITNESS':
        require(tuple(map(exact, record['center_witness']['offsets'])) == tuple((lo + hi) / 2 for lo, hi in intervals), 'search: actual interval midpoint center')
        same(record['witness'], record['center_witness'], 'search: first successful center chosen')
        require(record['status'] == 'FORWARD_WITNESS' and starts == 0 and record['numeric_evaluations'] == 1, 'search: stop at center success')
    if record['status'] == 'FORWARD_WITNESS':
        require(attempts >= 1 and record['termination'] == 'first_exact_witness', 'search: exact recovery provenance')
    else:
        require(record['termination'] != 'first_exact_witness', 'search: false positive stop')
    return record['seconds']


def verify_branch(record, source, saved, public, frozen_arm, anchor, task, arm):
    require(isinstance(record, dict) and set(record) == BRANCH_FIELDS, 'branch: exact schema')
    branch, depth = source['branch'], task['incumbent']
    same(record['branch'], branch, 'branch: original ordered identity')
    require(record['status'] in STATES and isinstance(record['termination'], str) and record['termination'], 'branch: valid status/termination')
    original = source['point_classification']
    rows, bounds = _point.point_problem(public, frozen_arm, 'joint', branch, depth)
    _proof.verify_classification(rows, bounds, original, 'source A6 global twelve-variable classification')
    same(record['source_status'], original['status'], 'branch: original source status')
    if original['status'] == 'CERTIFIED_INFEASIBLE':
        same(record['inherited_classification'], original, 'branch: full exact original global certificate')
        require(record['status'] == 'INHERITED_INFEASIBLE' and record['saved_witness_status'] == 'NOT_AVAILABLE' and
                all(record[k] is None for k in ('outer_classification', 'saved_extended_witness', 'search', 'witness')), 'branch: inherited certificate cannot search/assert witness')
        return {'status': record['status'], 'search_seconds': 0.0, 'saved_status': 'NOT_AVAILABLE'}
    require(record['inherited_classification'] is None, 'branch: unsupported inherited certificate')
    old_witness = saved['witness'] if saved is not None and saved['status'] == 'FORWARD_WITNESS' else None
    extension = None if old_witness is None else extend_saved_witness(public, frozen_arm, anchor, depth, branch, arm, old_witness)
    expected_saved_status = 'NOT_AVAILABLE' if old_witness is None else 'REJECTED' if extension is None else 'ACCEPTED'
    same(record['saved_witness_status'], expected_saved_status, 'saved: exact existential extension decision; rejection is not depth exclusion')
    if extension is not None:
        supplied = record['saved_extended_witness']
        verify_forward(public, frozen_arm, anchor, depth, branch, arm, supplied, 'saved accepted full world')
        same({key: supplied[key] for key in old_witness}, old_witness, 'saved: no old camera/material/photo parameter refit')
        same(record['witness'], supplied, 'saved: chosen accepted world')
        require(record['status'] == 'FORWARD_WITNESS' and record['outer_classification'] is None and record['search'] is None, 'saved: success terminates branch')
        return {'status': record['status'], 'search_seconds': 0.0, 'saved_status': expected_saved_status}
    require(record['saved_extended_witness'] is None, 'saved: no extended world for unavailable/rejected seed')
    outer = record['outer_classification']
    if outer is not None:
        rows, bounds = necessary_problem(public, frozen_arm, anchor, depth, branch, arm)
        _proof.verify_classification(rows, bounds, outer, 'new full-varying-pose fifteen-variable outer')
        if outer['status'] == 'CERTIFIED_INFEASIBLE':
            require(record['status'] == 'CERTIFIED_INFEASIBLE' and record['witness'] is None and record['search'] is None, 'outer: certified full-pose contradiction is the only new depth exclusion')
            return {'status': record['status'], 'search_seconds': 0.0, 'saved_status': expected_saved_status}
    search = record['search']
    seconds = 0.0 if search is None else verify_search(public, frozen_arm, anchor, depth, branch, arm, search)
    chosen = None if search is None else search['witness']
    same(record['witness'], chosen, 'branch: chosen refitted full world')
    same(record['status'], 'FORWARD_WITNESS' if chosen is not None else 'UNKNOWN', 'branch: no exclusion from numeric/fixed-pose failure')
    if search is None:
        require('budget' in record['termination'].lower() or 'deadline' in record['termination'].lower(), 'branch: explicit budget UNKNOWN when search omitted')
    return {'status': record['status'], 'search_seconds': seconds, 'saved_status': expected_saved_status}


def verify_record_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 1296, 'records: all1296 saved task/arm records required before private')
    identities = indexed(records, lambda value: (value['task_id'], value['arm']), 'record identities')
    same(list(identities), [(task, arm) for task in tasks for arm in ARMS], 'records: exact task-major four-arm ordered matrix')
    for record in records:
        require(isinstance(record, dict) and set(record) == RECORD_FIELDS, 'records: exact schema')
        task = tasks[record['task_id']]
        require(task['observation_id'] in observations, 'records: known image required')
        same(record['observation_id'], task['observation_id'], 'records: exact image identity')
        same(record['incumbent'], task['incumbent'], 'records: unchanged incumbent')
        require(isinstance(record['branches'], list) and len(record['branches']) == 3, 'records: all3 branches required')
        same([b['branch'] for b in record['branches']], list(BRANCHES), 'records: exact original branch order')
        same(record['status'], aggregate(b['status'] for b in record['branches']), 'records: existential branch union')
    require(sum(len(record['branches']) for record in records) == 3888, 'records: all3888 branch records')
    return identities


A6 = PARENT / 'point-feasibility-1'
A7 = PARENT / 'common-pose-feasibility-1'
PRIOR = A6 / 'outputs/point-v1'
OLD_OUTPUT = A7 / 'outputs/pose-v1'
A5_OUTPUT = PARENT / 'joint-feasibility-1/outputs/polytope-v1'
OLD_DATA = PARENT / 'appearance-variation-1/data'
SCIENCE_NAMES = ('AGENTS.md', 'PROTOCOL.md', 'REGISTRATION.json', 'prepare_anchors.py', 'anchor_solver.py',
                 'run_anchors.py', 'execute_anchors.py', 'verify_anchors.py', 'test_anchor_solver.py',
                 'test_verify_anchors.py', 'test_anchor_pipeline.py')
BUDGET_SCOPE = 'A8 one generation/main/independent-verification window; no reset'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_target(output):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact A8 root on liekkas')
    require(Path(sys.executable) == PYTHON, 'target: existing single-CPU open3d-019 runtime')
    output = Path(output).resolve()
    require(output == ROOT / 'outputs/anchor-v1' and output.is_dir(), 'target: exact existing A8 formal output')
    return output


def source_files():
    # Inherit the sealed explicit source inventory, never re-glob an old root.
    # Its SOURCE_LOCK bytes are themselves covered by the A6 artifact seal.
    lock = load(PRIOR / 'SOURCE_LOCK.json')
    require(lock.get('source_root') == str(A6) and isinstance(lock.get('files'), dict), 'source: exact sealed A6 source graph')
    files = dict(lock['files'])
    for path, digest in files.items():
        same(sha(path), digest, 'source: unchanged inherited bytes ' + path)
    for path in (A7 / 'pose_solver.py', A7 / 'pose_lift.py', *(ROOT / name for name in SCIENCE_NAMES)):
        files[str(path)] = sha(path)
    return dict(sorted(files.items()))


def input_files():
    paths = (A6 / 'ARTIFACTS.json', A6 / 'FINAL_SEAL.json', PRIOR / 'diagnoses.json', PRIOR / 'observations.json',
             PRIOR / 'tasks.json', PRIOR / 'STAGE_STATUS.json', PRIOR / 'verification/VERIFY.json', A5_OUTPUT / 'outers.json',
             A7 / 'ARTIFACTS.json', A7 / 'FINAL_SEAL.json', OLD_OUTPUT / 'diagnoses.json', OLD_OUTPUT / 'STAGE_STATUS.json',
             A7 / 'diagnostic-audit/SAVED_RECORD_AUDIT.json', OLD_DATA / 'private/truth.json', ROOT / 'data/anchors.json',
             ROOT / 'data/private/measurements.json', ROOT / 'data/GENERATION.json')
    return {str(path): sha(path) for path in paths}


def verify_old_seals():
    for stage, root in (('A6', A6), ('A7', A7)):
        inventory, seal = load(root / 'ARTIFACTS.json'), load(root / 'FINAL_SEAL.json')
        require(inventory.get('stage') == stage and inventory.get('host') == 'liekkas' and inventory.get('root') == str(root), 'seal: exact old stage/host/root')
        files = inventory.get('files')
        require(isinstance(files, dict) and files and inventory.get('count') == len(files), 'seal: complete old archive count')
        for relative, digest in files.items():
            path = root / relative
            require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root) and path.is_file(), 'seal: local archived path')
            same(sha(path), digest, 'seal: unchanged archived bytes ' + relative)
        require(seal.get('stage') == stage and seal.get('host') == 'liekkas' and seal.get('root') == str(root) and
                seal.get('artifact_count_excluding_self_files') == len(files) and
                seal.get('artifact_index_sha256') == sha(root / 'ARTIFACTS.json'), 'seal: immutable inventory linkage')
        require(seal.get('status') == ('COMPLETE' if stage == 'A6' else 'INCOMPLETE'), 'seal: exact preserved formal status')
    # A6's sealed manifest also locks external old observations/truth/outers.
    # Hash them without parsing private JSON before the complete public gate.
    inherited_inputs = load(PRIOR / 'INPUTS.json')
    require(set(inherited_inputs) == {'files'} and isinstance(inherited_inputs['files'], dict), 'source: sealed A6 dependency manifest')
    for path, digest in inherited_inputs['files'].items():
        same(sha(path), digest, 'source: unchanged old external input bytes ' + path)
    require(load(PRIOR / 'STAGE_STATUS.json').get('status') == 'COMPLETE' and
            load(PRIOR / 'verification/VERIFY.json').get('status') == 'PASS', 'source: A6 complete and mechanically verified')
    audit = load(A7 / 'diagnostic-audit/SAVED_RECORD_AUDIT.json')
    require(audit.get('status') == 'SAVED_RECORD_NUMERICAL_PASS' and audit.get('formal_acceptance') is False and
            load(OLD_OUTPUT / 'STAGE_STATUS.json').get('status') == 'INCOMPLETE', 'source: A7 diagnostic evidence only; formal INCOMPLETE preserved')


def verify_locks(output):
    sources, inputs = source_files(), input_files()
    same(load(output / 'SOURCE_LOCK.json'), {'files': sources, 'explicit_local_allowlist': list(SCIENCE_NAMES)}, 'lock: independently enumerated explicit old/new source set')
    same(load(output / 'INPUTS.json'), {'files': inputs}, 'lock: exact frozen/generated byte graph')
    verify_old_seals()
    return sources, inputs


def verify_registration(value):
    same(value, {'stage': 'A8', 'seed': 61006, 'arms': list(ARMS), 'observations': 36, 'tasks_per_arm': 324,
                 'branches_per_arm': 972, 'total_records': 1296, 'total_branches': 3888, 'primary_per_arm': 40,
                 'primary_eta': '1/10', 'primary_truth': '9007/14', 'variables': 15, 'whole_seconds': 3600, 'main_seconds': 3000,
                 'branch_seconds': 3, 'max_lp_calls': 200, 'max_exact_attempts': 20, 'powell_starts': 2, 'powell_maxfev': 60,
                 'old_stage_status': 'INCOMPLETE', 'geometry_output': 'unchanged; feasibility diagnostics only',
                 'retry_policy': 'one formal batch; no parameter change after source freeze'}, 'registration: fixed A8 matrix/contracts')


def verify_budget(value, deadline):
    fields = {'start_epoch', 'deadline_epoch', 'main_deadline_epoch', 'limit_seconds', 'scope'}
    require(isinstance(value, dict) and set(value) == fields, 'budget: exact shared window schema')
    start = finite_seconds(value['start_epoch'], 'budget start')
    end = finite_seconds(value['deadline_epoch'], 'budget end')
    require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline == end == start + 3600 and
            value['main_deadline_epoch'] == start + 3000 and type(value['limit_seconds']) is int and value['limit_seconds'] == 3600 and
            value['scope'] == BUDGET_SCOPE, 'budget: same generation/main/verification deadline; no reset')
    if time.time() >= end:
        raise TimeoutError('A8 shared absolute deadline exhausted')
    return end


@contextmanager
def deadline_guard(output, deadline):
    end = verify_budget(load(output / 'BUDGET.json'), deadline)
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'budget: no active alarm reset')
    def expired(_signum, _frame):
        raise TimeoutError('A8 shared absolute deadline reached')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, end - time.time()))
    try:
        yield end
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify_public_population(output):
    publics, task_rows = load(PRIOR / 'observations.json'), load(PRIOR / 'tasks.json')
    observations = indexed(publics, lambda row: row['observation_id'], 'frozen public observation identity')
    tasks = indexed(task_rows, lambda row: row['task_id'], 'frozen public task identity')
    expected = [f'p{i:03d}b80e{variation}' for variation in range(2) for i in range(1, 19)]
    same(list(observations), expected, 'population: all36 original images')
    same(list(tasks), [oid + f's{state}' for oid in expected for state in range(9)], 'population: all324 original incumbents')
    same(load(output / 'observations.json'), publics, 'population: exact saved public observation readback')
    same(load(output / 'tasks.json'), task_rows, 'population: exact saved public task readback')
    anchors = indexed(load(ROOT / 'data/anchors.json'), lambda row: row['observation_id'], 'public measurement identity')
    same(list(anchors), expected, 'population: all36 independent measurements')
    for oid, public in observations.items():
        require(public['variation_index'] == int(oid[-1]) and public['unit_id'] == ('appearance_eta_0' if oid[-1] == '0' else 'appearance_eta_1_10'), 'population: eta/unit identity')
        validate_anchor(anchors[oid], oid)
        camera_intervals(anchors[oid], 'camera')
        if oid[-1] == '1':
            same(anchors[oid]['camera'], anchors[oid[:-1] + '0']['camera'], 'population: paired camera acquisition across eta')
    for tid, task in tasks.items():
        require(task['observation_id'] == tid[:-2], 'population: task/image mapping')
        exact(task['incumbent'])
    return observations, tasks, anchors


def audit_information_flow():
    """Source audit complements the concrete saved-record gate in this CLI."""
    path = ROOT / 'run_anchors.py'
    tree = ast.parse(path.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require('main' in functions and 'input_paths' in functions, 'runner audit: public entry points')
    private_literals = [(name, n.value) for name, function in functions.items() for n in ast.walk(function)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str) and 'private/' in n.value]
    require(private_literals and all(name == 'input_paths' for name, _ in private_literals), 'runner audit: private paths only in byte-hash enumeration')
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('eval', 'exec') for n in ast.walk(tree)), 'runner audit: no dynamic data evaluation')
    main = functions['main']
    dumps = {ast.dump(n, include_attributes=False): n for n in ast.walk(main)}
    for expression in ("save(output/'diagnoses.json',records)", "reread=load(output/'diagnoses.json')",
                       "if reread!=records:raise RuntimeError('record readback mismatch')"):
        require(ast.dump(ast.parse(expression).body[0], include_attributes=False) in dumps, 'runner audit: complete saved/readback public record gate')
    return {'status': 'PASS', 'runner': str(path), 'private_paths_only_hashed_in_runner': True,
            'private_evaluation_owner': str(Path(__file__).resolve()), 'evaluation_after_all_public_records': True}


def _expected_tapes(observations):
    rng = random.Random(61006)
    tapes = {}
    for public in observations.values():
        base = public['base_observation_id']
        if base in tapes:
            continue
        draw = lambda bound: Q(rng.randint(-bound, bound), 100000)
        tapes[base] = {'camera_k': encoded(draw(1000)), 'camera_noise': [encoded(draw(1000)) for _ in range(3)],
                       'appearance_t': [encoded(BETA - draw(1000)) for _ in range(3)],
                       'coupon_transfer': [encoded(draw(500)) for _ in range(9)],
                       'coupon_noise': [encoded(draw(500)) for _ in range(9)],
                       'reference_noise': [encoded(draw(500)) for _ in range(3)]}
    require(len(tapes) == 18, 'generation: exact18 paired base tapes')
    return tapes


def actual_witness(fact, measurement, arm):
    camera, appearance = enabled(arm)
    return {'offsets': [encoded(exact(x)) for x in fact['actual_offsets']],
            'darkness': [encoded(1 - exact(x)) for x in fact['view_colors']],
            't': [encoded(BETA - exact(x)) for x in fact['actual_view_bias']],
            'noise': [encoded(exact(x)) for x in fact['image_noise']],
            'tA': measurement['appearance_t'],
            'camera_k': measurement['camera_k'] if camera else None,
            'camera_noise': measurement['camera_noise'] if camera else None,
            'coupon_transfer': measurement['coupon_transfer'] if appearance else None,
            'coupon_noise': measurement['coupon_noise'] if appearance else None,
            'reference_noise': measurement['reference_noise'] if appearance else None}


def evaluate_after_public_gate(observations, tasks, anchors, frozen, records):
    """The only private parsing route, called after complete public verification."""
    facts = indexed(load(OLD_DATA / 'private/truth.json'), lambda row: row['observation_id'], 'old exact generating facts')
    measurements = indexed(load(ROOT / 'data/private/measurements.json'), lambda row: row['observation_id'], 'new independent acquisition tapes')
    same(list(facts), list(observations), 'private: all36 frozen generating worlds')
    same(list(measurements), list(observations), 'private: all36 acquisition records')
    tapes = _expected_tapes(observations)
    coverage = []
    for oid, public in observations.items():
        fact, measurement = facts[oid], measurements[oid]
        same(measurement, {'observation_id': oid, 'base_observation_id': public['base_observation_id'], **tapes[public['base_observation_id']]}, 'generation: exact seed61006 independent paired tape')
        require(fact['variation_index'] == public['variation_index'] and fact['unit_id'] == public['unit_id'], 'private: correct eta lineage')
        depth, branch = exact(fact['truth']), fact['rear_branch']
        for arm in ARMS:
            witness = actual_witness(fact, measurement, arm)
            oldarm = frozen[oid]['box']['joint']
            verify_forward(public, oldarm, anchors[oid], depth, branch, arm, witness, 'actual full photograph and independent acquisitions')
            actual_point = witness['darkness'] + witness['t'] + measurement['appearance_t']
            rows, bounds = necessary_problem(public, oldarm, anchors[oid], depth, branch, arm)
            _proof.verify_witness(rows, bounds, actual_point, 'actual world inside full-pose necessary outer')
            correct = [record for record in records if record['observation_id'] == oid and record['arm'] == arm and exact(record['incumbent']) == depth]
            require(len(correct) == 1, 'coverage: exactly one original correct incumbent per world/arm')
            generating = [b for b in correct[0]['branches'] if b['branch'] == branch]
            require(len(generating) == 1 and generating[0]['status'] not in ('INHERITED_INFEASIBLE', 'CERTIFIED_INFEASIBLE') and
                    correct[0]['status'] != 'CERTIFIED_INFEASIBLE', 'coverage: true generating world cannot be excluded')
            coverage.append({'observation_id': oid, 'arm': arm, 'generating_branch': branch, 'truth_mm': str(depth),
                             'actual_photo_and_enabled_anchors_verified': True, 'actual_necessary_outer_verified': True,
                             'correct_input_status': correct[0]['status'], 'generating_branch_status': generating[0]['status']})
    require(len(coverage) == 144, 'coverage: all36 worlds times4 arms')
    results = []
    for record in records:
        public, fact = observations[record['observation_id']], facts[record['observation_id']]
        truth, depth = exact(fact['truth']), exact(record['incumbent'])
        branches = record['branches']
        results.append({'task_id': record['task_id'], 'observation_id': record['observation_id'], 'arm': record['arm'],
                        'unit_id': public['unit_id'], 'variation_index': public['variation_index'], 'case': fact['case'],
                        'truth_mm': str(truth), 'incumbent_mm': str(depth), 'initial_ae_mm': str(abs(depth - truth)),
                        'a6_status': record['a6_status'], 'a7_status': record['a7_status'], 'status': record['status'],
                        'new_depth_exclusion': record['status'] == 'CERTIFIED_INFEASIBLE' and record['a6_status'] != 'CERTIFIED_INFEASIBLE',
                        'saved_witness_rejected_branches': sum(b['saved_witness_status'] == 'REJECTED' for b in branches),
                        'saved_witness_accepted_branches': sum(b['saved_witness_status'] == 'ACCEPTED' for b in branches),
                        'refitted_forward_branches': sum(b['search'] is not None and b['search']['status'] == 'FORWARD_WITNESS' for b in branches),
                        'new_certified_branches': sum(b['status'] == 'CERTIFIED_INFEASIBLE' for b in branches),
                        'inherited_certified_branches': sum(b['status'] == 'INHERITED_INFEASIBLE' for b in branches),
                        'unknown_branches': sum(b['status'] == 'UNKNOWN' for b in branches),
                        'additional_scalars': {'none': 0, 'camera': 3, 'appearance': 12, 'both': 15}[record['arm']]})
    return results, coverage


def summary_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        shifted = exact(row['initial_ae_mm']) != 0
        cells = [('all', ''), ('shifted' if shifted else 'correct', '')]
        if shifted:
            informative = row['case'] != 'equal_full_negative'
            cells.extend([('informative_shifted' if informative else 'negative_shifted', ''), ('case_shifted', row['case'])])
            if informative:
                cells.append(('depth_informative', row['truth_mm']))
        for group, value in cells:
            groups[row['arm'], row['unit_id'], group, value].append(row)
    return [{'arm': arm, 'unit_id': unit, 'group': group, 'value': value, 'n': len(selected),
             'status_counts': dict(sorted(Counter(row['status'] for row in selected).items())),
             'new_depth_exclusions': sum(row['new_depth_exclusion'] for row in selected),
             **{key: sum(row[key] for row in selected) for key in ('saved_witness_rejected_branches', 'saved_witness_accepted_branches',
                 'refitted_forward_branches', 'new_certified_branches', 'inherited_certified_branches', 'unknown_branches')},
             'additional_scalars_per_observation': {'none': 0, 'camera': 3, 'appearance': 12, 'both': 15}[arm]}
            for (arm, unit, group, value), selected in sorted(groups.items())]


def verify_summary_denominators(summary):
    for arm in ARMS:
        for unit in ('appearance_eta_0', 'appearance_eta_1_10'):
            cells = {(r['group'], r['value']): r for r in summary if r['arm'] == arm and r['unit_id'] == unit}
            for group, count in (('all', 162), ('shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
                require(cells.get((group, ''), {}).get('n') == count, 'summary: complete fixed ' + group + ' denominator')
            depths = [row for (group, _), row in cells.items() if group == 'depth_informative']
            cases = [row for (group, _), row in cells.items() if group == 'case_shifted']
            require(len(depths) == 3 and all(row['n'] == 40 for row in depths) and len(cases) == 6 and all(row['n'] == 24 for row in cases) and
                    cells.get(('depth_informative', '9007/14'), {}).get('n') == 40, 'summary: fixed primary40/depth/template denominator')


def save_json(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT / 'outputs/anchor-v1'), 'write: exact A8 formal output only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def verify(output, deadline):
    output = check_target(output)
    budget = load(output / 'BUDGET.json')
    verify_budget(budget, deadline)
    sources, inputs = verify_locks(output)
    registration = load(ROOT / 'REGISTRATION.json')
    verify_registration(registration)
    same(load(output / 'REGISTRATION.json'), registration, 'registration: exact saved copy')
    flow = audit_information_flow()
    observations, tasks, anchors = verify_public_population(output)
    records = load(output / 'diagnoses.json')
    verify_record_population(records, observations, tasks)
    prior_rows = load(PRIOR / 'diagnoses.json')
    prior = indexed([r for r in prior_rows if r['method'] == 'joint'], lambda r: r['task_id'], 'A6 joint source rows')
    saved = indexed(load(OLD_OUTPUT / 'diagnoses.json'), lambda r: r['task_id'], 'A7 saved diagnostic rows')
    same(list(prior), list(tasks), 'source: complete324 old joint diagnoses')
    same(list(saved), list(tasks), 'source: complete324 saved A7 diagnoses')
    frozen = load(A5_OUTPUT / 'outers.json')
    counts, saved_counts, seconds = Counter(), Counter(), 0.0
    for index, record in enumerate(records):
        task, arm = tasks[record['task_id']], record['arm']
        oid = task['observation_id']
        source, old_record = prior[task['task_id']], saved[task['task_id']]
        same(source['incumbent'], task['incumbent'], 'source: old A6 exact depth')
        same(old_record['incumbent'], task['incumbent'], 'source: old A7 exact depth')
        same([b['branch'] for b in source['global']], list(BRANCHES), 'source: A6 all3 global branches')
        same([b['branch'] for b in old_record['branches']], list(BRANCHES), 'source: A7 all3 saved branches')
        same(record['a6_status'], source['global_status'], 'record: unchanged A6 union context')
        same(record['a7_status'], old_record['status'], 'record: unchanged diagnostic A7 context')
        for branch, old6, old7 in zip(record['branches'], source['global'], old_record['branches']):
            checked = verify_branch(branch, old6, old7, observations[oid], frozen[oid]['box']['joint'], anchors[oid], task, arm)
            counts[checked['status']] += 1
            saved_counts[checked['saved_status']] += 1
            seconds += checked['search_seconds']
        if index % 36 == 35:
            print(json.dumps({'stage': 'A8-independent-public-verification', 'records_complete': index + 1,
                              'remaining_seconds': round(deadline - time.time(), 2)}), flush=True)
    require(sum(counts.values()) == sum(saved_counts.values()) == 3888, 'public: complete3888 checked branches')
    checkpoint = output / 'checkpoint'
    same(sorted(p.name for p in checkpoint.iterdir()), sorted(oid + '.json' for oid in observations), 'checkpoint: exact36 saved image groups')
    for oid in observations:
        same(load(checkpoint / (oid + '.json')), {'observation_id': oid, 'records': [r for r in records if r['observation_id'] == oid]}, 'checkpoint: full public readback')
    status = load(output / 'RUN_STATUS.json')
    require(set(status) == {'status', 'records_complete', 'branch_records', 'observations_complete', 'seconds', 'deadline_epoch'}, 'main status: exact completed schema')
    same({k: x for k, x in status.items() if k != 'seconds'}, {'status': 'COMPLETE', 'records_complete': 1296,
         'branch_records': 3888, 'observations_complete': 36, 'deadline_epoch': deadline}, 'main status: full saved matrix/common window')
    require(seconds <= finite_seconds(status['seconds'], 'main seconds') < 3600, 'timing: cumulative searches bounded by main window')
    # This is the sole private parse, after all1296 records/all3888 branches.
    results, coverage = evaluate_after_public_gate(observations, tasks, anchors, frozen, records)
    summary = summary_rows(results)
    verify_summary_denominators(summary)
    generation = load(ROOT / 'data/GENERATION.json')
    same(generation, {'stage': 'A8', 'seed': 61006, 'observations': 36, 'base_tapes': 18,
         'camera_scalars_per_observation': 3, 'appearance_scalars_per_observation': 12,
         'old_private_sha256': inputs[str(OLD_DATA / 'private/truth.json')], 'old_public_sha256': inputs[str(PRIOR / 'observations.json')],
         'forward': 'independent shared survey, independent appearance acquisition bias, bounded errors, clip once',
         'paired_across_eta': True, 'deadline_epoch': deadline}, 'generation: exact independent acquisition provenance')
    same(source_files(), sources, 'lock: source bytes unchanged through independent verification')
    same(input_files(), inputs, 'lock: input bytes unchanged through independent verification')
    verify_budget(budget, deadline)
    save_json(output / 'results.json', results)
    save_json(output / 'coverage.json', coverage)
    save_json(output / 'summary.json', summary)
    with (output / 'results.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    same(load(output / 'results.json'), results, 'evaluation: exact typed result readback')
    same(load(output / 'coverage.json'), coverage, 'evaluation: full144 actual-world readback')
    same(load(output / 'summary.json'), summary, 'evaluation: exact summary readback')
    _old._baseline.verify_csv(output / 'results.csv', results)
    verify_budget(budget, deadline)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'public_records': 1296, 'branch_records': 3888, 'actual_world_arm_checks': 144,
            'branch_status_counts': dict(sorted(counts.items())), 'saved_witness_status_counts': dict(sorted(saved_counts.items())),
            'source_files_verified': len(sources), 'input_files_verified': len(inputs), 'information_flow': flow,
            'old_A7_formal_status': 'INCOMPLETE', 'old_A7_evidence': 'SAVED_RECORD_NUMERICAL_PASS only',
            'shared_deadline_epoch': deadline, 'elapsed_generation_main_verification_seconds': time.time() - budget['start_epoch'],
            'assurance': 'same-family finite-instance mechanical consistency; no scientific novelty or physical-transfer claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/anchor-v1')
    parser.add_argument('--deadline', type=float, required=True)
    arguments = parser.parse_args()
    output = check_target(arguments.output)
    with deadline_guard(output, arguments.deadline):
        receipt = verify(output, arguments.deadline)
        save_json(output / 'verification/VERIFY.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
