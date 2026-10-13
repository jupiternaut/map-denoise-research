"""Fresh A12 worlds using the frozen exact camera/material/sensor equations.

Only the evaluator/generator may import this module. The public method packet
is the three-key observation returned by public_packet, never a world record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

sys.dont_write_bytecode = True
for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_setting] = '1'
ROOT = Path(__file__).resolve().parent
LOGICAL_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
OLD = LOGICAL_ROOT.parent / 'lovelace-baseline-intervention-20261012T045547Z'
SEEDS = {'dev': (20261013071,), 'holdout': (20261013081, 20261013082, 20261013083)}
OLD_DEPTHS = frozenset((Q(6203, 11), Q(7861, 13), Q(9007, 14)))
BRANCHES = ('none', 'partial', 'full')
STRATA = ('contrast', 'lowcontrast')
MECHANISM = {
    'family': 'same physical scene/camera/sensor family; new rational parameter holdout',
    'sampling_change': 'expanded latent sampling distribution, not the historical six fixed materials/three depths/discrete noise tape',
    'depth': '550 + integer(1,100*100003-1)/100003; reject historical depths',
    'camera_offsets': 'side offsets independent integers[-20000,20000]/100000; center=0',
    'photo_bias_and_noise': 'independent integers[-1000,1000]/100000; one shared bias per view',
    'materials': 'three new base colors; original A4-V patterned view variation eta=1/10 except all-equal full negative eta=0',
    'anchors': 'original A8 tape/acquire, independent of photo errors',
    'tasks': 'one correct + six independent nonzero signed magnitudes integer[1000,80000]/10000; reject accidental mean/median symmetry; shuffle',
    'clipping': 'exact ideal pixel integral + photo noise + shared photo bias, then one clip',
    'split_unit': 'independently generated world; repeated incumbents are not independent scenes',
}


def exact(value):
    return Q(value['numerator'], value['denominator']) if isinstance(value, dict) else Q(value)


def enc(value):
    value = exact(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@lru_cache(maxsize=1)
def dependencies():
    return (
        module('_a12_legacy_photo_generator', OLD / 'appearance-variation-1/prepare_variation.py'),
        module('_a12_legacy_anchor_generator', OLD / 'independent-anchors-1/prepare_anchors.py'),
        module('_a12_independent_photo_verifier', OLD / 'appearance-variation-1/verify_variation.py'),
        module('_a12_independent_anchor_verifier', OLD / 'independent-anchors-1/verify_anchors.py'),
        module('_a12_public_float_forward', ROOT / 'task_bundle/forward.py'),
    )


def float_tree(value):
    if isinstance(value, dict):
        if set(value) == {'numerator', 'denominator'}:
            return float(exact(value))
        return {k: float_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [float_tree(v) for v in value]
    return value


def public_packet(photo, anchor):
    return {'photo': [float(exact(v)) for v in photo],
            'camera': float_tree(anchor['camera']), 'appearance': float_tree(anchor['appearance'])}


def sample_colors(rng, stratum, negative):
    if negative:
        c = Q(rng.randint(15000, 65000), 100000)
        return (c, c, c)
    if stratum == 'contrast':
        return (Q(rng.randint(10000, 25000), 100000),
                Q(rng.randint(65000, 85000), 100000),
                Q(rng.randint(35000, 55000), 100000))
    center = rng.randint(20000, 60000)
    return tuple(Q(center + rng.randint(-500, 500), 100000) for _ in range(3))


def sample_incumbents(rng, truth):
    # No paired magnitudes, catalog offsets, or task ordering that labels truth.
    # Conditioning only prevents accidental deterministic center disclosure.
    while True:
        shifts = [rng.choice((-1, 1)) * Q(rng.randint(1000, 80000), 10000) for _ in range(6)]
        if (len(set(map(abs, shifts))) == 6 and sum(shifts, Q(0)) != 0
                and sorted([Q(0), *shifts])[3] != 0):
            break
    values = [truth, *(truth + shift for shift in shifts)]
    rng.shuffle(values)
    return values


def world(rng, branch, stratum, split, seed, block):
    generate, anchor_generate, verify_photo, verify_anchor, float_forward = dependencies()
    oid = 'w_' + f'{rng.getrandbits(96):024x}'
    depth = Q(550) + Q(rng.randint(1, 100 * 100003 - 1), 100003)
    while depth in OLD_DEPTHS:
        depth = Q(550) + Q(rng.randint(1, 100 * 100003 - 1), 100003)
    negative = branch == 'full' and stratum == 'lowcontrast'
    colors = sample_colors(rng, stratum, negative)
    eta = Q(0) if negative else Q(1, 10)
    draw = lambda n: Q(rng.randint(-n, n), 100000)
    fact = {'observation_id': oid, 'truth': enc(depth), 'rear_branch': branch,
            'colors': list(map(enc, colors)), 'baseline': '80',
            'actual_offsets': list(map(enc, (draw(20000), Q(0), draw(20000)))),
            'actual_view_bias': [enc(draw(1000)) for _ in range(3)],
            'image_noise': [enc(draw(1000)) for _ in range(432)]}
    image, darkness = generate.render_views(fact, eta)
    fact['view_colors'] = [enc(1 - d) for d in darkness]
    fact.update({'eta': enc(eta), 'stratum': stratum, 'negative_control': negative,
                 'split': split, 'seed_block': block, 'seed': seed,
                 'actual_material_spread': enc(max(max(darkness[k::3]) - min(darkness[k::3]) for k in range(3)))})
    measurement = anchor_generate.tape(rng)
    anchor = anchor_generate.acquire(fact, measurement)
    ideal = verify_photo.independent_rgb_per_view(branch, depth, fact['actual_offsets'], fact['view_colors'], Q(80))
    noise, bias = tuple(map(exact, fact['image_noise'])), tuple(map(exact, fact['actual_view_bias']))
    independently_observed = [enc(min(Q(1), max(Q(0), value + noise[i] + bias[i // 144]))) for i, value in enumerate(ideal)]
    if image != independently_observed:
        raise AssertionError('independent exact photographic replay failed')
    floats = float_forward.render(float(depth), branch, list(map(float, map(exact, fact['actual_offsets']))),
                                  list(map(float, map(exact, fact['view_colors']))))
    max_error = max(abs(float(value) - float(other)) for value, other in zip(ideal, floats))
    observed_error = max(abs(float(exact(y)) - min(1., max(0., float(floats[i]) + float(noise[i]) + float(bias[i // 144]))))
                         for i, y in enumerate(image))
    if len(floats) != 432 or max(max_error, observed_error) > 1e-12:
        raise AssertionError('public floating renderer disagrees with exact original')
    public_exact = {'observation_id': oid, 'baseline': '80', 'radius': '1/5', 'clipped': True, 'observation': image}
    public_model = {'appearance_model': 'bounded', 'drift_bound': enc(Q(1, 10)),
                    'calibration_intervals': [[enc(0), enc(Q(1, 50))] for _ in range(3)]}
    witness = verify_anchor.actual_witness(fact, measurement, 'both')
    verify_anchor.verify_forward(public_exact, public_model, anchor, depth, branch, 'both', witness)
    tasks = []
    for value in sample_incumbents(rng, depth):
        tasks.append({'task_id': 't_' + f'{rng.getrandbits(96):024x}', 'observation_id': oid,
                      'incumbent': float(value), 'incumbent_exact': enc(value)})
    fact.update({'anchor_tape': measurement, 'photo_exact': image, 'anchor_exact': anchor,
                 'independent_exact_verified': True, 'float_forward_max_abs_error': max_error,
                 'float_observation_max_abs_error': observed_error})
    return {'observation_id': oid, 'observation': public_packet(image, anchor)}, fact, tasks


def build_population(split, seeds=None):
    if split not in SEEDS:
        raise ValueError('unknown split')
    observations, facts, tasks = [], [], []
    for block, seed in enumerate(SEEDS[split] if seeds is None else seeds):
        rng = random.Random(seed)
        for branch in BRANCHES:
            for stratum in STRATA:
                public, fact, current = world(rng, branch, stratum, split, seed, block)
                observations.append(public); facts.append(fact); tasks.extend(current)
    return observations, facts, tasks


def check_freeze(path):
    if path is None:
        raise ValueError('holdout requires --freeze manifest after methods freeze')
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(ROOT):
        raise ValueError('freeze manifest must exist inside exact A12 root')
    freeze = json.loads(path.read_text())
    files = freeze.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('freeze manifest requires nonempty files hash mapping')
    covered = set()
    for name, digest in files.items():
        p = Path(name)
        p = p if p.is_absolute() else ROOT / p
        if not p.resolve().is_relative_to(ROOT) or not p.is_file() or sha(p) != digest:
            raise ValueError('freeze file missing, outside target, or changed')
        covered.add(p.resolve())
    for name in ('fixed', 'library', 'free'):
        if (ROOT / 'arms' / name / 'method.py').resolve() not in covered:
            raise ValueError('all three method.py files must be frozen')
    for p in (Path(__file__).resolve(), ROOT / 'task_bundle/forward.py', ROOT / 'task_bundle/TASK.md'):
        if p not in covered:
            raise ValueError('generator and public physical contract must be frozen')
    return {'path': str(path), 'sha256': sha(path)}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def generate(split, output, freeze=None):
    if socket.gethostname() != 'liekkas' or ROOT != LOGICAL_ROOT.resolve():
        raise RuntimeError('wrong host or exact root')
    output = Path(output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError('fresh output directory must be inside exact A12 root')
    if output.exists():
        raise FileExistsError('output exists; generation never overwrites')
    frozen = check_freeze(freeze) if split == 'holdout' else None
    started = time.perf_counter()
    observations, facts, tasks = build_population(split)
    expected = 6 if split == 'dev' else 18
    if len(observations) != expected or len(facts) != expected or len(tasks) != 7 * expected:
        raise AssertionError('incomplete generation')
    if len({x['observation_id'] for x in observations}) != expected or len({t['task_id'] for t in tasks}) != len(tasks):
        raise AssertionError('duplicate identifiers')
    output.mkdir(parents=True, exist_ok=False)
    private = output / 'private'; private.mkdir()
    save(output / 'public.json', {'schema': 'a12-public-v1', 'observations': observations})
    save(private / 'truth.json', facts)
    save(private / 'tasks.json', tasks)
    if split == 'dev':
        save(output / 'dev.json', {'schema': 'a12-development-v1', 'observations': observations,
             'truths': [{'observation_id': f['observation_id'], 'truth': float(exact(f['truth'])), 'truth_exact': f['truth']} for f in facts],
             'tasks': tasks})
    info = {'split': split, 'worlds': expected, 'tasks': len(tasks), 'seeds': list(SEEDS[split]),
            'generation_wall_seconds': time.perf_counter() - started, 'utc': datetime.now(timezone.utc).isoformat(),
            'host': socket.gethostname(), 'logical_root': str(LOGICAL_ROOT), 'canonical_root': str(ROOT),
            'mechanism': MECHANISM, 'freeze': frozen, 'generator_sha256': sha(__file__),
            'forward_sha256': sha(ROOT / 'task_bundle/forward.py'), 'all_exact_forward_verified': True,
            'float_forward_max_abs_error': max(f['float_forward_max_abs_error'] for f in facts),
            'float_observation_max_abs_error': max(f['float_observation_max_abs_error'] for f in facts)}
    save(private / 'GENERATION.json', info)
    # Public generation status contains no seed, truth, strata, or task list.
    save(output / 'STATUS.json', {'status': 'COMPLETE', 'split': split, 'worlds': expected,
                                  'generation_wall_seconds': info['generation_wall_seconds'],
                                  'all_exact_forward_verified': True})
    return {'output': str(output), 'worlds': expected, 'tasks': len(tasks),
            'generation_wall_seconds': info['generation_wall_seconds']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=tuple(SEEDS), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.split, args.output or ROOT / 'data' / args.split, args.freeze)), flush=True)


if __name__ == '__main__':
    main()
