"""Frozen independent mechanism demonstration; see PROTOCOL.json and THEORY.md."""
import hashlib
import json
import os
import time
from pathlib import Path

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
from PIL import Image, ImageDraw
from kernel import gain_bounds, strictly_improves

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'PROTOCOL.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_scene(seed, scenario, scene_spec, matching_spec):
    """No repair candidates supplied: target/opaque planes/textures fixed first."""
    rng = np.random.default_rng(seed)
    radius = matching_spec['patch_radius']
    p = rng.normal(size=(2 * radius + 1) * 3)
    p -= p.mean()
    v = rng.normal(size=p.size)
    v -= v.mean()
    v -= np.dot(v, p) / np.dot(p, p) * p
    v *= np.linalg.norm(p) / np.linalg.norm(v)
    scale = 0.38 / max(np.max(np.abs(p)), np.max(np.abs(p + .6*v)), np.max(np.abs(p - .6*v)))
    motifs = {key: (.5 + scale * value).reshape(-1, 3)
              for key, value in [('p', p), ('plus', p+.6*v), ('minus', p-.6*v)]}
    rear_x = np.arange(-300, 301) / 30.0
    front_x = np.arange(-600, 601) / 60.0
    rear = rng.uniform(.15, .85, size=(rear_x.size, 3))
    front = rng.uniform(.15, .85, size=(front_x.size, 3))
    if scenario == 'positive':
        assignments = [(-.3, 'p'), (.3, 'plus'), (-1., 'minus')]
    elif scenario == 'repeated_texture':
        assignments = [(-.3, 'p'), (.3, 'p'), (-1., 'p')]
    elif scenario == 'occlusion_wrong_layer':
        assignments = [(-.3, 'p'), (.1, 'p')]
        center = int(np.argmin(np.abs(front_x + 1/30)))
        front[center-radius:center+radius+1] = motifs['p']
    else:
        assignments = [(-.3, 'p')]
    for x, motif in assignments:
        center = int(np.argmin(np.abs(rear_x - x)))
        rear[center-radius:center+radius+1] = motifs[motif]
    extent = scene_spec['front_plane']['occlusion_x_extent' if scenario == 'occlusion_wrong_layer' else 'ordinary_x_extent']
    return {'rear_x': rear_x, 'rear_rgb': rear, 'front_x': front_x,
            'front_rgb': front, 'front_extent': extent,
            'target': dict(scene_spec['true_target'])}


def render(scene, scene_spec, seed):
    """Pinhole rays intersect finite opaque planes; nearest positive hit wins."""
    width = scene_spec['image_width']
    q = (np.arange(width) - (width-1)/2) / scene_spec['focal_pixels']
    rng = np.random.default_rng(seed + 10000)
    photos, physical_x, layers = [], [], []
    for c in scene_spec['camera_centers_x']:
        x_rear, x_front = c + q*6., c + q*3.
        hit_front = (x_front >= scene['front_extent'][0]) & (x_front <= scene['front_extent'][1])
        hit_rear = (x_rear >= -10.) & (x_rear <= 10.)
        image = np.zeros((width, 3))
        layer = np.full(width, -1, dtype=int)
        x_hit = np.full(width, np.nan)
        for visible, name, coords, layer_id in [(hit_rear & ~hit_front, 'rear', x_rear, 1), (hit_front, 'front', x_front, 0)]:
            for channel in range(3):
                image[visible, channel] = np.interp(coords[visible], scene[name+'_x'], scene[name+'_rgb'][:, channel])
            layer[visible], x_hit[visible] = layer_id, coords[visible]
        image = np.clip(image + rng.normal(0, scene_spec['sensor_noise_std'], image.shape), 0, 1)
        photos.append(image)
        physical_x.append(x_hit)
        layers.append(layer)
    return np.array(photos), np.array(physical_x), np.array(layers)


def features(photo, radius):
    centers = np.arange(radius, len(photo)-radius)
    patches = np.array([photo[i-radius:i+radius+1].ravel() for i in centers])
    patches -= patches.mean(1, keepdims=True)
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    return centers, np.divide(patches, norms, out=np.zeros_like(patches), where=norms > 1e-12)


def observe(photos, reference_pixel, matching):
    """Only measured images and reference pixel; no truth or candidates."""
    radius, threshold = matching['patch_radius'], matching['ncc_min']
    centers, reference = features(photos[0], radius)
    _, source1 = features(photos[1], radius)
    _, source2 = features(photos[2], radius)
    reference_feature = reference[reference_pixel-radius]
    ncc1, ncc2 = source1 @ reference_feature, source2 @ reference_feature
    matches1, matches2 = np.flatnonzero(ncc1 >= threshold), np.flatnonzero(ncc2 >= threshold)
    # Measured all-image source-to-source search, including every repeated match.
    pair_ncc = source1 @ source2.T
    pairs = np.argwhere(pair_ncc >= threshold)
    return {'centers': centers, 'matches1': matches1, 'matches2': matches2,
            'ncc1': ncc1, 'ncc2': ncc2, 'pair_ncc': pair_ncc, 'all_source_pairs': pairs}


def match_interval(pixel, reference_pixel, alpha, tolerance, domain):
    lower, upper = sorted(((pixel-reference_pixel-tolerance)/alpha,
                           (pixel-reference_pixel+tolerance)/alpha))
    lower, upper = max(lower, 1/domain[1]), min(upper, 1/domain[0])
    return None if lower > upper else (1/upper, 1/lower)


def union(intervals):
    output = []
    for lo, hi in sorted(intervals):
        if output and lo <= output[-1][1] + 1e-12:
            output[-1][1] = max(hi, output[-1][1])
        else:
            output.append([float(lo), float(hi)])
    return output


def feasible(observations, geometry, matching, add_source_source):
    """No candidate depths or truth supplied to correspondence/set construction."""
    centers = observations['centers']
    intervals, admitted_pairs = [], []
    for i in observations['matches1']:
        first = match_interval(centers[i], geometry['reference_pixel'], -geometry['focal']*geometry['centers'][1], matching['pixel_tolerance'], matching['depth_domain'])
        if first is None:
            continue
        for j in observations['matches2']:
            if add_source_source and observations['pair_ncc'][i, j] < matching['ncc_min']:
                continue
            second = match_interval(centers[j], geometry['reference_pixel'], -geometry['focal']*geometry['centers'][2], matching['pixel_tolerance'], matching['depth_domain'])
            if second is None:
                continue
            lo, hi = max(first[0], second[0]), min(first[1], second[1])
            if lo <= hi:
                intervals.append([lo, hi])
                admitted_pairs.append({'u1': int(centers[i]), 'u2': int(centers[j]), 'interval': [lo, hi], 'ncc01': float(observations['ncc1'][i]), 'ncc02': float(observations['ncc2'][j]), 'ncc12': float(observations['pair_ncc'][i, j])})
    return union(intervals), admitted_pairs


def run():
    started = time.time()
    expected_hash = (ROOT/'PROTOCOL.sha256').read_text().split()[0]
    assert digest(ROOT/'PROTOCOL.json') == expected_hash, 'frozen protocol changed'
    output_dir = ROOT/'results'
    output_dir.mkdir(exist_ok=True)
    all_results = []
    for scenario, specification in PROTOCOL['scenarios'].items():
        for seed in PROTOCOL['seeds']:
            label = f'{scenario}_{seed}'
            case_dir = output_dir/label
            case_dir.mkdir(exist_ok=True)
            # Scene and target identity exist and are serialized before rendering.
            scene = make_scene(seed, scenario, PROTOCOL['scene'], PROTOCOL['matching'])
            np.savez_compressed(case_dir/'world_textures.npz', **{k:v for k,v in scene.items() if isinstance(v, np.ndarray)})
            scene_record = {'seed': seed, 'scenario': scenario, 'true_target': scene['target'], 'front_extent': scene['front_extent'], 'camera_centers': PROTOCOL['scene']['camera_centers_x'], 'texture_hash': digest(case_dir/'world_textures.npz')}
            (case_dir/'scene_before_render.json').write_text(json.dumps(scene_record, indent=2)+'\n')
            photos, physical_x, layers = render(scene, PROTOCOL['scene'], seed)
            np.savez_compressed(case_dir/'rendered.npz', photos=photos, physical_x=physical_x, layers=layers)
            for index, photo in enumerate(photos):
                Image.fromarray(np.repeat((photo[None, :, :]*255).astype(np.uint8), 24, axis=0)).save(case_dir/f'view{index}.png')
            observation = observe(photos, PROTOCOL['scene']['reference_pixel'], PROTOCOL['matching'])
            np.savez_compressed(case_dir/'observations.npz', **observation)
            geometry = {'reference_pixel': PROTOCOL['scene']['reference_pixel'], 'focal': PROTOCOL['scene']['focal_pixels'], 'centers': np.array(PROTOCOL['scene']['camera_centers_x'])*specification['assumed_camera_scale']}
            # Matching is sealed before candidate gain and truth diagnostics.
            observation_hash = digest(case_dir/'observations.npz')
            row = {'scenario': scenario, 'seed': seed, 'image_hash': digest(case_dir/'rendered.npz'), 'observation_hash': observation_hash,
                   'reference_source_matches': [len(observation['matches1']), len(observation['matches2'])], 'all_source_source_matches': len(observation['all_source_pairs']), 'a': specification['a'], 'b': specification['b'], 'arms': {}}
            for arm, use_pair in [('star', False), ('common_track', True)]:
                intervals, pairs = feasible(observation, geometry, PROTOCOL['matching'], use_pair)
                bounds = gain_bounds(specification['a'], specification['b'], intervals)
                accepted = strictly_improves(bounds)
                z_true = scene['target']['z']
                actual_gain = (specification['a']-z_true)**2 - (specification['b']-z_true)**2
                for pair in pairs:
                    pair['posthoc_source_layers'] = [int(layers[1, pair['u1']]), int(layers[2, pair['u2']])]
                    pair['posthoc_world_x'] = [float(physical_x[1, pair['u1']]), float(physical_x[2, pair['u2']])]
                    pair['posthoc_same_true_target'] = all(abs(x-scene['target']['x']) < 1e-9 for x in pair['posthoc_world_x']) and pair['posthoc_source_layers'] == [1, 1]
                row['arms'][arm] = {'intervals': intervals, 'components': len(intervals), 'gain_bounds': bounds, 'accept_b': accepted, 'true_target_covered': any(lo <= z_true <= hi for lo,hi in intervals), 'actual_gain_if_b': actual_gain, 'actual_selected_gain': actual_gain if accepted else 0., 'pairs': pairs}
            projected_true = [int(round((scene['target']['x']-c)/scene['target']['z']*PROTOCOL['scene']['focal_pixels']+(PROTOCOL['scene']['image_width']-1)/2)) for c in PROTOCOL['scene']['camera_centers_x']]
            row['posthoc_target_visible'] = [bool(layers[i, u] == 1 and abs(physical_x[i,u]-scene['target']['x']) < 1e-9) for i,u in enumerate(projected_true)]
            (case_dir/'result.json').write_text(json.dumps(row, indent=2)+'\n')
            all_results.append(row)
    summary = {'protocol_hash': expected_hash, 'code_hashes': {name:digest(ROOT/name) for name in ['mechanism.py', 'kernel.py', 'test_kernel.py']}, 'scope': PROTOCOL['scope'], 'runtime_seconds': time.time()-started, 'cases': all_results}
    (output_dir/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['scenario,seed,arm,components,covered,accept_b,gain_low,gain_high,actual_selected_gain']
    for row in all_results:
        for arm, result in row['arms'].items():
            bound = result['gain_bounds'] or [None, None]
            lines.append(','.join(map(str, [row['scenario'], row['seed'], arm, result['components'], result['true_target_covered'], result['accept_b'], *bound, result['actual_selected_gain']])))
    (output_dir/'summary.csv').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(f'Runtime: {summary["runtime_seconds"]:.3f} seconds')


if __name__ == '__main__':
    run()
