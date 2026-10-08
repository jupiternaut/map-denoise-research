"""Pinhole rendering, shared correspondence scoring, sealed evaluation.

All authored outputs are confined to this directory. World truths enter only
the renderer and evaluation; support.py does not receive fixture truth.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
XY = np.array([64., 64.])
GRID = np.arange(300., 1001., 2.)
CANDIDATES = np.array([450., 540., 600., 660., 900.])
INCUMBENT = 900.


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def cameras(bias=1.):
    k = np.array([[160., 0, 64.], [0, 160., 64.], [0, 0, 1.]])
    out = [dict(K=k, R=np.eye(3), C=np.zeros(3))]
    for sign in (-1, 1):
        out.append(dict(K=k, R=rotation_y(sign*.07) @ rotation_z(sign*.11),
                        C=np.array([sign*60.*bias, 0., 0.])))
    return out


def serialize_cam(cam):
    return {k: v.tolist() for k, v in cam.items()}


def deserialize_cam(cam):
    return {k: np.asarray(v, float) for k, v in cam.items()}


def ray(cam, uv):
    uv = np.asarray(uv)
    return np.concatenate((uv, np.ones(uv.shape[:-1]+(1,))), -1) @ np.linalg.inv(cam['K']).T @ cam['R']


def project(cam, points):
    a = (points-cam['C']) @ cam['R'].T @ cam['K'].T
    return a[..., :2]/a[..., 2:]


def make_scene(name, seed):
    phases = np.random.default_rng(seed).uniform(-np.pi, np.pi, 5).tolist()
    # Plane equation n dot X = d. Bounds are physical world-space predicates.
    scene = dict(name=name, seed=seed, phases=phases, target_id=0, planes=[])
    target = dict(n=[0., 0., 1.], d=600., bound="all", texture="good", id=0)
    if name.startswith('ring'):
        half_pixels = 1.65 if name.startswith('ring9') else 2.65
        target.update(bound="square", half_width=half_pixels*600/160, texture="core")
        bg = dict(n=[0., 0., 1.], d=900., bound="all", id=1,
                  texture="flat" if name.endswith('flat') else "background")
        scene['planes'] = [target, bg]
    elif name == 'source_occlusion':
        near = dict(n=[0., 0., 1.], d=400., bound="right", edge_x=8., texture="core", id=1)
        scene['planes'] = [target, near]
    else:
        if name == 'weak_texture':
            target['texture'] = 'weak'
        elif name == 'periodic_repeats':
            target['texture'] = 'periodic'
        elif name == 'slanted_plane':
            target['n'] = [-2.2, -1.1, 1.]
        scene['planes'] = [target]
    return scene


def texture(kind, xyz, phases):
    x, y = xyz[..., 0], xyz[..., 1]
    p = phases
    if kind == 'flat':
        return np.full_like(x, 150.)
    if kind == 'core':
        return 80+8*np.sin(2*np.pi*x/20+p[0])+7*np.sin(2*np.pi*y/27+p[1])+3*np.sin(2*np.pi*(x+y)/34+p[2])
    if kind == 'background':
        return 150+20*np.sin(2*np.pi*x/25+p[2])+20*np.sin(2*np.pi*y/31+p[3])
    if kind == 'periodic':
        return 120+30*np.sin(2*np.pi*x/20+p[0])+20*np.sin(2*np.pi*y/20+p[1])
    wave = (22*np.sin(2*np.pi*x/24+p[0])+18*np.sin(2*np.pi*y/31+p[1])
            +14*np.sin(2*np.pi*(x+.7*y)/19+p[2]))
    if kind == 'weak':
        wave *= .2/54
    return 120+wave


def intersect(scene, cam, uv):
    """Render an arbitrary continuous image coordinate, returning physical hit."""
    directions = ray(cam, uv)
    shape = directions.shape[:-1]
    depths = np.full(shape, np.inf)
    owners = np.full(shape, -1, int)
    points = np.full(shape+(3,), np.nan)
    values = np.zeros(shape)
    for plane in scene['planes']:
        normal = np.asarray(plane['n'])
        denominator = directions @ normal
        distance = np.divide(plane['d']-cam['C'] @ normal, denominator,
                             out=np.full(shape, np.inf), where=np.abs(denominator)>1e-10)
        candidate = cam['C']+distance[..., None]*directions
        valid = (distance > 0) & (distance < depths) & np.isfinite(distance)
        if plane['bound'] == 'square':
            valid &= (np.abs(candidate[..., 0]) <= plane['half_width']) & (np.abs(candidate[..., 1]) <= plane['half_width'])
        elif plane['bound'] == 'right':
            valid &= candidate[..., 0] >= plane['edge_x']
        depths[valid] = distance[valid]
        owners[valid] = plane['id']
        points[valid] = candidate[valid]
        values[valid] = texture(plane['texture'], candidate[valid], scene['phases'])
    return dict(value=values, depth=depths, owner=owners, point=points)


def render(scene, cam):
    yy, xx = np.mgrid[0:128, 0:128]
    uv = np.stack((xx, yy), -1).astype(float)
    values = np.zeros((128, 128))
    for oy in (-1/3, 0, 1/3):
        for ox in (-1/3, 0, 1/3):
            values += intersect(scene, cam, uv+np.array([ox, oy]))['value']/9
    truth = intersect(scene, cam, uv)
    return values, truth


def render_all():
    (ROOT/'fixtures').mkdir(exist_ok=True)
    manifest = []
    for seed in PROTOCOL['seeds']:
        for name in PROTOCOL['cases']:
            scene = make_scene(name, seed)
            actual = cameras()
            supplied = cameras(1.15 if name == 'common_camera_bias' else 1.)
            images, depths, owners = [], [], []
            for cam in actual:
                image, truth = render(scene, cam)
                images.append(image)
                depths.append(truth['depth'])
                owners.append(truth['owner'])
            fixture_id = f'{name}_seed{seed}'
            npz = ROOT/'fixtures'/f'{fixture_id}.npz'
            np.savez_compressed(npz, images=np.asarray(images), depths=np.asarray(depths), owners=np.asarray(owners))
            # Images are displayed only; scoring consumes floating grayscale arrays.
            strip = np.concatenate(images, axis=1)
            Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8)).save(ROOT/'fixtures'/f'{fixture_id}.png')
            manifest.append(dict(id=fixture_id, name=name, seed=seed, scene=scene,
                                 actual_cameras=[serialize_cam(c) for c in actual],
                                 score_cameras=[serialize_cam(c) for c in supplied],
                                 arrays=npz.name, sha256=digest(npz)))
    dump(ROOT/'FIXTURES.json', manifest)
    dump(ROOT/'RENDER_LOCK.json', dict(protocol_sha256=digest(ROOT/'PROTOCOL.json'),
                                     renderer_sha256=digest(Path(__file__)),
                                     fixture_manifest_sha256=digest(ROOT/'FIXTURES.json'), count=len(manifest)))
    print(json.dumps(dict(stage='rendered', fixtures=len(manifest))))


def finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def score_all():
    sys.path.insert(0, str(ROOT.parent))
    from support import score_support, support_intervals, NCC_MIN
    # Construct observations without reading world planes, rendered depths or owners.
    observations = []
    (ROOT/'curves').mkdir(exist_ok=True)
    for meta in json.loads((ROOT/'FIXTURES.json').read_text()):
        with np.load(ROOT/'fixtures'/meta['arrays']) as data:
            images = data['images']
        cams = [deserialize_cam(c) for c in meta['score_cameras']]
        for warp in ('translation', 'plane'):
            for kind in ('full9', 'center3', 'connected9'):
                results = [score_support(images[0], images[s], cams[0], cams[s], XY, GRID, warp, kind) for s in (1, 2)]
                scores = np.stack([r['scores'] for r in results])
                accepted = np.all(np.isfinite(scores) & (scores >= NCC_MIN), axis=0)
                intervals = support_intervals(GRID, accepted, padding=1.)
                arm = f'{warp}_{kind}'
                curve_file = ROOT/'curves'/f"{meta['id']}_{arm}.npz"
                np.savez_compressed(curve_file, grid=GRID, scores=scores, accepted=accepted, mask=results[0]['mask'])
                observations.append(dict(fixture_id=meta['id'], arm=arm, intervals=intervals,
                                         accepted_grid_count=int(accepted.sum()), mask_count=results[0]['mask_count'],
                                         anchor_std=results[0]['anchor_std'],
                                         valid_counts=[r['valid_count'] for r in results],
                                         curve_file=curve_file.name, curve_sha256=digest(curve_file)))
    dump(ROOT/'OBSERVATIONS.json', observations)
    dump(ROOT/'OBSERVATIONS_LOCK.json', dict(observations_sha256=digest(ROOT/'OBSERVATIONS.json'),
                                           shared_support_sha256=digest(ROOT.parent/'support.py'),
                                           renderer_sha256=digest(Path(__file__)),
                                           protocol_sha256=digest(ROOT/'PROTOCOL.json')))
    print(json.dumps(dict(stage='scored_and_sealed', observations=len(observations))))


def select(intervals):
    mass = sum(hi-lo for lo, hi in intervals)
    if mass <= 0:
        return dict(support_mean=None, selected_depth=INCUMBENT, chosen_index=4, point_gain=0.)
    mean = sum((hi-lo)*(hi+lo)/2 for lo, hi in intervals)/mass
    idx = int(np.argmin(np.abs(CANDIDATES-mean)))
    gain = abs(INCUMBENT-mean)-abs(CANDIDATES[idx]-mean)
    if gain <= 0:
        idx = 4
    return dict(support_mean=mean, selected_depth=float(CANDIDATES[idx]), chosen_index=idx, point_gain=float(max(0, gain)))


def evaluate_all():
    observations = json.loads((ROOT/'OBSERVATIONS.json').read_text())
    observation_lock = json.loads((ROOT/'OBSERVATIONS_LOCK.json').read_text())
    assert digest(ROOT/'OBSERVATIONS.json') == observation_lock['observations_sha256']
    # Candidate access occurs after sealing the image observations. Selection is
    # itself sealed before the world-space truth below is loaded.
    selections = [dict(fixture_id=o['fixture_id'], arm=o['arm'], **select(o['intervals'])) for o in observations]
    dump(ROOT/'SELECTIONS.json', selections)
    dump(ROOT/'SELECTIONS_LOCK.json', dict(selections_sha256=digest(ROOT/'SELECTIONS.json'),
                                         observations_sha256=observation_lock['observations_sha256']))
    fixture_meta = {f['id']: f for f in json.loads((ROOT/'FIXTURES.json').read_text())}
    rows = []
    for obs, chosen in zip(observations, selections):
        meta = fixture_meta[obs['fixture_id']]
        scene = meta['scene']
        actual = [deserialize_cam(c) for c in meta['actual_cameras']]
        with np.load(ROOT/'curves'/obs['curve_file']) as data:
            scores, mask = data['scores'], data['mask']
        yy, xx = np.mgrid[-4:5, -4:5]
        uv = XY+np.stack((xx, yy), -1)[mask]
        ref_truth = intersect(scene, actual[0], uv)
        center = intersect(scene, actual[0], XY)
        true_depth = float(center['depth'])
        owned = ref_truth['owner'] == scene['target_id']
        source_visible, center_visible = [], []
        for cam in actual[1:]:
            actual_hit = intersect(scene, cam, project(cam, ref_truth['point']))
            visibility = np.linalg.norm(actual_hit['point']-ref_truth['point'], axis=-1) < 1e-5
            source_visible.append(float(np.mean(visibility & owned)))
            center_hit = intersect(scene, cam, project(cam, center['point']))
            center_visible.append(bool(np.linalg.norm(center_hit['point']-center['point']) < 1e-5))
        assumed_points = actual[0]['C']+true_depth*ray(actual[0], uv)
        warp_errors = np.linalg.norm(assumed_points-ref_truth['point'], axis=-1)
        length = sum(hi-lo for lo, hi in obs['intervals'])
        near_length = sum(max(0., min(hi, true_depth+10)-max(lo, true_depth-10)) for lo, hi in obs['intervals'])
        true_idx = int(np.argmin(abs(GRID-true_depth)))
        rows.append(dict(**obs, **{k:v for k,v in chosen.items() if k not in obs}, name=meta['name'], seed=meta['seed'],
                         true_depth=true_depth, true_in_support=any(lo<=true_depth<=hi for lo,hi in obs['intervals']),
                         true_depth_ncc=finite_list(scores[:, true_idx]), support_length=length,
                         off_target_support_length=length-near_length,
                         off_target_fraction=(length-near_length)/length if length else None,
                         selected_absolute_error=abs(chosen['selected_depth']-true_depth),
                         target_mask_fraction=float(owned.mean()), source_visible_target_fraction=source_visible,
                         source_center_visible=center_visible,
                         mean_frontoparallel_world_error=float(np.mean(warp_errors)),
                         max_frontoparallel_world_error=float(np.max(warp_errors))))
    dump(ROOT/'EVALUATION.json', rows)
    pairing = []
    for seed in PROTOCOL['seeds']:
        for size in ('ring9', 'ring25'):
            with np.load(ROOT/'fixtures'/f'{size}_flat_seed{seed}.npz') as f, np.load(ROOT/'fixtures'/f'{size}_textured_seed{seed}.npz') as t:
                pairing.append(dict(seed=seed, pair=size,
                                    reference_center3_max_image_difference=float(np.max(abs(f['images'][0,63:66,63:66]-t['images'][0,63:66,63:66]))),
                                    reference_center3_owners_identical=bool(np.array_equal(f['owners'][0,63:66,63:66],t['owners'][0,63:66,63:66])),
                                    reference_center3_depths_identical=bool(np.array_equal(f['depths'][0,63:66,63:66],t['depths'][0,63:66,63:66]))))
    dump(ROOT/'PAIRING_CHECKS.json', pairing)
    summary = []
    for name in PROTOCOL['cases']:
        for arm in PROTOCOL['arms']:
            rr = [r for r in rows if r['name']==name and r['arm']==arm]
            summary.append(dict(name=name, arm=arm, n=len(rr), true_in_support=sum(r['true_in_support'] for r in rr),
                                empty=sum(not r['intervals'] for r in rr),
                                selected_exact=sum(r['selected_absolute_error']==0 for r in rr),
                                mean_selected_error=float(np.mean([r['selected_absolute_error'] for r in rr])),
                                mean_off_target_length=float(np.mean([r['off_target_support_length'] for r in rr])),
                                mean_mask_count=float(np.mean([r['mask_count'] for r in rr])),
                                mean_target_mask_fraction=float(np.mean([r['target_mask_fraction'] for r in rr]))))
    dump(ROOT/'SUMMARY.json', summary)
    write_report(rows, summary, pairing)
    print(json.dumps(dict(stage='evaluated', rows=len(rows), paired_center3_identical=all(p['reference_center3_max_image_difference']==0 for p in pairing))))


def write_report(rows, summary, pairing):
    lines = ['# 受控成像机制实验', '',
             '## 目标与方法', '',
             '使用解析针孔相机与不透明平面的最近正向交点渲染灰度图。世界纹理由连续正弦函数定义，像素采用3×3面积采样；没有从真值深度手工构造代价曲线。三个固定随机种子、十种场景、六种方法，共180次双源支持估计。单位为毫米。', '',
             '六个方法共用父目录support.py。每个深度要求两个源视图同时通过NCC≥0.6，保留全部连通深度区间，区间扩展半网格步长+1mm。固定候选为450/540/600/660/900，原输出900；按支持区间长度均匀分布的均值选择最近候选，空支持保留原输出。真值只在渲染和观测/选择封存后的评估阶段使用。', '',
             '环形对照保持中心平面、中心纹理和所有相机不变，仅将900mm背景由常量替换为纹理。其余负对照分别破坏纹理强度、纹理唯一性、源可见性、正平面假设和共同相机标定。', '',
             '## 结果', '',
             '表内“含真值/精确选择/空支持”均为三个种子的计数；非目标长度是离600mm超过10mm的支持总长度，按种子平均。', '',
             '| 场景 | 方法 | 含真值 | 精确选择 | 空支持 | 平均选择误差 mm | 平均非目标长度 mm | 目标表面掩码比例 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['name']} | {r['arm']} | {r['true_in_support']}/3 | {r['selected_exact']}/3 | {r['empty']}/3 | {r['mean_selected_error']:.1f} | {r['mean_off_target_length']:.1f} | {r['mean_target_mask_fraction']:.3f} |")
    lines += ['', '## 配对有效性与边界', '',
              f"全部{len(pairing)}组配对的参考中心3×3像素最大差为{max(p['reference_center3_max_image_difference'] for p in pairing):.6g}，真值深度与表面标签逐像素相同。源图因像素积分和边界遮挡仍可能混合不同表面；这些变化本身是观测机制的一部分。", '',
              'EVALUATION.json逐例记录参考掩码真实目标表面比例、每个源视图可见的目标掩码比例、中心是否被遮挡、正平面假设的世界坐标误差、真值处双源NCC、全部支持区间和选择结果。这些诊断不进入掩码、评分或选择。', '',
              '## 解释与下一步', '',
              '这些是公开设计的机制场景，不是随机自然场景样本。连接灰度掩码只表示可观测的所有权假设；即使掩码全部属于目标，重复纹理与共同相机误差仍可产生错误而一致的支持。正平面投影只在参考支撑实际近似该平面时有物理依据。固定NCC阈值不保证不同掩码大小拥有同等误匹配率。', '',
              '应将本机制证据与真实回放并列解释，保留失败场景。不能据此宣称目标身份得到证明、支持具有校准覆盖率，或真实最近激光误差必然改善。下一步若继续，应预注册独立的几何/可见性条件及新场景，不根据这些结果反复调整阈值。', '',
              '## 文件与封存', '',
              '- PROTOCOL.json：评分前协议。',
              '- FIXTURES.json、fixtures/*.npz/png：全部世界定义、相机、渲染浮点图、深度/表面标签和预览。',
              '- RENDER_LOCK.json：渲染源码、协议和场景清单哈希。',
              '- OBSERVATIONS.json、curves/*.npz、OBSERVATIONS_LOCK.json：候选/真值无关的支持与原始曲线，及共享评分源码哈希。',
              '- SELECTIONS.json、SELECTIONS_LOCK.json：评估前封存的候选选择。',
              '- EVALUATION.json、SUMMARY.json、PAIRING_CHECKS.json：原始逐例评估、聚合和配对检查。', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['render', 'score', 'evaluate'])
    args = parser.parse_args()
    {'render': render_all, 'score': score_all, 'evaluate': evaluate_all}[args.stage]()
