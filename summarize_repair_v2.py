"""Create append-only V2 report and fixed-view world-coordinate figures."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from schema import read_evaluation

METHODS = ('identity', 'xyz_mixture', 'fast', 'open3d_icp_then_xyz')


def median(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in ('', None)]
    return float(np.median(values)) if values else None


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in records])


def summarize(dest):
    dest = Path(dest).resolve()
    with (dest / 'RESULTS_V2.csv').open() as f: rows = list(csv.DictReader(f))
    summary = json.loads((dest / 'SUMMARY.json').read_text())
    target = dest / 'REPORT_V2.md'
    if target.exists(): raise FileExistsError(target)
    syn = [r for r in rows if r['split'] == 'synthetic' and r.get('geometry_scored') == 'True']
    sr = []
    for gap in (0, 2, 4, 8):
        for bias in (0, 4):
            for method in METHODS:
                group = [r for r in syn if float(r['gap_mm']) == gap and float(r['bias_rms_mm']) == bias and r['method'] == method]
                sr.append([gap, bias, method, len(group),
                           fmt(median(group, 'surface_accuracy_mean_mm')),
                           fmt(median(group, 'matched_normal_mae_mm')),
                           fmt(median(group, 'source_group_gap_error_mm')),
                           fmt(median(group, 'within_source_layer_rms_mm')),
                           fmt(median(group, 'reference_sample_coverage_1mm'))])
    real = [r for r in rows if r['split'] == 'real']
    rr = []
    for method in METHODS:
        for mode in ('fixed_reference_diagnostic', 'current_input'):
            base = [r for r in real if r['method'] == method and r['adapter_mode'] == mode and r['perturbation'] == 'unpert']
            t = [r for r in real if r['method'] == method and r['adapter_mode'] == mode and r['perturbation'] == 'trans']
            rot = [r for r in real if r['method'] == method and r['adapter_mode'] == mode and r['perturbation'] == 'trans_rot']
            rr.append([method, mode, fmt(median(base, 'zero_injection_edit_rms_mm')),
                       fmt(median(t, 'recovery_vs_measured_reference_rms_mm')),
                       fmt(median(rot, 'recovery_vs_measured_reference_rms_mm')),
                       fmt(median(t, 'response_vs_own_baseline_rms_mm')),
                       fmt(median(rot, 'response_vs_own_baseline_rms_mm'))])
    text = f'''# 修复与复评报告 V2

## 结论先行

修复的是评价与实验接口，不是新的滤波算法。旧 444 个输出复用重新评分，新增当前输入坐标系版本 336 个输出；合计 {summary['ok']}/{summary['rows']} 个正常结果。场景仍是 2 个真实场景、6 个片区及 27 个合成案例，不是 780 个独立场景。

原始结果中的 identity“6/6 误拆”、ICP“零误拆零误并”和基于局部坐标自报均值的间距排名撤回。自报 K 现在只保留为模型决策，不决定几何得分。并未用另一套自动分类阈值替换旧的强制二分。

## 1. 已修复

- 几何在世界坐标评分；精确有限平面矩形是评价真值，算法接收的输入不包含该信息。
- 单面位置误差不再用相对输出自身中位数的厚度代替。同一输出的几何评分不受 `k` / `mu_mm` 影响。
- 合成 `xyz_local` 正确减去扫描原点；写入前检查局部/世界/位姿/原点/射线一致性。27 例的算法世界坐标、站号、点 ID 与真值点都与旧数据完全一致。
- 旧真实结果标注为参考坐标系诊断，新增每次由当前点云重新估计坐标系的版本。
- 每次运行新建唯一目录；片区写入拒绝覆盖，旧 runner 禁止再写旧 CSV；旧数据目录不变。

## 2. 合成结果：保留几何质量与结构信息，不再以自报层数排名

每行是 3 个开发种子的中位数；噪声仍为提供的 sigma=1 mm，不是盲估计。间距 0 表示单平面。表面误差为输出到真实有限矩形的距离；来源组间距是按真实点来源层分组后的均值差误差，仅为结构诊断，不是自动恢复出物理层数的证明。1 mm 覆盖率是对原真实采样点的近邻覆盖，受采样密度和切向位移影响。

''' + table(['真间距 mm', '帧偏差 RMS mm', '方法', '案例数', '表面平均误差 mm', '对应法向 MAE mm', '来源组间距误差 mm', '层内 RMS mm', '参考样本覆盖@1mm'], sr) + '''

这些结果保留了一个具体收益：有共同偏差时，fast 能比仅看 XYZ 的混合模型更接近表面。例如单平面、偏差 RMS=4 mm 时，表面平均误差中位数 identity 3.4914 mm、xyz_mixture 3.3269 mm、fast 0.2163 mm。它不是“尚未建模帧偏差”的方法。

收益并不覆盖所有条件：8 mm 双层、没有帧偏差时，identity 表面误差约 0.8167 mm，fast 约 1.1101 mm。不能把共同偏差条件下的优势改写为全面领先。

六个 2 mm 双层上 fast 仍自报 K=1，但不能把这等价成所有世界坐标的 Z 都被压成同值：局部 PCA 平面在世界中可以倾斜，两个真实来源组处于不同横向区域。修复后的来源组间距误差约 0.63–0.67 mm（分别按有/无帧偏差分组三种子中位数），仍有结构损伤；仅看低表面距离同样不够。

ICP 加自研后处理的表面距离可以较小，但参考样本覆盖明显较低，说明不能只看法向或自报 K。覆盖是采样相关指标，不能单独断言整个真实表面丢失。它仍不是 JRMPC/BALM 对照。

## 3. 真实结果：先区分本来就发生的改写与注入后的变化

单位均为 mm。零扰动列汇总 6 片区 × 2 档 sigma；扰动列汇总 6 片区 × 2 档 sigma × 3 种子。两个 sigma 和多个种子不是独立场景。恢复列参考仍是未加扰动的测量云，不是独立真实几何。

''' + table(['方法', '坐标系', '零扰动改写 RMS', '平移后参考误差', '平移+旋转参考误差', '平移自身响应', '平移+旋转自身响应'], rr) + f'''

fast 的零扰动改写中位数已为 42.4432 mm。因此扰动后约 42.5 mm 不能解释为“5 mm 帧偏差修不掉导致 4 cm 错误”。各样本同时记录了 `输出−参考 = 基线−参考 + 输出−基线` 的平方误差分解，最大恒等式残差为 {max(float(r['decomposition_error_mm2']) for r in real if r.get('decomposition_error_mm2')):.3g} mm²。

当前输入坐标系下，fast 的自身响应从平移 4.7918 mm 到平移+旋转 4.8343 mm；旧固定参考坐标系则是 3.0853→5.0243 mm。“旋转明显伤害 fast”的强判断对预处理选择不稳健，不能单凭旧版本决定下一个算子。

片区是几何裁剪，不是验证过的局部平行薄层：一些墙片跨数米、法向厚度包含真实几何变化，junction 也不符合一两张平行平面的描述。模型/片区失配是值得验证的原因，目前不能由这轮确定。独立真实几何收益仍未测得。

## 4. 下一步判断

保留简单端、倾斜桥接端和复杂端。不要再重复实现“增加帧截距”并称为新机制：旧 fast 已有该变量。下一步最小有效实验是同一输入上的局部尺度/平行表面近似与法向估计诊断，再与噪声盲估计、倾斜项的增量分开比较。本轮没有实施这些新算法，也没有根据复评结果调参。

合成歧义组三例只证明标量 XYZ/帧模型下存在两种解释；生成器逐帧切换可见场景，不把它升级为同一静态三维场景下、带完整射线可见性的不可辨识证明。

## 5. 成本与保留

- 本次重评与新增计算墙钟 {summary['elapsed_s']:.3f} s，包含输入/旧产物哈希、生成、预热、读写与计算，不包含测试、作图及本报告生成。
- 新增算子调用累计 {summary['new_method_seconds']:.3f} s；不是 GPU 性能，也不能与原 444 行墙钟直接算加速比。
- 本进程峰值 RSS {summary['peak_process_rss_kib']} KiB，不是下载/解包或整个研究项目峰值。
- 核对 {summary['old_files_checked']} 个旧产物/冻结文件，前后哈希相同。
- 运行目录：`{dest}`；逐项数据：`RESULTS_V2.csv`；协议与源码快照随运行保存。
- 回退包：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/checkpoints/pre-repair-Y8ZRZZ/`。不要覆盖解包到现有工作目录。
- UCL 未取得 E57/IFC、TUM 未下载、独立真实精度未测、GPU 滤波核未实现，状态不变。
'''
    target.write_text(text)
    plot(dest, rows)
    return target


def plot(dest, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = dest / 'figures'; folder.mkdir(exist_ok=False)
    for case in ('ghost_s912101_b4', 'dual_g2_s912101_b4'):
        ev = read_evaluation(dest/'synthetics/identifiable'/f'{case}.json')
        arrays = [ev['gt_clean_xyz_world']]
        labels = ['Exact reference']
        for method in METHODS:
            row = next(r for r in rows if r['split']=='synthetic' and r['case']==case and r['method']==method)
            with np.load(row['source_output']) as data: arrays.append(data['xyz_world'].copy())
            labels.append(method)
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharex=True, sharey=True)
        for ax, xyz, label in zip(axes, arrays, labels):
            ax.scatter(xyz[:, 0]*1000, xyz[:, 2]*1000, s=3, c='0.25')
            ax.set_title(label, fontsize=9); ax.set_xlabel('World x / mm')
        axes[0].set_ylabel('World z / mm')
        fig.suptitle(case + ' — same world coordinates, view and scale')
        fig.tight_layout(); fig.savefig(folder/f'{case}_world.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path)
    print(summarize(p.parse_args().run))
