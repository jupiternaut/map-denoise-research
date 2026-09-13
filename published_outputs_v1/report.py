import json,csv,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v1')
OUT=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v1')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((DATA/'DOWNLOAD_MANIFEST.json').read_text())
    unchanged=[]
    for item in manifest['files']:
        match=sha(item['path'])==item['sha256'];assert match
        unchanged.append(dict(path=item['path'],sha256=item['sha256'],unchanged=match))
    audits=[json.loads((OUT/s/'audit.json').read_text()) for s in ('room','scan24','Courthouse')]
    rows=sum([json.loads((OUT/s/'tests.json').read_text()) for s in ('room','scan24','Courthouse')],[])
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['# 已公开重建结果下载与首轮实测报告', '',
        '## 结论','',
        '三场景下载、完整文件检查、54 次局部对照和 5 项单元测试完成。不是重新训练，也不是自建数据集。',
        '现有帧偏差校正器不能直接应用于这些缺少点—帧对应的成品；本轮复用的是空间混合求解器的几何子模块，去除了帧校正，不是旧方法真实精度复现。',
        '局部双层模型在部分真实复杂片区中造成明显几何改写与三角形法向翻转。当前不能认定精度改善，也不宜直接全图应用。',
        '', '## 完整输入及实际处理规模','',
        '|场景|文件 MB（十进制）|顶点/高斯数|三角形数|局部采样数上限|处理及审计墙钟秒|峰值 RSS MiB|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in audits:
        name=Path(a['source_path']).name if 'opacity_baseline' not in a else 'room'
        lines.append(f"|{name}|{a['bytes']/1e6:.2f}|{a['vertices']:,}|{a.get('faces',0):,}|3072 ({3072/a['vertices']*100:.4f}%)|{a['elapsed_seconds']:.2f}|{a['process_peak_rss_KiB']/1024:.1f}|")
    lines+=['','局部采样是三块各 1024 点，片区可能重叠；上表不是唯一点覆盖率。完整场景均做字段/有限值/索引检查，只有 room 做了一个全场景剪枝基线。所有计算为 CPU；未训练模型、未运行 CUDA。',
        '峰值 RSS 是各场景独立 Python 进程的高水位，不是 GPU 显存。下载墙钟约 '+f"{manifest['seconds']:.2f} 秒。",'',
        '## 实验设置','',
        '三个确定性坐标分位锚点选择邻域，未按滤波效果挑片区。每片区比较 identity、单平面、空间混合三种残差尺度（0.5/1/2 倍输入中位近邻间距）、官方 PyMeshLab APSS(scale=4)。',
        '混合求解器复用 exploration_v11/spatial_filter.py 的 solve，36 次迭代。残差尺度不是盲估计出的传感器噪声；场景单位未核定，故不汇报毫米精度。',
        '', '## 全部网格片区的形变诊断','',
        '|方法|6 个片区累计法向翻转数|P95 位移/点间距（各片区中位数）|',
        '|---|---:|---:|']
    for method in ('identity','plane','spatial_s05','spatial_s1','spatial_s2','apss4'):
        rr=[r for r in rows if r['scene']!='room' and r['method']==method and r['status']=='ok']
        lines.append(f"|{method}|{sum(r['orientation_reversals'] for r in rr)}|{np.median([r['displacement_p95_spacing'] for r in rr]):.3f}|")
    lines+=['','法向翻转是新旧三角形法向内积非正的诊断，包括编辑片区边界外连三角形；不是全局自交测试，也不是精度指标。累计数按实验片区计数，可能重复包含同一三角形。identity 的零改动不代表它精度最优。',
        'scan24 patch1：单平面 36 次翻转，spatial_s1 11 次，APSS 3 次。双层比单平面少损伤，但这不足以胜过对口方法，更不代表恢复了正确表面。',
        '', '## room 全场景导出','']
    b=audits[0]['opacity_baseline']
    lines += [f"使用 sigmoid(opacity)<0.01 的普通剪枝基线：原始 {audits[0]['vertices']:,} 个高斯，删除 {b['removed']:,}（{100*b['fraction_removed']:.2f}%），保留 {b['retained']:,}。",
        '导出后重新读取，保留基元的全部 62 项字段逐项与原模型相等，包括颜色球谐、尺度、旋转和不透明度。没有依据长细比删除薄高斯。',
        '**这不是本文新方法，也不是已经验证的飞点清理。低 opacity 高斯仍可能具有累积渲染贡献；本轮没有相机视角渲染对照，不能据此推荐生产使用。**',
        f"[完整候选 Gaussian PLY]({b['path']})",'',
        '## 已完成 / 尚未完成','',
        '- 已完成：三个现成成果下载、ZIP CRC、SHA256、全输入读取检查、54/54 局部运行、5/5 单元测试、9 张几何剖面对照、完整 room 剪枝候选。',
        '- 未完成：原/处理后相机视角渲染、独立几何参考配准与精度评价、整场景网格滤波、帧来源恢复。',
        '- 下一步：先对 room 原模型与剪枝候选做相同相机渲染；几何方向使用 scan24 的官方参考与正确坐标映射量误差。不能将裸高斯中心直接当作真实表面，也不能把任意复杂邻域强制拟合为两个平行面。',
        '- 上述后续任务未在本轮偷偷下载新数据或安装重建模型。', '',
        '## 可复现与输出','',
        f'[下载清单与源 URL]({DATA}/DOWNLOAD_MANIFEST.json)',
        f'[全部实验数据表]({OUT}/RESULTS.csv)',
        f'[预先记录的协议]({HERE}/PROTOCOL.md)',
        f'[scan24 patch1 对比]({OUT}/scan24/patch1_comparison.png)',
        f'[Courthouse patch1 对比]({OUT}/Courthouse/patch1_comparison.png)',
        '', '运行命令（仅写独立本轮输出目录，不改旧检查点）：','', '```bash',
        f'python {HERE}/download.py',
        f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python {HERE}/test_outputs.py',
        f'OPENBLAS_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python {HERE}/unit_tests.py',
        f'/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python {HERE}/report.py','```']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'FINAL_AUDIT.json').write_text(json.dumps(dict(originals=unchanged,runs=len(rows),ok=sum(r['status']=='ok' for r in rows),
        source_hashes={str(p):sha(p) for p in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE.parent/'exploration_v11/spatial_filter.py',HERE.parent/'exploration_v12/external.py']}),indent=2))
    print('\n'.join(lines[:31]));print('REPORT',HERE/'REPORT.md')

if __name__=='__main__':main()
