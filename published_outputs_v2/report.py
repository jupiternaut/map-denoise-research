import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v2')
REF=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    r=json.loads((OUT/'room/RENDER_RESULTS.json').read_text());g=json.loads((OUT/'scan24/GEOMETRY_RESULTS.json').read_text())
    lines=['# 第二轮：同相机渲染、独立几何参考和 OSM 适用性','',
        '## 主要结果','',
        '1. room：普通 opacity<0.01 剪枝减少 21.04% 高斯，在四个固定同相机视角、1024×682 分辨率下，画面变化很小。这是压缩/渲染保真信号，不是新几何去噪方法的胜利。',
        '2. scan24：官方参考点云已接入；沿用上一轮三个片区及全部候选。使用公开 DTU 相机归一化矩阵，不做 GT-ICP 或拟合尺度，得到局部距离诊断。空间混合三个设置均未改善汇总点到参考距离。',
        '3. OSM：这些结果尚不是可直接导入 OSM 的地理要素。OSM 没有统一的毫米级点云验收阈值。', '',
        '## room 同相机真实 Gaussian 渲染','',
        '使用官方 gsplat 1.5.3，SH degree=3，黑背景，classic 光栅化，radius_clip=0，固定原包 cameras.json 的位置、旋转和内参。无重新训练。',
        '四个视角在渲染前固定为 camera index 0、103、206、310。原模型和上一轮已保存的剪枝模型分别加载，所有显示与光栅化设置一致。',
        '', '|相机|平均 RGB 绝对差（0–1）|任一通道差 >1/255 的像素比例|相对原渲染 PSNR|',
        '|---|---:|---:|---:|']
    for m in r['metrics']:
        lines.append(f"|{m['camera_index']}|{m['mean_absolute_rgb_difference']:.8f}|{100*m['changed_pixel_fraction_gt_1_over_255']:.4f}%|{m['original_reference_psnr_db']:.2f} dB|")
    lines+=['','PSNR 的参考是原模型渲染，不是真实照片。没有计算相对照片的图像质量提升。只覆盖四个视角，不能承诺所有视角、分辨率或材质均不损伤。',
        f"同一原模型重复渲染最大绝对差为 {r['original_repeat_max_absolute_error']}。GPU 为 {r['gpu']}；PyTorch 最大分配显存约 {r['gpu_max_allocated_bytes']/1024**2:.1f} MiB，不包括桌面和其他进程。",
        '各次耗时在 JSON 中；没有进行充分全场景预热和重复计时，不能据首轮加载差异声称渲染加速。',
        f'[视角 103：原图、剪枝、十倍差分]({OUT}/room/camera103_comparison.jpg)',
        f'[渲染数据]({OUT}/room/RENDER_RESULTS.json)', '',
        '## scan24 独立几何参考：条件性局部评测','',
        '从 DTU 官方 Points.zip 按范围下载 stl024_total.ply，5,169,152 点；同样提取官方 ObsMask24_10.mat、Plane24.mat。参考点云不是我们生成的 clean cloud。',
        '坐标矩阵来自 turandai/gaussian-surfels-dtu 的 scan24/cameras.npz，等比尺度 324.65518 与平移 [-51.731995, -37.041767, 660.1396]；矩阵未用 GT 拟合，所有候选采用同一矩阵。',
        '**坐标来源限制：尚未取得 GeoSVR 作者链接的同版 scan24/cameras.npz。已做的 100 MB 有界压缩流探测从 scan63 开始，因此停止，未下载整套约数 GB 输入。当前数值以公共 DTU 归一化一致为条件，不应包装成已完成官方同版复现。**',
        '已检查变换后的抽样点到官方参考的中位距离约 0.264 mm；这是坐标合理性检查，不足以单独证明所有坐标参数正确。GeoSVR 的 eval 网格也仍是归一化坐标，不能直接当作毫米 GT。',
        '用官方 ObsMask 在原始点位置固定评价行，共 3069/3072 点；不能让候选通过移动出掩码来逃避评分。采用原始点到官方 STL 点集的最近点距离，无重新拟合算法、无噪声注入。',
        '', '|方法|固定观测支持内平均距离 mm|', '|---|---:|']
    for s in g['summary']:lines.append(f"|{s['method']}|{s['observed_mean_mm']:.6f}|")
    lines+=['','原始 0.498070、APSS 0.498676 的差异很小，不能据此宣布算法优劣有统计显著性。空间混合 s1 为 0.520884，约比原始距离高 4.6%；该结果仅限三个片区及上述坐标条件。',
        '这里是局部单向点到参考误差，不是完整 DTU overall 分数，也没有以它替代完整度、薄结构保持率或三角形损伤。',
        f'[全部几何数据]({OUT}/scan24/GEOMETRY_RESULTS.csv)',
        f'[坐标与评测说明 JSON]({OUT}/scan24/GEOMETRY_RESULTS.json)', '',
        '## 对研究方向的影响','',
        '- 已有成品足够支持真实后处理测试，不需要自建采集数据集。',
        '- 原版 room 中确有可剪掉、对这四视角影响很小的低 opacity 基元；下一步可扩大视角和分辨率覆盖，不能把简单剪枝包装为创新。',
        '- 成品网格的任意局部曲面并非两个平行面。当前几何-only 适配器不能直接替代原有帧来源算法；本轮没有证明它对真实网格有收益。',
        '- OSM 可用作建筑轮廓/位置先验，但不应当作毫米级 GT 或为了“验收”另起完整制图生产线。', '',
        '## 工程与复现','',
        '全部工作在 liekkas。未改系统驱动、未修改旧 Python 环境包、未覆盖原 PLY 或 v1 候选。新编译依赖放在 /srv/slam-research/grf/map-denoise/tools 下。',
        '渲染工具链：现有 PyTorch 2.5.1+cu121；隔离 gsplat 1.5.3；NVIDIA nvcc 12.1.105；CCCL 12.4.127.post1 头文件；本地解包 g++12。旧 CCCL pip 包缺少所需公共头文件，因此未使用其不完整目录。首次成功 CUDA 编译约 178 秒。',
        '使用 gsplat 默认黑背景 backgrounds=None，避免其 packed 模式背景尺寸断言；没有修改渲染器算法源码。',
        '已做：原模型重复渲染、全部输出有限值、相机中心变换检查、所有帧 scale_mat 一致性、固定掩码对照、旧输入 SHA256 复核。',
        '', '```bash',
        f'python {HERE}/get_reference.py',
        f'bash {HERE}/run_render.sh',
        f'OPENBLAS_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python {HERE}/eval_reference.py',
        f'python {HERE}/report.py','```', '',
        f'[OSM 数据与适用性说明]({HERE}/OSM_ASSESSMENT.md)',
        '[DTU/GeoSVR 官方评测说明](https://github.com/Fictionarry/GeoSVR)',
        '[gsplat 官方渲染器](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3)',
        '[官方 OSM 导入指南](https://wiki.openstreetmap.org/wiki/Import/Guidelines)']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    old_manifest=json.loads(Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v1/DOWNLOAD_MANIFEST.json').read_text())
    originals=[]
    for item in old_manifest['files']:
        assert sha(item['path'])==item['sha256'];originals.append(item['path'])
    audits=dict(original_inputs_unchanged=originals,source_hashes={str(p):sha(p) for p in HERE.glob('*.py')},
        render_results_sha256=sha(OUT/'room/RENDER_RESULTS.json'),geometry_results_sha256=sha(OUT/'scan24/GEOMETRY_RESULTS.json'))
    (OUT/'FINAL_AUDIT.json').write_text(json.dumps(audits,indent=2));print(HERE/'REPORT.md')
if __name__=='__main__':main()
