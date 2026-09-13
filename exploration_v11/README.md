# V11 空间关联滤波原型

已完成结果见 [REPORT.md](REPORT.md)。新种子薄层合成有收益，真实迁移尚未成立。
Skill 用于同输入比较、记录失败与复核，不作形式证明前置要求；本轮未修改Skill。

项目根目录：`/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`
主机：`liekkas`。以下命令创建新目录，旧检查点只读。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v11/run_v11.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v11/run_v11.py --fresh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v11/check_real.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v11/audit_v11.py
```

此处 `--fresh` 重跑固定确认种子，不是每次生成新的未知测试集；这些种子本轮之后已经暴露。
每次运行的 source/ 保存当时源码，不将后续报告或修正回写历史快照。

API：`api.estimate(xyz_world_m, scan_id, sigma_mm, method='spatial_free', iterations=12)`。
返回输出XYZ、诊断和支持信息，完整调用旧合法输入上游。输入必须有扫描来源，长度单位米，sigma参数毫米。
CLI：`api.py --input 输入.npz --output 新输出.npz --sigma-mm 1 --iterations 12`；
NPZ至少含 xyz_world 和 scan_id，输出必须不存在。仅输出实验候选，不覆盖源地图。

谱初始化方向修正保存在额外 `spatial_seed_aligned` 方法，原四条支路默认行为不变。
