# V6 B：association × sharing

已完成：[结果与限制](REPORT.md)。仅公开数据的GT诊断，不能部署或选模型。
源码/协议冻结运行：[association-v6-bgcq8zff](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/association-v6-bgcq8zff)。

API：`freeze(xyz_world_m, scan_id, sigma_mm)` 仅接受合法输入；
`fit_original(state, sharing='independent'|'shared')` 没有GT入口；
`fit_oracle_split(state, gt_layer, sharing=...)` 是显式evaluation-only入口。
拟合返回 `(world_m_same_order, JSON_info, group_assignment_original_order)`。

`RESULTS.csv` 的 `output/output_sha256` 指向完整NPZ。NPZ另存扫描ID、来源点ID、
support mask、active拟合原序索引及组分配。JSON内 `info.common_fingerprints` 包含
weights/active/support及完整合法状态指纹；GT组构成单独放在evaluation-only元数据。
`states/` 保存原冻结数组；`source/` 保存运行时源码与协议；旧文件均只读。

复现必须在liekkas，运行会创建新的唯一目录，不覆盖历史：

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v6/association
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_association.py
```

独立核验新的运行目录时，用相同环境执行 `verify_association.py /absolute/new/run`。
核验不重新拟合、不生成数据；已有`VERIFICATION.json`时拒绝覆盖。当前旧运行已核验。

核心算法SHA256：`8c0034bd031cf141cf5ad2e8fe283df72d8c6f66a4aaa397c1e894f5bb7d40c7`。
冻结协议SHA256：`77b6f3dad8aa0539c2a6dd04054b5e0fe153181539aa6aa2471e92f78b518ccc`。
其余源码哈希见运行的`SOURCE_MANIFEST.json`。
