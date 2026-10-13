# A7 复现命令

主机liekkas；工作根 /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1；仅现有CPU环境。所有旧阶段只读。

手工夹具（54项）：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -p 'test_*.py' -v
```
上条须从本工作根执行。

锁定正式入口（仅审查门有效、目标目录未存在时；程序拒绝覆盖）：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1/execute_pose.py --output /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1/outputs/pose-v1
```
main+verify使用一个3600秒绝对期限。不能把已生成输出补跑为新结果。

查看完成状态：outputs/pose-v1/STAGE_STATUS.json；标准逐参数核验计划路径：outputs/pose-v1/verification/VERIFY.json（本次未生成，见下节）。UNKNOWN只是未找到，并非无解。
## 实际状态与诊断（本次未正式验收）

正式计算已经消费 outputs/pose-v1，最终检查失败，STAGE_STATUS 为 INCOMPLETE。不得删除目标重跑或改写成 COMPLETE。上文标准 verification/VERIFY.json 未生成，不能把计划路径当作已完成核验。

在原共享绝对截止内，已执行以下独立保存记录诊断；它不导入或运行新求解器，不改原状态，只使用独立核验原语检查保存结果：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B refine-logs/audit_saved_records.py
```

仅本次原截止内一次，程序拒绝覆盖诊断收据；不要重置时间重跑。实际收据为 diagnostic-audit/SAVED_RECORD_AUDIT.json，status=SAVED_RECORD_NUMERICAL_PASS、formal_acceptance=false。最初源码集合失败保留在 outputs/pose-v1/INCOMPLETE.json。

可随时做只读描述性复算（不求解、不改参数、不解析新私有数据）：

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B refine-logs/analyze_pose.py
```

该命令输出JSON到stdout，对照 ANALYSIS.json。它明确要求原INCOMPLETE与已保存数值诊断PASS并存，不制造正式验收。
