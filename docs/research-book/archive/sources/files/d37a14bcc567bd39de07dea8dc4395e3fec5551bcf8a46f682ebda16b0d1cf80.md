# V9：分组补偿、条件交叉拟合与选择混合解混

状态：实现与开发测试已完成，**暂不替换原保留滤波器**。先看 [总报告](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v9/REPORT.md)。

## 代码入口

- `action_ablation.action_frozen(state, artifacts, sigma_mm, mode)`：固定分组的 none/constant/slope/full 输出动作。
- `crossfit.candidate_search(state, original_artifacts, sigma_mm, budget, folds)`：固定上游后的候选交叉拟合；一折为匹配全数据对照。
- `crossfit.project_candidate(state, artifacts)`：直接投影到所选候选，不再硬组拟合。
- `unmix.unmix_frozen(state, artifacts, sigma_mm, mode)`：offset/affine 联合解混，固定 SVD 截断，不接收真值。

上述返回同点序世界坐标，单位米；`sigma_mm` 单位毫米，是已给定测量噪声尺度。
`state/artifacts` 是项目内合法前级输入，来自原始观测的 V7 接口；不是面向任意点云的成熟部署包。
原始点云的调用可按以下方式组合（项目根目录加入 Python 导入路径）：

```python
from exploration_v7.algorithm.reassociation import freeze, fit_frozen
from exploration_v9.unmix import unmix_frozen

state = freeze(xyz_world_m, scan_id, sigma_mm=1.0)
_, _, candidates = fit_frozen(state, variant="reassociate", budget=6,
                              sharing="independent")
output, info, artifacts = unmix_frozen(state, candidates, 1.0, mode="offset")
```

这是研究接口示例，不是推荐替换旧算法；开发成绩见报告。全部代码均不接收真实层号、真实噪声或真实位姿。

## 复现

主机 `liekkas`，沿用已有环境，不安装新依赖。每次运行新建目录，不覆盖旧产物。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v9/run_v9.py
```

第二阶段把最后的脚本路径换为 `/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v9/run_unmix.py`。
独立审计分别是 `audit_v9.py`、`audit_unmix.py`，传入对应输出目录；脚本禁止覆盖已有审计记录。
`math_diagnosis.py` 运行受限模型的公式、积分和模拟，不读取项目实验真值。

测试：在项目根目录用相同环境运行 `python -m unittest discover -s exploration_v9 -p 'test_*.py' -v`，当前共 28 项。

## 已有证据

- 第一阶段：[/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/crossfit-actions-v9-dcz583qe](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/crossfit-actions-v9-dcz583qe)
- 第二阶段：[/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/selection-unmix-v9-qe05wnli](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/selection-unmix-v9-qe05wnli)
- 历史保护清单：[/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/session-v9-bzr612se](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/session-v9-bzr612se)

24 条件只来自 3 个已使用种子。没有新增真实地图效果或未见确认。下一轮另开目录，不能覆盖 V8/V9 的代码快照和结果，不能把当前最小二乘解误称为物理真值。
