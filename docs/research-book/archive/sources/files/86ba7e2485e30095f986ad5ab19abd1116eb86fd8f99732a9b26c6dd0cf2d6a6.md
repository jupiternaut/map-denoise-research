# V8 条件选择噪声补偿实验

状态：已实现、已测、保留实验分支；**不能替换当前保留滤波器**。

请先看 [报告](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v8/REPORT.md)。它减少了重关联后的部分点误差，但会损伤双层间距和倾角，整体仍未胜过原保留模型。

## 文件

- `compensation.py`：无真值输入的解析条件均值补偿及完整输入 API。
- `PROTOCOL.md`：本轮实验方案，生成结果前保存。
- `run_v8.py`：24 个开发输入、3 个预算、3 个动作，先保存输出再评价。
- `test_compensation.py`：14 项数学/代码单元测试。
- `audit_v8.py`：不导入新算法的积分、投影与指标复核。
- `check_api.py`：从合法原始输入复核两例完整运行。
- `checkpoint.py`：只读检查 V7 及更早证据的哈希。

## 调用

在项目根目录的 Python 环境中：

```python
from exploration_v8.compensation import estimate

# xyz_world_m: N×3 世界坐标，单位米；scan_id: N 个扫描来源编号。
# sigma_mm: 已给定的随机测量噪声尺度，不是帧偏差总量。
output, info, artifacts = estimate(xyz_world_m, scan_id, sigma_mm=1.0,
                                  budget=6, mode="conditional")
```

输出仍是全部 N 点；未支持点保持原值。公开 API 不接收真实层号、真实法向或真实噪声。
`artifacts['subtracted_noise_mm']` 是模型估计的条件噪声均值，不是实际噪声的恢复值。
完整 API 暂时仍计算并丢弃一次旧硬重拟合，不宣称工程最优或 GPU 加速。

## 重跑本轮

使用已验证的 `liekkas` 主机和原环境，不安装新依赖。以下每次生成独立结果目录，不覆盖旧运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v8/run_v8.py
```

将输出的新目录作为 `audit_v8.py` 和 `check_api.py` 的唯一位置参数，使用相同解释器及线程环境运行。
两项核验只新增 JSON，不覆盖已有结果；同目录再次运行会拒绝覆盖已有核验文件。

本轮输出：[selection-compensation-v8-ey291bta](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/selection-compensation-v8-ey291bta)。
旧数据保护清单：[session-v8-1c5rygxc](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/session-v8-1c5rygxc)。

## 后续接手约束

不要把 24 个已有开发条件称为新确认或 24 个独立场景。不要把 0.744 ms 后级耗时当端到端延迟。
不要用真值给单面/双层分别设置补偿强度或开关。不要把同 XY 拟合截距间距等同于直接测得的物理厚度。
原检查点只读，后续构造另开目录并保留此次失败；本轮未更新 Skill。
