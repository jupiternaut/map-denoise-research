# V7 决策消融：不要把硬分配后的截断噪声重新拟合成面

日期：2026-09-12，主机 liekkas。

**结果：取消硬分配后的 WLS 重拟合，显著减轻了 V7 重关联的表面误差；
但仍未优于原独立模型，而且层距指标经常恶化。该修改解释了部分损害，不是完整修复。**

运行：[decision-v7-nk7sm4yf](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/decision-v7-nk7sm4yf)。
这是看完公开开发结果后提出的后级版本化消融，不是未见确认。

## 唯一改变

读取已经保存的 24 输入 × 预算 1/3/6 的合法软优化状态：

- 原动作：MAP 关联 → 按硬标签重新 WLS 拟合 → 投影到硬重拟合的面。
- 新动作：**同一 MAP 关联 → 直接投影到该候选已经软 M 步拟合的面**。

没有改变输入、上游方向、扫描偏差、权重、有效行、支持、候选范围、软优化次数、
责任度、MAP 标签或软目标。不是后验均值，不在两层之间做软平均，也没有新阈值。
新算子只接受合法几何和保存的推断状态，不接受 GT 或文件路径。

先保存并哈希封存全部 72 个输出，随后才加载真值及旧指标评分。
[封存记录](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/decision-v7-nk7sm4yf/OUTPUTS_SEALED_BEFORE_GT.json)。

## 实际结果

全 24 例表面 MAE（mm，越小越好）：

| 预算 | 原 V7 硬重拟合 | 新 soft-M 面投影 | 对硬重拟合胜／平／负 | 原独立模型 |
|---|---:|---:|---:|---:|
| 1 | 0.492327 | 0.367572 | 22 / 2 / 0 | 0.205928 |
| 3 | 0.537617 | 0.388514 | 22 / 2 / 0 | 0.205928 |
| 6 | 0.561227 | 0.403107 | 22 / 2 / 0 | 0.205928 |

但每个预算相对原独立模型的 MAE 都是 **0 胜、2 平、22 负**。
对应点 RMS 新结果为 0.679391 / 0.633840 / 0.629086 mm；旧原独立为 0.458664 mm。

18 个双面输入层距误差（mm）：

| 预算 | 新 soft-M 面投影 | 相对硬重拟合层距胜／平／负 |
|---|---:|---:|
| 1 | 0.547557 | 5 / 0 / 13 |
| 3 | 0.614423 | 3 / 0 / 15 |
| 6 | 0.568868 | 3 / 0 / 15 |

因此，“把更小训练噪声当结构”的部分问题减轻了，但不能只放大 MAE 收益而省略层距损害。
例如 gap=8 的六例 MAE 在三个预算下都改善，但预算6的层距六例全部比硬重拟合差。
所有结果见 [RESULTS.csv](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/decision-v7-nk7sm4yf/RESULTS.csv)，
分预算、分层距及配对比较见同目录 `AGGREGATES.json`、`PAIRED.json`。

## 为什么此消融有判别力

一个受限例子说明风险：单面点级噪声 \(Y\sim N(0,\sigma^2)\)，两个对称候选面使
最近面硬分配按 \(Y\) 的正负切分，则硬分配后的独立均值拟合为

\[
E[Y\mid Y>0]=\sigma\sqrt{2/\pi},\qquad
E[Y\mid Y<0]=-\sigma\sqrt{2/\pi}.
\]

这会把同一真实面的两半噪声推成相距 \(2\sigma\sqrt{2/\pi}\) 的两张估计面。
该例需要上述分配和噪声假设，并不声称所有局部面都完全符合；它展示了
“分类完成后再按类拟合”的决策可以改变几何统计意义。

本消融保持整个前级状态不变后确实减轻 MAE，支持后级动作产生额外损害的判断。
但仍保留错误候选、MAP 关联以及软拟合偏差，所以未能修复整体也符合该构造的有限作用。
这不是软优化目标单调就等于几何正确，更不是决策科学能绕开观测歧义。

## 验证、成本和下一步

- 单元测试 5/5：坐标顺序、支持、原地不变、无 GT 参数、输入顺序置换等价。
- [独立审计](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/decision-v7-nk7sm4yf/INDEPENDENT_AUDIT.json)：
  72 个输出独立重建，864 个非坐标数组与旧状态完全相同，396 个指标重算通过；
  最大坐标差 3.47e-15 mm，120 个原文件哈希未变。
- 从缓存状态生成72输出并写文件共0.371秒；加评价总0.739秒。
  **这是缓存后级消融成本，不是端到端算法速度，也不能用于声称加速倍数。**
- 本轮没有扩展其他动作、没有搜索最优预算或逐案例选择赢家。旧独立模型仍应保留为
  当前有效参照；新结果进入失败机制记录，不宣布替换已验证底座。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/decision/run_decision.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/decision -p 'test_*.py' -v
```

每次生成新的 `decision-v7-*` 目录；`audit_decision.py` 接该目录参数执行独立审计。
