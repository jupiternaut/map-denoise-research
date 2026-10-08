# LOVELACE 连续深度实验

[结果报告](REPORT.zh.md) · [交互审阅](review.html) · [可导出图](results.svg) · [冻结方案](EXPERIMENT_PLAN.md) · [数学推导](THEORY.md)

实验把深度从九个离散候选扩展为整个 `[540,660] mm` 实数区间，并包住两侧相机在 x 方向的连续位置误差。中间参考相机固定。求解器保存所有不能严格排除的区间，只在整个外包上都能证明平方误差下降时移动点。

科学实验、测试及证据复核仅使用 Python 标准库；确认环境 Python 3.14.0。可选的页面/科研图重建使用已安装的 Matplotlib，依赖见 `presentation/requirements.txt`；交付的 HTML/SVG 可直接离线查看。请从包的顶层目录执行，不要进入 `continuous_world` 后直接运行模块：

```powershell
Set-Location 'C:\Users\gengr\Downloads\Lovelace\mathematical-closure-20261008'
python -B -m unittest continuous_world.test_interval_model continuous_world.test_interval_solver -v
python -B -m continuous_world.verify_continuous continuous_world/outputs/confirmation-v1
python -B -m continuous_world.run_experiment --output continuous_world/outputs/my-reproduction
python -B continuous_world/evidence/audit_sources_and_summary.py
python -B continuous_world/presentation/build_review.py
```

运行器要求一个不存在的新输出目录，保护原结果。重新验收会更新 `VERIFICATION.json`；交付时版本及验收器 hash 已封存。预测输入、区间分支记录、输出、源码及几何资产均随包保留，无需访问原始完整 GLB 或 TPU/GPU。

确认集含 72 个参数与噪声配置、216 次初点决策；72 个正确初点全部保留，144 个 ±60 mm 偏移全部改善。它们来自两个固定几何模板，不是 72 个独立物理物体。LOVELACE 分支仅渲染来源可核对的 M4 孤立种子三角面；另一分支是矩形解析对照。

`outputs/confirmation-v1` 是冻结确认；`outputs/replay-v1` 是冻结源码重跑，同一批数据，不能增加样本数。`VERIFICATION.json` 验收区间分支和决策，`AUDIT.json` 独立复核来源与汇总，`REPRODUCIBILITY.json` 记录重跑一致性，`TESTS.json` 记录 33 项开发检查。独立验收器复用了经数学审阅的几何内核，并非另一套几何实现。

保留区间是可能含多余候选的保守外包，宽度不是置信区间。毫米来自归一化模型尺度，未经真实物理校准。安全结论要求真实世界及误差都在声明模型内；违约控制已单独归档。此阶段尚未覆盖完整角色、相机旋转、未知材质、变形或真实地图。
