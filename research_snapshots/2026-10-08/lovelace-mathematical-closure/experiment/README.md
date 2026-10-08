# LOVELACE · 数学封闭实践

[第二阶段：连续深度与相机误差实验](continuous_world/README.md) · [第二阶段结果](continuous_world/REPORT.zh.md)

`STAGE2_PACKAGE_MANIFEST.json` 核对当前整树；`PACKAGE_MANIFEST.json` 是保留的第一阶段历史打包收据，原版 README 仍在第一阶段 ZIP 中。

[交互审阅页](review.html) · [结果与边界](REPORT.zh.md) · [问题定义](PROBLEM_SPEC.zh.md) · [冻结方案](finite_world/EXPERIMENT_PLAN.md)

第一阶段实现可自包含复算的有限多视图几何世界及精确安全证书；无第三方依赖，无需 TPU/GPU。确认使用的环境为 Python 3.14.0。

在 PowerShell 中：

```powershell
Set-Location 'C:\Users\gengr\Downloads\Lovelace\mathematical-closure-20261008\finite_world'
python -B -m unittest discover -s . -p test_certificates.py -v
python -B renderer_smoke.py
python -B mesh_smoke.py
python -B verify_evidence.py outputs/confirmation-v1
python -B run_experiment.py --output outputs/my-reproduction
python -B ..\analyze_witness.py
```

`run_experiment.py` 要求新输出目录，保留已有结果。随包 `assets/lovelace_patch.json` 足以复算，本轮运行无需访问完整原 GLB；重提取母体时需原 GLB，并校验源 hash。原始网格与此前素材未改动。

每份确认含源码锁、源码快照、预测封存、逐决策输入/输出、违约控制、汇总及独立验收。`outputs/replay-v1` 是冻结源码的再运行，不计为另一批独立确认。解析反例独立于确认统计。

第一阶段范围是指定九状态模型族及其有界噪声。连续目标与可认证误差外包已实现于独立的 `continuous_world` 第二阶段目录；第一阶段源码、证据和报告保留。两阶段都未完成完整 LOVELACE 重建，也未复现现有 map-denoise full9。
