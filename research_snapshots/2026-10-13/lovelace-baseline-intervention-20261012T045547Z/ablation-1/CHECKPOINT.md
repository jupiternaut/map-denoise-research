# A7→A1+A5 完成检查点

生成：2026-10-12 15:28:31（Asia/Shanghai）；精确主机liekkas。
根：`/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1`。

## 状态

主运行COMPLETE，独立机械核验PASS；全部所选里程碑DONE，无后台科学任务。旧data、recovery-1与父报告只读，当前阶段仅在ablation-1写入。

A7：1944动作，0新求解；A1：144冷求解、1296动作；A5复用四格。主+核验586.409933秒，共享窗口3600秒，无重置。34项精确路径测试PASS；fresh代码审查未完成，复用上下文代理+本地清单暂定通过，完整trace保留。

## 接续时必须继承的发现

- B80深档joint完整11/40修复，删远缘F2/40，删邻列7/40；MSE2.367334/3.216199/2.694615 mm²。B70深档删F无变化。
- B80深档F×共享偏差交互0.162537244 mm²。
- projection cal→joint0配对退步，hybrid共5；但投影平均MSE较hybrid略差。不替换默认。
- 全部正确初态和负控保留；零损伤只是本批事实。
- 最强支持是观测约束的条件机制，不是新实景净收益或最优架构证明。

## 唯一建议的下一问题（尚未执行）

固定B80图像，将跨视图共享材料改成逐视图材料，检验同视图面积比是否足以保留深档修复。先形成独立协议与可审查变量/约束，再正式运行。不能从本次继续授权自动执行历史全部A2–A13菜单。

## 未做

新场景、新seed、新生成器、GPU/API/SSH/install、真实照片迁移、发布GitHub/GitBook、改变部署默认。当前输出不覆盖，不为改变结论重跑。

## 入口及命令

- REPORT.md：结果与意义；analysis/CONTRASTS.json：精确配对账。
- outputs/ablation-v1/verification/VERIFY.json：独立收据。
- .aris/traces/experiment-code-review：完整审查上下文。
- PROCESS_AUDIT.md：夹具错配/更正的限制。

已执行：
```sh
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/test_suite.py
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/run_ablation.py --output /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/outputs/ablation-v1
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/verify_ablation.py --output /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/outputs/ablation-v1
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/analysis/analyze_ablation.py
```

analysis程序使用唯一创建模式；现有正式目录不应重新运行并覆盖。预算截止约束已经作为本阶段执行记录封存，不为下一阶段续用。

