# Research Contract：共享视图偏差

## Selected Idea

原目标保留有效几何、修复不足。上轮同图放宽误差界使115/99修复归零，固定宽界偏差后raw48/clip32；8→16无新终点。本轮检验C+D误差结构，不重跑A+B。

## Core Claims

1. 与独立宽界比较，共同view偏差表示能否恢复有效修复，保持覆盖和正确几何。
2. 与同信息背景校准比，联合依赖有没有需要保留的增量。

## Method Summary

darkness三变量与t_v=beta-b_v三变量非负线性必要约束。raw双侧、clip端点单侧。公开全depth/branch背景推导共同T_cal；calibrated固定它，joint每盒联合收缩。两法均无额外观测/真值选择，旧动作不变。

## Experiment Design

两图像，四方法，同54基础观测/486初态；3888动作，216新coldsolves、108历史wide外包。epsilon=beta=.01、radius公开、511盒/branch、8轮、15/32 mm，0GPU/API。主运行+核验5400秒一次窗口。

## Baselines

历史wide：raw48/432修复，MAE1.398222；clip32/432，MAE1.430189。同源上轮已核验工件，只读复用。KEEP偏移MAE1.5、MSE3.40625。数字是历史事实，不是本轮新重测。

## Current Results

尚无本轮正式结果；已确认公开几何背景可枚举48pixel，待独立核验。两个新方法实现/审查进行中。

## Key Decisions

共同背景warm-start隔离联合依赖，不让joint浪费轮数重学背景。白背景饱和不能保证识别正偏差，不用均值点纠正缩noise界。复杂度只在新增修复时保留。

## Status

计划已写；solver与verifier作者并行，root负责runner/集成。尚未启动主批，无发布、部署或外部服务操作。
