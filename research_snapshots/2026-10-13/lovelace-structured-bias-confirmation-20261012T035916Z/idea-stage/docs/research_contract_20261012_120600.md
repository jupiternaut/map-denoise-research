# Research Contract：共享偏差冻结确认

## Selected Idea

上轮clip简单校准68修复、joint94修复，MSE相对校准下降9.98%，没有初态损伤；未裁剪校准已够用。最小下一步是冻结clip方法，在未使用seed确认，而不是继续设计新方法。

## Core Claims

1. 联合增量在预注册seed61004同族noise/pose实现上复现。
2. 真值覆盖、正确几何与负控仍保持。

## Method Summary

原三材料/三视图偏差必要约束；clip饱和单侧；同图公开背景T_cal；固定校准边缘化vs联合依赖，原hybrid动作。冻结旧源码，不增加信息或算法。

## Experiment Design

18clip观测、每方法144偏移/18正确初态；KEEP/calibrated/joint，共486动作、36新冷求解。原prefix，仅seed改61004；材料calibration样本删去。M2+M3一个1800秒窗口，0GPU。

## Baselines

KEEP MAE1.5 mm/MSE3.40625 mm²为原offset设计。上轮clip开发三seed：calibrated MAE1.325056/MSE2.751516；joint MAE1.239848/MSE2.476901。这些不是本轮实测。

## Current Results

尚未生成seed61004或启动正式批次。无当前结果。

## Key Decisions / Status

不调参、不改动作、不选seed。fresh审查与不同作者核验，明确同模型族暂定审查与历史物理核复用界限；无发布部署。
