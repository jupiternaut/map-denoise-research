# Research Contract：共享视图偏差，最终结算

## Selected Idea

原目标保留有效几何、修复不足。本轮只检验C+D：共同视图偏差和同图像公开背景校准，不重跑A+B，不改候选或动作。

## Core Claims

1. C1：结构化观测表示相较逐测量宽界，能恢复多少有效修复？本批支持；raw48→115，clip32→94，但归因包含校准和观测行处理。
2. C2：保留共同偏差是否超出相同背景校准？raw零增量，clip新增26修复、MSE相对校准下降9.98%；有限主张支持。

## Method Summary

三个材料暗度与三个非负偏差变量；raw双侧、clip饱和端点单侧。全depth/branch保证背景推出共同T_cal；calibrated固定边缘化，joint联合收紧。无新增采样或真实偏差输入；旧hybrid动作不变。

## Experiment Design / Baselines

同54合成基础观测，raw/clip两单元，每单元432偏移与54正确初态；四方法3888动作。历史wide外包108份精确复用；两新法216次冷求解。KEEP偏移MAE1.5 mm/MSE3.40625 mm²。epsilon=beta=.01，511盒/branch、8轮、15/32 mm；不声称同CPU。

## Current Results

raw calibrated/joint MAE1.164813，MSE2.266513，修复115/432，MSE相对KEEP下降33.46%；clip joint MAE1.239848，MSE2.476901，修复94/432，MSE相对KEEP下降27.28%。所有方法覆盖/正确保留54/54，负控72/72，初态损伤0；配对退步raw1、clip1完整保留。

raw背景把全部偏差点识别，两新法输出及外包与无偏差参考逐项相同。clip背景留下两个视图区间，joint39/54观测进一步缩小凸包。集合更小不保证中点逐例更准。

## Key Decisions

研究上raw采用简单校准、clip保留联合增量；部署不改。后续首选另注册未用seed确认clip；不在本轮追加新生成器、LP、学习器或动作。

## Status

M0–M4完成；主批COMPLETE、核验PASS、后分析完成。M2+M3实际831.718秒，小于原5400秒窗口。44测试通过，70源码/37历史输入不变。不同作者机械核验复用历史几何核；非作者审查同族暂定。0GPU/API/安装/SSH，0发布/部署；无后台实验。
