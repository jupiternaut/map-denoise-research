# Research Contract：有界视图亮度偏差

## Selected Idea

从此前共同材料解释的几何增量出发，检验有界加性视图亮度偏差下是否保留有效修复；不把旧成绩预设为新成绩。

## Core Claims

C1：同公开ε+β下，共享在至少一个锁定β保留真深度覆盖、正确KEEP、零损伤，并比channel修复更多、MAE更低。
C2：严格ε诊断的合同失配按覆盖/损伤准确呈现，不作为合法修复性能。

## Method Summary

复用冻结grouped solver；图像由raw基础图加(−β,0,+β)后裁剪；合法残差界ε+β。求解只读图像/公开参数，不用实际偏差反校正，保持旧几何和动作器，无训练。

## Experiment Design

3β×54观测×9初态×4臂=5832决策。511盒/分支、8轮、容差15/32；全部动作落盘读回后私有评价，再独立核验。单CPU、5400秒上限。

## Baselines

KEEP、channel_wide；shared_strict故意失配只用于诊断。

## Current Results

正式批次1744.387s，独立核验446.835s/PASS。β=.005/.01/.02时，共享改善44/32/25（分母432）、零恶化，MAE比KEEP低6.57%/4.65%/2.66%；channel均0改善。每β合法臂覆盖54/54、正确KEEP54/54、偏移负控KEEP72/72。

C1三档SUPPORTED。strict每β空观测54/54，覆盖0/54，486状态全为INCOMPATIBLE，输出保持初态；根盒channel白背景矛盾先排除，共享传播未执行。

## Key Decisions

继承共同材料解释与认证动作，下一步规划显式视图偏差及同图像消融。当前宽界有增量，不等于已解决真实成像。旧无偏差图未裁剪，新图加偏差后裁剪，不能把相对旧成绩差异全归因偏差。汇总只用analysis-v2；v1坐标不变/KEEP字段更正保留。

## Status

- [x] Idea selected / protocol frozen
- [x] Implementation / review / fixed run
- [x] Independent verification / claim analysis / report
- [ ] Next ablation implementation（本批不执行）

无GPU/API/安装、GitHub/GitBook发布或真实部署修改。
