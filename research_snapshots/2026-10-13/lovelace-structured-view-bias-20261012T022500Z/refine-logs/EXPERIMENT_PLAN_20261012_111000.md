# Experiment Plan：C+D完成结算

Problem：保留有效几何、修复不足。
Method thesis：同图像的共同偏差先公开校准，残余不确定性保留共享依赖；不改候选或动作。

## Claim map

| Claim | Minimum evidence | Final status |
|---|---|---|
| C1：结构化观测表示能否恢复宽界下丢失的修复 | 同图像及初态比较，真值覆盖、正确/负控保留、配对得失完整 | 本批支持；raw48→115、clip32→94；归因包含校准与行处理 |
| C2：joint是否超出相同T_cal的简单校准 | 同信息/初始化/树/动作删除共享依赖，报告平均及逐例、实际成本 | raw相同；clip新增26修复、MSE再降9.98%，多29.24%求解成本 |

## Blocks and fixed design

B1：KEEP、历史wide、calibrated、joint；raw/clip同54源观测、每单元432偏移/54正确初态，3888动作。B2：calibrated→joint的配对、T_cal、集合关系、种子及成本。两项均完成。原protocol、epsilon/beta、camera范围、三分支、511盒/8轮/15/32 mm及hybrid动作从launch冻结。

## Milestones and measured budget

| Milestone | Status | Measured evidence |
|---|---|---|
| M0 | COMPLETE | 44 tests通过及四夹具集成，失败冒烟保留且launch前修正 |
| M1 | PASS / provisional | 同族非作者完整source审查，无blocker |
| M2 | COMPLETE | 216新冷求解、108历史外包、3888动作；452.208秒 |
| M3 | PASS | 全量独立实现核验；356.747秒；共同绝对窗口831.718秒<5400秒 |
| M4 | COMPLETE | 精确配对/集合/校准/参考匹配、结果主张、消融、报告 |

预算：0GPU/API/安装/SSH，未重跑正式主批或A+B。后处理与文档不充作新科学样本。历史wide身份与计时明确标为复用，无新场景/新seed。

## Decision and cuts

raw研究上选择简单校准；clip保留joint有限收益及成本。部署和发布不变。未执行LP、学习器、额外采样、第二生成器、新动作器或敏感性网格。后续未用seed确认只是一项新注册建议。

## Checklist

- [x] 两主张、同信息、统计单位明确。
- [x] 两新法共同公开T_cal初始化。
- [x] M0全测试与夹具通过。
- [x] M1完整源码审查。
- [x] M2完整3888动作先保存再评价。
- [x] M3全量PASS。
- [x] M4精确结算与报告。
