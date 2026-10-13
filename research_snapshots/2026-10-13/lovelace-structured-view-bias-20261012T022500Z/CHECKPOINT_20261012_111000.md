# Checkpoint：结构化视图偏差已完成

host: liekkas
root: /srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z
output: outputs/structured-v1
status: COMPLETE / VERIFIED / ANALYZED

## 当前科学结论

- raw：公开背景校准恢复115/432修复，joint相同；两法与无偏差raw参考486/486输出、54/54外包相同。
- clip：wide32、calibrated68、joint94修复；joint相对calibrated新增26，MSE下降9.98%。
- 每单元/方法覆盖54/54、正确KEEP54/54、负控KEEP72/72、相对初态损伤0；配对退步raw1、clip1仍改善初态。
- 采用研究选择raw简单校准、clip联合模型；部署默认未变。

## 完成记录

44 tests及四夹具集成通过；M1同模型家族非作者完整源码PASS/provisional。M2仅运行一次，216新求解、108复用外包、3888动作，452.208秒。M3不同作者实现核验PASS，356.747秒；同5400秒窗口实际831.718秒。M4从保存工件精确复算，analysis/RESULTS.json已经产生，不覆盖原评价。

科学源码和PROTOCOL从主批启动冻结。核验70源码、37历史输入身份一致。核验复用历史几何核，不能冒称物理soundness外部证明。旧实验全只读，未重跑A+B。

## 恢复时先读

REPORT.md → refine-logs/RESULT_TO_CLAIM_20261012_111000.md → analysis/RESULTS.json → verification/VERIFY.json（位于output）。只读检查，不把既有structured-v1再次作为运行输出。

## 仍未执行

未用seed确认、偏差排列变化、连续噪声、真实场景、等CPU比较、LP、学习器和新动作器。无GPU/API/安装/SSH，无GitHub/GitBook发布，当前无后台实验。

下一最小候选为另注册clip18组观测/两新法/36冷求解。不得把建议写成已运行，也不得改冻结实现或覆盖旧输出。主批原绝对deadline不能用作新批重置授权。
