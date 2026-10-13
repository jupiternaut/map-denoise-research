# Experiment Tracker：最终

| Run ID | Milestone | Status | Evidence |
|---|---|---|---|
| R001 | M0 | COMPLETE | 44 tests OK，含四夹具独立核验；原冒烟失败日志保留 |
| R002 | M1 | PASS / provisional | 同模型家族非作者完整源码审查，冻结hash一致 |
| R003 | M2 | COMPLETE | 一次主批exit0；108图像、3888动作、216新求解，452.208秒 |
| R004 | M3 | PASS | 57,276节点、649,342精确更新、6,055矛盾证书；356.747秒；70源码/37历史输入未变 |
| R005 | M4 | COMPLETE | 精确analysis/RESULTS.json、配对例外、C1/C2、消融和最终报告 |

本批无遗漏/缩样本/重跑；主批+核验同5400秒窗口实际831.718秒。raw115/115修复，clip68/94；全方法相对初態损伤0。raw两新法相同，clip联合额外26修复、MSE再降9.98%。

研究决策为raw简单calibrated、clip保留joint；无部署或发布。当前无后台实验。未用seed等后续仅建议，不计本批。
