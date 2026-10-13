# A13 检查点

主机 liekkas；目标 `/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`，现有runs链接解析为同名物理目录。使用既有open3d-019 Python、单线程CPU，无安装/GPU/外部API。

## 已完成

- 源码40文件冻结：SOURCE_FREEZE.json；新种子20261013131/132/133生成18世界、126输入。
- 三臂两预算108冷调用，756预测全部完成，随后同提案诊断、独立核验、评分。
- certified与numeric两预算252输出逐项相同；297个不同排除各在两预算运行，594记录全部核验通过；开发89/89通过。
- 108扰动输入：认证MSE6.536208，identity20.772442，改善70/108、不变38/108、恶化0；18正确现任未动。
- certified平均冷调用约0.397秒；旧6秒基线4.467秒/MSE6.600580；旧24秒基线7.767秒/MSE6.397256。
- 24秒旧法在15个有信息世界均更准；三负控世界持平。不能把本轮写成全面精度胜利。
- 新增评分后残差分析：inspect_residuals.py、PAIRED_RESIDUALS.json。不是方法修订或预注册新主终点。
- 开发失败与freeze模块碰撞已记账；未生成污染的确认结果，旧源码未改。

## 当前状态

正式实验无后台进程。fresh结果审查进行中；报告草稿REPORT_DRAFT.md，最终审查落盘后生成版本报告和REPORT.md。不可把“审查进行中”写成已经通过。

## 恢复任务时

不要原地重跑freeze/generate/predict/score，输出为独占创建。先读REPORT、EXPERIMENT_AUDIT、SOURCE_FREEZE及outputs/formal-v1/SUMMARY/ANALYSIS/INDEPENDENT_VERIFICATION，再决定下一研究任务。

下一建议未执行：固定已有快搜索，研究少数联合约束能否低成本收紧残余边界；先区分旧法精度优势来自更强约束、相关性还是停止分辨率。不可用这轮确认真值选择应用位置。此轮不新增数据机制、算法库、部署或GitHub/GitBook推送。

## 必须保留的限定

这是冻结有界误差成像族的18新参数世界，不是真实点云或新物理机制迁移。legacy在十进制化读数上精确求解；A13才显式处理原有理数观测的binary64舍入原像。当前证书全部接受，没有测到拒证导致保留集合变宽的实际确认例（只通过强制拒证路径测试）。6/24预算的认证输出相同，不算两组独立科学复现。
