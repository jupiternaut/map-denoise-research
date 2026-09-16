# V25 任务队列

状态含义：TODO / RUNNING / DONE / FAILED_WITH_EVIDENCE / BLOCKED / SUPERSEDED。

| ID | 状态 | 工作 | 依赖 | 结束工件 |
|---|---|---|---|---|
| Q00 | DONE | 读入口、身份预检、建立 SESSION | 无 | SESSION.json；preflight INPUT_PATHS_READY |
| Q01 | DONE | 复用相机解析并核对投影/尺度 | Q00 | tests.test_cameras；scan24 0.356/1.794、scan37 0.441/1.696 |
| Q02 | DONE | 选输入定义的 ROI 与普通面控制 | Q01 | runs/2026-09-14T165314Z/rois.json（8 父 ROI + 8 core） |
| Q03 | DONE | 实现固定空间域/密度控制新评价器 | Q00 | evaluation/metrics.py；空输出与倍点测试通过 |
| Q04 | BLOCKED | 官方 MVS 可用性核验/有限安装 | Q00 | colmap 未找到；官方基线缺口保留 |
| Q05 | DONE | 实现共享证据与标准融合备用臂 | Q01,Q02 | 各 ROI `evidence.npz` + `fusion_wta.ply` |
| Q06 | DONE | 实现受限输出与 V25 再生 | Q05 | restricted_* 与 v25_* 实际几何 |
| Q07 | DONE | 首轮两个场景/约8 ROI比较 | Q03,Q06 | RESULTS.csv 父 ROI；SEALED_v1.json |
| Q08 | DONE | 按瓶颈做大胆构造/组合 | Q07 | v25_wta_atlas / depth_cc / gated_move + core；LEDGER.jsonl |
| Q09 | DONE | 复查密度/覆盖/控制损伤与选模 | Q07,Q08 | REPORT.md §6；不升级 |
| Q10 | BLOCKED | 输入侧准备1–2新场景 | Q00 | downloads/NEW_SCENES_MANIFEST.json；无可用确认包 |
| Q11 | DONE | 锁定主候选与确认协议 | Q09 | METHOD_LOCK.json：不部署 V25，默认 identity |
| Q12 | BLOCKED | 新场景确认 | Q10,Q11 | 无确认场景且方法未升级；scan37 仍是开发集 |
| Q13 | DONE | 独立重算核心指标/旧源哈希 | Q07 | VERIFY.json 128 行最大差 0；旧源哈希未变 |
| Q14 | DONE | 真实图、报告、复现命令、续跑 | Q09,Q13 | REPORT.md、COMMANDS.md、figures/、CHECKPOINT.md |

Q04 与 Q10/Q12 是权限/工具缺口，不是科学成功。假说未成立，队列在授权范围内已闭合。

## 准备结束当前回复前

1. 已有真实局部输出与同观测对照。
2. 官方基线仍缺，已写明 BLOCKED。
3. 队列无更多可执行安全任务（确认被场景和方法锁定挡住）。
4. 无后台长进程。恢复命令见 CHECKPOINT.md。
