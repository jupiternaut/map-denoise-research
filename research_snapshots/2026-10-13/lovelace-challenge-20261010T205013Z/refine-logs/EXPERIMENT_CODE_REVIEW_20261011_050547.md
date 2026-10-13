# 实验代码复核

## 最终状态

确认前两项实际阻塞已关闭；82项单测通过。gpt-6-astra xhigh fresh-agent只读审查，同家族provisional；最终确认还需确定性核验。

## 初轮

初轮结论：算法层未发现可复现错误；有两个 runner 协议缺口。以下针对首次读取版本，尚未复查刚收到的修补。

审查目标已核对为主机 `liekkas`、目录 `/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z`。完整读取指定协议、计划、证明与代码；全程只读，未联网、未启动确认实验。检查时没有 `outputs`，也没有确认进程。

BLOCKING

1. 各臂计时不完整。原 [run_challenge.py:125](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:125) 至 142 仅记录外包求解和网格评分；[challenge.py:83](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/challenge.py:83) 的逐初点精确预测、残差计算和选择没有归入基线成本，其他臂的选择时间也未记录。总 elapsed 包含这些工作，但存档无法重建各臂成本。最小修补：记录各臂 selector 时间，明确共享外包求解及基线预计算成本。

2. 30 分钟硬预算及未完成记录未落实。原 [run_challenge.py:95](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:95) 至 145 没有截止检查；decisions、outers 到全部选择完成才保存。实际失败情形：完成若干观测后被超时终止，磁盘没有这些已完成决策及外包，无法按协议记录未完成部分。最小修补：执行硬截止并保存已完成记录，明确标记未完成；完整科学汇总仅用于完整运行。

主代理已告知上述两项已修补；本答复不把未复查的修改视为验收通过。

NON-BLOCKING

- 现有测试覆盖实际几何、预测及策略函数，但未覆盖 runner 保存、计时和超时路径。修补后应针对这些新增路径做小规模确定性验证。
- 检查时 `verification/verify_results.py` 尚不存在，因此未审查，也未完成独立结果核验。
- `aggregates` 的 `n` 是相关决策条数；协议已明确五模板、135 配置和同观测九初态的关系。报告需继续沿用这种解释，不能把决策数当独立样本量。

已正确部分

- **连续几何外包成立。** [layered_model.py:202](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/layered_model.py:202) 使用逐赋值有效的内外矩形包含；共享相机变量丢失相关性只会放松界。异色使用可见后层差集，同色使用 union，避免重复计面积。hidden 的同姿态包含证明支持深度无信息，不依赖错误的跨姿态包含。
- **投影及严格收益合同正确。** [challenge.py:27](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/challenge.py:27) 与 [interval_solver.py:81](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/source_snapshot/continuous_world/interval_solver.py:81) 实现端点最坏平方增益；投影唯一最大化该目标。零收益 KEEP、空外包 INCOMPATIBLE、KEEP 返回实际初点及零收益均正确。最坏收益最优不代表实际 MAE 支配中点，证明文件也明确说明。
- **基线分离等价于声明的联合搜索。** [challenge.py:68](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/challenge.py:68) 满足  
  `min(l,r) max(L(l), Ref, R(r)) = max(min L, Ref, min R)`。参考相机始终零误差，三个固定半径选中正确五档；初点是实际候选，同分优先 KEEP，其后最小位移。全预测转 float64、`1e-12` tie 与协议一致。
- **未发现真值进入 selector。** [run_challenge.py:120](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:120) 的选择仅使用观测、公开模型、误差半径及该任务初点；真值参与生成和之后评价。基线候选未加入实际相机误差或确认真值。
- **参数和分账正确。** 首次版本固定参数对应 135 配置、1215 任务、4860 决策；小/大/正确初点及闭区间 prior 内外标签正确。每臂预期分母为正确 135、小偏移 810、大偏移 270；prior 内 1080、外 135。hidden 的先验外投影收益不能解释为图像深度信息，协议和证明已有此限制。
- **已有输入锁覆盖实际科学输入。** 根目录 Python、源副本 Python/JSON 及协议均纳入现有首尾检查。运行期间必须保持这些文件不变；本轮未提出新锁机制。
- **验证结果：** 新测试 31/31、旧模型与求解器测试 33/33 通过。另外对五模板各用两个诊断深度、独立直接渲染的 25 组左右姿态，核对分离残差与联合搜索，全部逐项相等。没有使用完整确认运行。

这是同家族 provisional 审查。测试通过补充实现证据，不是科学主张或连续证明的替代；最终接收仍需修补复查及确定性结果核验。

## 唯一整轮修补复查

复查结论：初轮第 1 项 BLOCKING 已关闭；第 2 项的截止机制已修复，但检查点被超时打断时仍有一个记录错误，尚不能完全关闭。未发现新增算法错误。

BLOCKING

- [run_challenge.py:30](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:30) 直接覆盖 JSON 文件。若 SIGALRM 在 `RUN_STATUS.json` 被截断后、写完前触发，[run_challenge.py:241](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:241) 的超时处理再次读取并解析该文件，会抛出 `JSONDecodeError`，无法保存 `INCOMPLETE`。
- 已用**内存文件注入上述中断时机**，随后执行当前源码中原样的异常处理体，复现该结果；没有修改磁盘文件。类似地，[run_challenge.py:33](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:33) 的四个连续覆盖写入可被 alarm 分割，留下不同批次的 decisions、outers 和计数。
- 最小修补是让这一小段检查点保存完整结束后再处理到期信号，或保留一个不会被覆盖中断破坏的完整检查点；最终重复写入也需要同样保护。这是原“超时保留可用部分档案”要求的残余问题，不涉及新算法或框架。

NON-BLOCKING

- [test_run_recording.py:30](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/test_run_recording.py:30) 验证了真实 alarm 中断，但未执行 CLI 的 `TimeoutError` 记录路径，因此不能发现上述问题。
- 核验器文件现在已经存在，并实际进入输入清单。本次没有展开第二轮核验算法审稿。

已正确部分

- [run_challenge.py:154](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:154) 已逐臂包围实际 selector 调用计时，包含基线逐初点精确预测；TIMING 对共享外包和预计算成本的说明正确。第 1 项关闭。
- [run_challenge.py:235](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:235) 设置 1800 秒 SIGALRM，覆盖完整运行；真实中断测试通过。正常完成九个观测后才保存检查点，完整评价位于全决策保存之后。
- 正常检查点序列化及计数已通过内存验证。alarm 与输入清单两个只读测试通过；实际输入清单包含 `verification/verify_results.py` 和其测试文件。
- 主机及绝对目录再次核对一致；没有启动确认实验，没有写入或修改文件。初轮关于几何、投影、联合残差等价和真值隔离的结论保持不变。

本结论仍为同家族 provisional 审查；上述记录边界修复后，还需确定性结果核验才能最终接收，单测通过本身不证明科学主张。

## 原记录缺陷定点关闭

原第二项 BLOCKING 的残余已关闭，两项初轮阻塞现均关闭。

[run_challenge.py:29](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/run_challenge.py:29) 正确保存并恢复信号 mask；嵌套写入保持 SIGALRM 屏蔽，整个检查点完成后才交付挂起信号。

只读验证通过：以内存文件模拟 `RUN_STATUS` 截断，并触发真实 SIGALRM；四份记录均完整可解析，随后抛出 `TimeoutError`，原信号 mask 已恢复。新增测试也针对该失败时机。

未发现此项修补引入的新错误；没有修改文件或启动确认实验。该结论仅关闭记录缺陷，最终实验接收仍依赖确定性结果核验。

## 主代理修复及执行门槛

补逐臂选择计时、共享成本说明、1800秒截止、每9个完成观测的部分记录；短JSON/检查点保存期间延迟到期信号，在写完后立即交付。单测实际注入truncate时刻的SIGALRM验证，未造新实验框架。确认前64项既有/新模型策略测试与新增记录、独立核验测试合计82项全部通过（3.552秒）。

完整request/response/meta保存在本地 .aris/traces/experiment-bridge/2026-10-11_run01，不用于公开提交。没有按审稿意见改变科学主张、数据或阈值。

