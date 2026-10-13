# Research Output Manifest

Exact host liekkas；root /srv/slam-research/grf/map-denoise/runs/lovelace-structured-bias-confirmation-20261012T035916Z。
标签是版本标识，实际主批时间见BUDGET/RUN_STATUS。科学文件在M2启动后冻结；分析与文档不进入冻结科学哈希图。所有实验状态以保存记录为准。

| Timestamp (Asia/Shanghai) | Skill / role | File | Stage | Description |
|---|---|---|---|---|
| 2026-10-12 12:06 | experiment-plan | AGENTS.md; PROTOCOL_20261012_120600.md / PROTOCOL.md; REGISTRATION.json | plan | 新数据前注册seed61004和18观测/36求解/486动作 |
| 2026-10-12 12:06 | experiment-plan | refine-logs/EXPERIMENT_PLAN_20261012_120600.md / EXPERIMENT_PLAN.md | plan | 两主张、五阶段、1800秒共享窗口 |
| 2026-10-12 12:06 | experiment-plan | refine-logs/EXPERIMENT_TRACKER_20261012_120600.md | plan | 初始状态，保持原版本 |
| 2026-10-12 12:06 | experiment-plan | idea-stage/docs/research_contract_20261012_120600.md | contract | 未见确认与旧开发分开 |
| 2026-10-12 12:14–12:19 | experiment-bridge | prepare_confirmation.py; run_confirmation.py; test_confirmation.py; verify_confirmation.py; test_verify_confirmation.py | M0 | 五源码，16项非确认测试；冻结后未改 |
| 2026-10-12 12:14–12:19 | experiment-bridge | refine-logs/M0_TESTS_20261012_121400.md; M0_M1_ACCEPTANCE_20261012_121900.md | M0/M1 | 测试及准入记录 |
| 2026-10-12 before generation | fresh reviewer | .aris/review.request.md; review.response.md; review.meta.json; REVIEW_GATE.json | M1 | 请求、整理回复、元数据、准入PASS；原始最终回复另见raw文件 |
| 2026-10-12 12:18–12:20 | experiment-bridge | data/observations.json; tasks.json; GENERATION.json; private/truth.json | M2 | 18公共观测162初态，评价真值分离 |
| 2026-10-12 12:18–12:20 | experiment-bridge | outputs/confirmation-v1/REGISTRATION.json; SOURCE_LOCK.json; INPUTS.json; BUDGET.json; RUN_STATUS.json | M2 | 科学78文件、新输入4文件、原绝对窗口；COMPLETE |
| 2026-10-12 12:18–12:20 | experiment-bridge | outputs/confirmation-v1/observations.json; tasks.json; outers.json; checkpoint/ | M2 | 公共输入副本和18联合/校准外包证书 |
| 2026-10-12 12:18–12:20 | experiment-bridge | outputs/confirmation-v1/decisions.json; timing.json; selector_timing.json | M2 | 486保存动作、36冷求解及动作计时 |
| 2026-10-12 12:20–12:22 | independent verifier | outputs/confirmation-v1/verification/VERIFY.json | M3 | 正式PASS，完整必要行、树、动作、指标精确重算 |
| 2026-10-12 12:20 | experiment-bridge | outputs/confirmation-v1/verification-premature-attempt/VERIFY.json; CORRECTION_20261012_122600.md / CORRECTION.md | correction | 提前核验FAIL保留，同原deadline恢复；未重跑主批 |
| 2026-10-12 12:20–12:22 | analyze-results | outputs/confirmation-v1/results.json / results.csv; coverage.json / COVERAGE.csv; PAIRED.json / PAIRED.csv; summary.json / SUMMARY.csv | M4 | 全486结果、54覆盖、162配对、171分组；全部正负结果 |
| 2026-10-12 12:26 | analyze-results | REPORT_20261012_122600.md; idea-stage/docs/research_contract_20261012_122600.md / research_contract.md; refine-logs/EXPERIMENT_TRACKER_20261012_122600.md / EXPERIMENT_TRACKER.md | M4 | 首版完成结算与有限C1/C2 |
| 2026-10-12 12:30 (version tag) | analyze-results | analysis/MECHANISM_TRACE_20261012_123000.json | M4 | 已有18观测的凸包、初态、动作、7新增与6进一步改善；0新求解 |
| 2026-10-12 12:33 | ablation-planner | refine-logs/ABLATION_ASSESSMENT_20261012_123300.md / ABLATION_ASSESSMENT.md; .aris/ablation.request.md; ablation.response.raw.md; ablation.meta.json | M4 | 同族非作者消融评估，现有关键对照已完成，未来项未执行 |
| 2026-10-12 12:33 | goal-first-reporting | REPORT_20261012_123300.md / REPORT.md; refine-logs/EXPERIMENT_RESULTS_20261012_123300.md / EXPERIMENT_RESULTS.md | M4 | 最终报告，增加改善定义与机制轨迹/消融链接 |
| 2026-10-12 12:35 | independent verifier | .aris/final_integrity.request.md; final_integrity.response.raw.md; final_integrity.meta.json | audit | 只读SOURCE_LOCK 78/78、INPUTS 4/4；无重跑 |
| 2026-10-12 12:35 | experiment-bridge | CHECKPOINT_20261012_123500.md / CHECKPOINT.md; MANIFEST_20261012_123500.md / MANIFEST.md | complete | 完成交付，无发布/部署或后台科学作业 |

报告未声称可移植独立包；绝对路径历史依赖仍需发布阶段处理。.aris是本地完整审查轨迹，不自动发布。原版文档、失败记录与科学输出均保留。


## 12:38最终逐文件索引

下面每行一个工件；时间为清单核对版本标识，不冒充文件生成时间。科学输出的实际时间见原记录。

| Timestamp | Skill | File | Stage | Description |
|---|---|---|---|---|
| 2026-10-12 12:38 | review-tracing | .aris/REVIEW_GATE.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/ablation.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/ablation.request.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/ablation.response.raw.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/final_integrity.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/final_integrity.request.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/final_integrity.response.raw.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/meta/events.jsonl | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/review.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/review.request.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/review.response.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/review.response.raw.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/ablation-planner/2026-10-12_M4-ablation-assessment/001-M4-ablation-assessment.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/ablation-planner/2026-10-12_M4-ablation-assessment/001-M4-ablation-assessment.request.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/ablation-planner/2026-10-12_M4-ablation-assessment/001-M4-ablation-assessment.response.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/ablation-planner/2026-10-12_M4-ablation-assessment/run.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M1-source-review/001-M1-source-review.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M1-source-review/001-M1-source-review.request.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M1-source-review/001-M1-source-review.response.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M1-source-review/run.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M4-final-integrity/001-M4-final-integrity.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M4-final-integrity/001-M4-final-integrity.request.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M4-final-integrity/001-M4-final-integrity.response.md | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | review-tracing | .aris/traces/experiment-bridge/2026-10-12_M4-final-integrity/run.meta.json | review | 本地审查轨迹，不自动发布 |
| 2026-10-12 12:38 | experiment-bridge | AGENTS.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | CHECKPOINT.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | CHECKPOINT_20261012_123500.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | CORRECTION.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | CORRECTION_20261012_122600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | MANIFEST.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | MANIFEST_20261012_123500.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | PROTOCOL.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | PROTOCOL_20261012_120600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | REGISTRATION.json | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | REPORT.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | REPORT_20261012_122600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | REPORT_20261012_123300.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | analysis/MECHANISM_TRACE_20261012_123000.json | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | data/GENERATION.json | implementation | 冻结输入 |
| 2026-10-12 12:38 | experiment-bridge | data/observations.json | implementation | 冻结输入 |
| 2026-10-12 12:38 | experiment-bridge | data/private/truth.json | implementation | 冻结输入 |
| 2026-10-12 12:38 | experiment-bridge | data/tasks.json | implementation | 冻结输入 |
| 2026-10-12 12:38 | experiment-bridge | idea-stage/docs/research_contract.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | idea-stage/docs/research_contract_20261012_120600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | idea-stage/docs/research_contract_20261012_122600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/BUDGET.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/COVERAGE.csv | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/INPUTS.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/PAIRED.csv | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/PAIRED.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/REGISTRATION.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/RUN_STATUS.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/SOURCE_LOCK.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/SUMMARY.csv | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c001.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c002.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c003.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c004.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c005.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c006.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c007.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c008.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c009.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c010.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c011.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c012.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c013.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c014.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c015.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c016.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c017.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/checkpoint/c018.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/coverage.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/decisions.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/observations.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/outers.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/results.csv | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/results.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/selector_timing.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/summary.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/tasks.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/timing.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/verification-premature-attempt/VERIFY.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | outputs/confirmation-v1/verification/VERIFY.json | implementation | 冻结正式输出/原复核记录 |
| 2026-10-12 12:38 | experiment-bridge | prepare_confirmation.py | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/ABLATION_ASSESSMENT.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/ABLATION_ASSESSMENT_20261012_123300.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_PLAN_20261012_120600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_PLAN_20261012_123800.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_RESULTS.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_RESULTS_20261012_123300.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20261012_120600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20261012_122600.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20261012_123800.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/M0_M1_ACCEPTANCE_20261012_121900.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | refine-logs/M0_TESTS_20261012_121400.md | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | run_confirmation.py | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | test_confirmation.py | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | test_verify_confirmation.py | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | verify_confirmation.py | implementation | 实现、计划、分析或版本文档 |
| 2026-10-12 12:38 | experiment-bridge | MANIFEST_20261012_123800.md | implementation | 实现、计划、分析或版本文档 |

