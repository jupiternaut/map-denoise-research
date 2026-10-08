# 实验进度

| 阶段 | 实施状态 | 科学/验收状态 | 证据 |
|---|---|---|---|
| B0实现 | DONE | 46测试通过 | B0/RESULTS.json |
| B1标定 | DONE | 48场景；独立冻结标定 | calibration/CALIBRATION.json |
| B2旧回放 | DONE | PASS；旧对照完整保留 | replay/evaluation/GATE.json |
| B3新合成确认 | DONE_WITH_LOGGING_ERROR | FAIL：正确输入1/48小误移 | confirmation/evaluation/GATE.json |
| 算术复核 | DONE | 78/78 PASS | audit/FINAL_RECHECK_CORRECTED.json |
| 语义审计 | DONE | WARN，same-family/provisional | EXPERIMENT_AUDIT.md |
| 作图 | DONE | 三组PNG/SVG，输入未改 | figures/PROVENANCE.json |
| 真实数据/部署 | NOT_RUN | 无新增主张，默认不变 | REPORT.md |

DONE表示工件完整，不表示科学假说成立。B2评价进程也有相同的封存后日志异常，已保留失败收据。
