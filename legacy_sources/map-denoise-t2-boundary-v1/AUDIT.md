# 本轮基本执行检查

2026-09-11，liekkas。

- `test_common.py`：4项通过。
- `candidate/test_candidate.py`：3项通过。
- `alternative/test_split_consensus.py`：5项通过。
- 统一开发3249输出、复测396输出：保存输出后评估，0执行异常。
- Oxford真实接入18输出：有限、保点、原输入hash前后一致。不是精度核验。
- CLI默认fast：在复测768点输入001-73013执行并另存成功。
- 不将倾斜时方向不统一的原始bias参数RMSE用于方法排名。
- 旧检查点20260911-1404-pre-paper/research-state.tar.gz SHA256通过。
- 旧检查点20260911-six-track-v1/six-track-source-results.tar.gz SHA256通过。

这些是程序与产物的基本复核，不是形式完备性研究；研究主产物是滤波器及对应几何效果。
