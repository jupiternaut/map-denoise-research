# 本轮范围

Host: liekkas；本目录是新实验，不是历史目录的替代副本。
language: zh
只读依赖：`/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z`。
该目录的 source_snapshot 来自 GitHub 提交 bd8d05aad65e2c55cd9cd288beae4ef949606984。
不修改历史文件、GitHub、GitBook 或部署默认；不运行 GPU。
真值只用于生成观测和终点评价，不进入选择器。

Root: hybrid.py、协议、计划、报告。runner_impl: run_hybrid.py、test_run_hybrid.py。
hybrid_theory: THEORY.md、test_hybrid.py。hybrid_verifier: verification/ 下的独立核验。
所有作者文件用 apply_patch。不同 agent 不修改其他分工文件。
实验源在确认前冻结；所有结果保存到新的 outputs/confirmation-v1。

## Pipeline Status

implementation: complete
confirmation: complete
verification: PASS

确认工件：outputs/confirmation-v1。一次正式运行，未调参，未修改科学源。
部署默认仍为 identity；此目录完成不等于真实点云迁移已完成。
