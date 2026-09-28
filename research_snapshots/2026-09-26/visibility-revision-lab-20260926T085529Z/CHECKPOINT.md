# 完成检查点

Host: liekkas。
Code: `/home/grf/Documents/Codex/2026-09-26/visibility-revision-lab-20260926T085529Z`。
Artifacts: `/srv/slam-research/grf/map-denoise/runs/visibility-revision-20260926T085529Z`。

完成六臂开发探索、24工作点锁定、三场景60案例回放；所有方法保留，共47评价臂/2,820行/120新点云。全部CPU，无安装、下载、部署或Git变动。所有旧目录只读。

四个开发目标都选择base_shallow。主方法均衡降幅（native/−1/+1/−3/+3）为−2.31362/3.99062/7.44184/43.90175/35.03828%；锁定恢复工作点为−7.67324/1.16394/10.03585/46.99438/39.02216%。它们是同一浅基线，不是新的可见性优势。

新证据的条件收益：相同浅模型/零阈值的+1 mm改善9.61075→12.54190%，代价是其他多数条件回退。事后最高+3为base_rich recovery的41.38534%，同时native恶化9.70410%。不能拼成一个假想赢家。

候选GT上界32.60195/38.87290/43.89709/69.94271/63.26696%，仅评价诊断。当前输入推算遮挡不是独立深度；下一轮可探索独立视角判别，不在本目录继续调参。

阅读REPORT.md、PROTOCOL.md、AUDIT.md、产物evaluation/SUMMARY.json。最终完整性由MANIFEST.json与VERIFICATION.json记录；运行`python -B verify.py`可只读复核。所有本轮后台计算已结束。
