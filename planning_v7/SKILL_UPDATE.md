# 研究 Skill 的小幅更新

2026-09-12，liekkas。已修改实际安装目标：
[map-denoise-research/SKILL.md](/home/grf/.codex/skills/map-denoise-research/SKILL.md)。

仅两处新增决策指引：

1. 关联oracle需要说明观测范围/权重/拟合点/支持的变化，以及拆组导致的参数数与秩变化。
   保守拆组是诊断例子，不是最终算法必须采用的架构。
2. 允许弃权的算法要区分减少损害、没有修改和实际恢复。支持mask不是移动mask，
   方向限制只在有助于解释时检查，不增加形式证明前置条件。

[证据记录](/home/grf/.codex/skills/map-denoise-research/references/evidence.md)已追加本项目V6的
触发、原始运行位置、决策与可能修正判断的证据。旧项目的未验证假说没有被误标成已验证。
未新增“禁用PCA”“必须共享”或“应放宽门限”等算法禁令。

Skill编写技能用于限制修改范围；地图去噪研究技能用于把经验连到实际决策。
原description、调用策略、agents/openai.yaml、geometry_checks.md、checks.py保持原样。
未修改其他AI4S技能，没有把一次领域实验推广成全部技能包的结论。

完整旧版本备份：[before](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/skill-update-v6-aw7utK5q/before)。
更新后的快照、前后哈希和[校验记录](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/skill-update-v6-aw7utK5q/VALIDATION.json)保存于同一个skill-update目录。
五个包文件中仅上述两文件变化，其余三个逐字节一致；当前安装文件也已与更新后快照核对。

格式检查通过。它证明文件可用，不证明研究效率或成果率已提高。独立审查使用的是
已公开经验，只能作内容审查；本轮未运行新旧Skill的前瞻能力对照。
内容审查提出的两处执行歧义已补齐：条件后级留出不冒充完整管线留出；逐站独立
拟合与跨站共享的总容量不同，不把其差异全部归因于关联。
若后续测试Skill效果，应使用未用于编写规则的新任务，相同输入/模型/预算，
不在提示中给出预期诊断，观察实际决策与产物，而不是是否复述新增措辞。

[下一轮实验计划](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/planning_v7/EXPERIMENT_PLAN.md)
已经按这两处更新写出，但没有启动V7算法或新的确认运行。
