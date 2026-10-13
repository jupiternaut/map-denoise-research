# 本批职责与边界

language: zh
确切host liekkas；新工作根 /srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z。
只读本根PROTOCOL.md以及它明确列出的历史依赖；所有旧lovelace目录不可编辑。
本批执行A+B（裁剪/宽界桥接与传播轮数），不擅自上C/D/E/F、机器学习、LP或新物理模型。
本地代码编辑用apply_patch。无SSH、GPU、API、依赖安装、GitHub/GitBook发布或部署改动。
运行作者只改run_bridge.py/test_run_bridge.py；核验作者只改verify_bridge.py/test_verify_bridge.py；主代理负责协议、集成、分析、报告。根科学.py和PROTOCOL在正式运行锁定后不可编辑；后处理分析在analysis/。
正式solver接口不接真值/偏差实际值/模板/种子/unit_id。所有7776动作保存读回后再读私有事实评价。严格区分KEEP与INCOMPATIBLE坐标回退。
原始复用与新求解分开计数，不静默替换目标/源版本。
