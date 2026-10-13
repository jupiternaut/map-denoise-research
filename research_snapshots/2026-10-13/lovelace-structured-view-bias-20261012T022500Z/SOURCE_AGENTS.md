# 结构化视图偏差：本批职责

language: zh
唯一工作host：liekkas；唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z。
用户目标：保留有效几何、修复不足。本批执行上轮提出的C+D：共享视图偏差表示与简单公开背景校准；不重跑A+B。
只读PROTOCOL.md及它明确指定的旧依赖。所有其他lovelace根只读。验证本机hostname和绝对路径；禁止SSH、换机、GPU、API、依赖安装、发布和部署改动。
本地源码只用apply_patch编辑。solver作者仅写structured_solver.py/test_structured_solver.py；核验作者仅写verify_structured.py/test_verify_structured.py；root负责runner、协议、集成、分析、报告。无其他文件写权限。
非作者审查代理仅可写analysis/SPEC_REVIEW.md和refine-logs/EXPERIMENT_CODE_REVIEW.md，其他文件只读；不自行运行正式批次。
正式launch后本根科学.py与PROTOCOL.md冻结，后处理只在analysis/；任何科学修复另立版本，不覆盖已运行工件。
求解只接公开图像、裁剪类型、camera radius及注册参数；不接实际偏差、颜色、深度、生成分支、case、seed或初态。
公开背景掩码必须在全depth、全branch、允许camera误差范围上用几何外包证明背景，不能用GT选位置。校准不增加新采样，只读同图像。
所有动作落盘读回、计数完整之后再解析私有真值评价；KEEP与INCOMPATIBLE分开。历史外包复用明确计数，不冒充新确认。
本批不改动作公式、不扩候选或LP、不加学习器、不换第二生成器。实际结果和机械核验分开汇报。
