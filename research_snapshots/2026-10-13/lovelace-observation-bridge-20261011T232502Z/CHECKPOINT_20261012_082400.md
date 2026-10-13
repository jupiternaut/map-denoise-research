# Checkpoint：A+B完成

2026-10-12，Asia/Shanghai；host liekkas。
唯一工作根：`/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z`。

## 当前成果

八配置432偏移+54正确初态的实验已完成，全部7776动作先保存读回后评价。新shared2916、历史shared972、KEEP3888。M0 37 tests、六配置冒烟、非作者M1、正式M2、独立核验M3、配对分析M4以及非作者结果算术复核均完成。无后台实验。

核心科学结果：窄界raw115修复、窄界裁剪99、宽界无偏差0、宽界偏差raw48/clip32；同图放宽界使原修复全退KEEP。所有配置正确初态与负控全部KEEP、真深度全部覆盖、零损伤。8→16无动作/外包增量；4→8一个偏移终点改善。

完整报告为REPORT_20261012_081900.md；结果入口analysis/analysis-v1/SUMMARY.json；完整机械核验outputs/bridge-v1/verification/VERIFY.json=PASS；独立结果解释analysis/RESULT_INTERPRETATION_REVIEW.md。

## 冻结与核对

正式launch后哈希保持：

| 文件 | SHA-256 |
|---|---|
| AGENTS.md | d61195de340080080384860b3fb6f985dce70fc1f3d3637707879b80034b519b |
| PROTOCOL.md | 2e2e3d929cb681d41341bc71c1eb42005aeb9a0f8770d03783eaa960d07d6a21 |
| run_bridge.py | 7926aba59cc59c13016aec9eb436e46e3d80314561217241d9458fc8655e85d4 |
| test_run_bridge.py | 4ce2f0c76c6796d028603eb125cb86360e7532ce9dac8639c9191e61f1ad454d |
| verify_bridge.py | d3a1404814abf39384101e355dd92904eff3edac7ab66f1f960bc7f7c50c773a |
| test_verify_bridge.py | 5640b38ab977d95f3c7b2a8ae99e858204e785806ca8d345c18ebd64e194ad5d |

M3核对63源和26历史输入。主运行及核验总1463.093064秒，在实际07:52:06启动的原始5400秒窗口内完成；前置测试及后处理不混入solver耗时。SOURCE_LOCK、INPUTS和原始日志保留。

## 边界与一次执行异常

本批仅同54源合成机制回放，未做真实MVS或跨场景确认；没有部署、发布、GPU、付费API或依赖安装。旧实验根未编辑。

本根要求无SSH。结果解释代理如实报告最初误尝试`ssh liekkas`，因host-key失败未执行远端命令；之后hostname确认本机liekkas，全部实际读写在确切本机根，无再SSH。该失败尝试违反本批无SSH的操作边界，不将它隐去或报告成成功远端执行。主代理与作者执行不依赖该尝试，成果身份仍由本机hostname和绝对路径核对。

## 下一轮尚未执行

候选C+D：显式共享未知视图偏差，保留原独立噪声界及裁剪单侧语义；与同信息、同预算的简单公开背景校准比较。不使用真实偏差反校正、不用真值挑背景；沿用正确KEEP、负控KEEP、覆盖与新增修复验收。

尚未写入科学实现、启动新求解或获得该构造的收益；不把其预期登记为成果。LP、新学习器、候选扩张、模式外推和发布均不属本批。下一会话从此检查点开始，避免重跑A+B或重用旧输出路径。
