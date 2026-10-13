# A4-V 检查点：完整矩阵完成，主终点无增量

时间：2026-10-12 19:22:11 Asia/Shanghai；host liekkas。  
新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1。父阶段、appearance-1与既有工件只读。

## 已完成

- 36观测、324任务、1944方法动作＋324KEEP；144新冷求解、72封存复用。
- 固定η=0/1/10，bounded两档同用δ=1/10；无结果驱动的界值修改。
- 49/49准备测试，fresh同族临时语义审查PASS。
- 共同3600秒窗口内完成生成＋主求解＋独立核验，1100.096秒；阶段COMPLETE、核验PASS。
- 原始输出位于outputs/variation-v1；报告REPORT.md与时间戳报告同内容；数学说明refine-logs/MODEL_NOTE.md。
- 正式源启动后未改；129项SOURCE_LOCK与21项INPUTS完整核对。

## 已测判断

主终点η=1/10第三深度40偏移：bounded=independent，cal修复0/40，joint修复1/40。  
次要全144偏移joint：bounded13/144，independent12/144；3更好/141相同/0更差，MSE下降0.9180%。  
非零bounded/independent两方法覆盖18/18；strict仅3/18、2/18且15/18、16/18不兼容。  
全部配置正确初态18/18 KEEP，无相对初态损伤。  
72外包中50收窄；非零主40中cal40/joint39个错误初态仍在bounded凸包内。

## 不应重复或误述

- 不再重跑此矩阵、调δ或用次要终点代替主终点。
- 18旧来源配对，不是324独立场景或真实迁移。
- strict失去覆盖未导致本批次实测损伤，多数是KEEP回退。
- 非零18张图片均改变，原negative不是未变观测负控。
- 必要外包包含初态，不证明其物理可行；机械PASS不等于科学主假说成立。
- 已知设计界不等于真实外观界估计；动作保证只针对模型条件下的深度平方误差。

## 下一候选，尚未执行

冻结数据、δ、几何预算及行动，比较逐变量盒传播与联合线性多面体可行性。  
看能否用可独立核验的不可行证书排除相关性伪解，同时保持真实参数见证，再增加主40修复。  
若仍可行，再转向观测辨识或必要约束不足；不直接调阈值。

无GPU/API/SSH/install/发布/部署/后台循环，本阶段到此停止。

