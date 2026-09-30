# Checkpoint

目标：保留有效几何、修复错误几何；本轮同像素补成熟MVS候选。

执行完毕：15个官方CUDA调用；两旧场景、四ROI、512固定查询；两个MVS主臂；scan24独立重复；全预测先封存再评分。

主结果：原497有效点，四ROI等权MSE CPU150.1211→Geo+fallback35.9203mm²，改善76.07%；4/4 ROI改善。75旧池难点Geo救回51个至≤5mm；原341个≤1mm点有23个改坏出界。Photo+fallback MSE261.3250，不能整体替代。

默认部署未改；未push GitHub或发布GitBook。本目录全部成果为本机开发实验。下一步不是继续调本轮阈值，而是冻结成熟基线做独立场景确认，后续自研增量围绕剩余缺失与误改评价。

汇报入口 REPORT.md；协议修订 ADDENDUM_PRE_RUN.md；可复现命令 COMMANDS.md；结果 evaluation/；图 figures/；独立核验 review/。旧目录只读。
