# 未见噪声种子确认：职责与范围

language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-structured-bias-confirmation-20261012T035916Z。
目标：保留有效几何、修复不足；继续上轮提出的冻结clip确认，不重新设计算法。
旧根全部只读；不得编辑冻结structured_solver.py、动作器、历史几何或旧结果。不得SSH/换机/GPU/API/安装/发布/部署。
root负责prepare_confirmation.py、run_confirmation.py及其测试、协议、分析与报告。核验代理仅写verify_confirmation.py/test_verify_confirmation.py；审查代理只读全部文件、不运行正式批次、不写文件，由root保存回复。
用apply_patch编辑源码及文档。新主批启动后本根科学源码和PROTOCOL冻结；修复必须另立版本，不覆盖已跑结果。
只生成seed61004、原prefix shared-appearance-v1、六模板×三原深度的18个clip观测。首次读取新标签前写协议。生成过程可使用真值，求解只读公开图像/相机范围/裁剪类型；去掉所有额外calibration样本。
KEEP、calibrated、joint三个方法；36新冷求解、486动作；18正确、144偏移（含24等色全遮挡负控）每方法。不追加seed/模板/raw/偏差排列/LP/学习器。
runner必须全486动作保存读回、唯一计数后才解析生成标签评价。核验不同作者入口不导入新runner/冻结solver计算答案，复用旧独立核验必要行和历史几何并注明范围。
一次1800秒主批+核验绝对窗口，不重置；报告正负结果，不把平均收益当机械PASS要求。
