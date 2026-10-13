# 本轮范围

主机 liekkas；新实验目录，不修改历史实验、GitHub 或部署默认。
来源为 jupiternaut/map-denoise-research 的 bd8d05aad65e2c55cd9cd288beae4ef949606984。
source_snapshot 是该提交的只读研究副本，不能修改。

目标：保留有效几何并修复不足。本轮只检验小偏移、双层面积遮挡、同相机误差信息的对照。
不得用确认真值选择深度、阈值、应用位置或停止条件；真值仅用于生成观测和终点评价。
模型已知的形状、颜色、背景、尺度及误差预算都是显式先验，不宣称真实重建或文献首创。
language: zh

分工：root 负责方案、challenge.py、run_challenge.py、报告；layered_model_impl 只写 layered_model.py、test_layered_model.py、LAYERED_THEORY.md；projection_theory 只写 THEORY_CHALLENGE.md、test_projection_policy.py。
