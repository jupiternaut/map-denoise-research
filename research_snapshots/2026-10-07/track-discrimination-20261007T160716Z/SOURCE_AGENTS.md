# 三视图轨迹与修正方向判别

目标主机 liekkas，唯一写入根为本目录。旧运行与 GitHub 工作树只读，不上传、不部署、不改历史文件。手写代码及文档使用 apply_patch。
CPU-only；使用 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python，BLAS线程设为1。禁止安装依赖或读取私钥。
原始目标是保留有效几何、修复不足。本次为受控机制构造和两个旧场景开发回放，不是新场景确认。
分工：主代理维护 protocol/requests/integration/evaluation/report；理论代理仅写 theory/；观测代理仅写 observation/；审查代理仅写 audit/ 与约定审查输出。
观测提取只读 REQUESTS.json、PROTOCOL.json 和其中的原始照片，不能读旧候选、源深度图、激光、旧误差/决策。参考像素及成像域可使用；不在旧候选投影周围搜对应。没有匹配是未知，不是错误或遮挡的证据。
同样原始图像下比较星形参考—源对应和额外实测源—源对应约束；几何循环不能由待验候选的投影制造。
候选坐标固定，观测封存后方可接入。推理封存后方可打开旧评价表。参数在看到本轮评价前锁定，失败不追加阈值。
中点定理适用于同射线、同目标层深度平方损失。真实回放为最近激光顶点距离，不能反向标注轨迹身份，不声称物理层定理已实证。
私有审查prompt/response置于.aris/，不得公开。所有审查same-family/provisional。

## Pipeline Status
language: zh
