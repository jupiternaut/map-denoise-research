# COLMAP 固定像素开发对照

- 目标主机 liekkas；本目录是唯一写入的实验目录，专用环境另在项目 envs/colmap-cuda12-421。
- 旧 camera-pairing-replay-20260930T162130Z 与 upstream-photo-holdout-20260930T113213Z 全部只读。
- 冻结修正后的相机、reference+4Q 照片、四个 ROI 的全部 512 个原查询像素。不要根据旧 GT 难点选择运行范围。
- 构造期间不读取激光 GT、历史 points3D 或 GT 衍生分数；先封存预测再评分。
- 使用官方 pycolmap-cuda12==4.2.1；不是自写近似 COLMAP。新建空 points3D.txt 仅用于官方格式。
- 不训练、不调评分阈值、不改部署默认、不发布或推送。C/H 照片不进入本轮 MVS。
- 缺失深度原位记录，不能换到邻近有效像素。报告覆盖及同点误差；fallback 是明确的方法，不是填补缺失的真值。
- 原始输入为主；旧 75 个所有候选误差超过 5mm 的位置只作封存后诊断分层。
