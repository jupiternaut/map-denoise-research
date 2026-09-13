# 官方 GICP 基础对照

2026-09-12 在 liekkas 核对安装版本 Open3D 0.19.0，并核对官方接口：
[registration_generalized_icp](https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_generalized_icp.html)、
[TransformationEstimationForGeneralizedICP](https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.TransformationEstimationForGeneralizedICP.html)。

实际调用官方 Generalized ICP；不替换其求解步骤，也不在其后附加自研点级滤波。多站包装是分别配准到点数最多的输入锚站，并非完整多路联合优化/JRMPC/BALM，更不宣称领域最强基线。

输入仍只有当前 XYZ（米）、scan_id 和点级 sigma（毫米）。初始化使用已粗配准世界坐标下的单位变换，锚站并列按 ID。库从输入点云计算所需局部几何，不输入答案法向或关联。预设 epsilon=.001，两个对应半径各30轮、相对fitness/RMSE收敛阈值均1e-6。

半径基准为 max(4*sigma, 全部站内正最近邻距离中位数)，粗/细半径为基准的3/1.5倍。它是输入密度适配参数，不是经过本数据真值调优的最优配置。未注册成功的站保持原样，不删除任何点。所有原始最终变换以及各阶段fitness/匹配数/RMSE保留。

为了避免纯平移案例被锚站坐标原点左右，最终仅对成功关联的分量施加一次**输入定义的**共同平移，使该分量总质心等于当前输入质心。没有对未扰动测量或GT作对齐。锚站的旋转规范仍然保留；结果可由保存的原始变换、原点和公共平移完整重建。配准误差、局部几何退化、部分重叠和锚站偏差仍可能影响结果。

本对照的任务是逐站刚性纠偏，不能消除每点独立噪声，不能与带局部投影去噪的算法进行功能无差别排名。没有后处理的原始结果全部评分，不因为表现差而追加GT驱动回退。异常必须在运行表中保留。
