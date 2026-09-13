# 外部实现与运行环境

实际主机 liekkas。新环境目录 `/srv/slam-research/grf/map-denoise/envs/spatial-v12-AXeJsv`，约363 MiB。
安装 PyMeshLab 2025.7.post1，没有修改旧 open3d-019 环境。正式实验由旧 Python 3.12 / NumPy 2.2.6 / SciPy
执行，运行时将新环境的 site-packages 追加到搜索路径，仅从中导入 pymeshlab。新环境另装的NumPy不是实验实际使用版本。

调用现成 `compute_mls_projection_apss` 与 `compute_mls_projection_rimls`，不是自研复刻。
基线定位和参数来自[官方过滤器文档](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#compute_mls_projection_rimls)。
APSS局部拟合代数球面；RIMLS面向尖锐特征保留。两者均使用观测估计的法向，不给真值法向。

适配流程：中心化并将米转为毫米；官方32邻居法向估计；按输入PCA参考方向统一法向符号；
官方16邻居半径估计；原样内核投影；转回世界米坐标；恢复共同支持之外的行。
默认参数外显：projectionaccuracy=1e-4、maxprojectioniters=15、maxsubdivisions=0；
filterscale=1/2；APSS sphericalparameter=1 / accuratenormal=True；RIMLS sigman=.75 / maxrefittingiters=3。
输入全局法向符号统一不适合所有闭合/多朝向表面，属于本轮明确适配假设，不能将该配置声称为每种场景的最佳配置。

首次300点纯平面测试，在 controlmesh 与 proxymesh 为同一对象时复现进程退出139。
加入显式半径估计后仍复现。参考[官方源码的对象复制分支](https://github.com/cnr-isti-vclab/meshlab/blob/main/src/meshlabplugins/filter_mls/mlsplugin.cpp)，
改为内容相同但ID不同的参考对象和投影对象，纯平面测试及后续全部批量输出均成功。
这里只报告本机现象，不宣称定位了通用上游bug；没有修改或重新编译库。

每点RGB编码来源行号，投影后核对颜色和点数，防止把重新排序当作对应误差。
核验报告记录 pmeshlab 扩展与 filter_mls 动态库SHA256。
这些是两个经典对口外部实现，不等于已完成最新学习式去噪/联合配准方法的全部强基线比较。
