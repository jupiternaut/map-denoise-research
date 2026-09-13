# 下一轮交接

## V2 优先说明（覆盖下方历史交接中的旧评分结论）

当前报告和数据入口：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/repair-v2-oyuie4pl/`。

1. 真实六片区仍读原 `patches/`，未改。合成必须读当前运行的 `synthetics/`：已修 `xyz_local` 与位姿不一致；算法实际读取的世界坐标和帧号与旧版完全相同。
2. 用 `evaluate_v2.py`，不再使用 `evaluate.py`。自报 K 与几何分数分开。不要引用 V1 的 identity 全误拆、ICP 零误拆零误并排名。
3. `RESULTS_V2.csv` 含 444 个复用输出的复评和 336 个当前输入坐标系新输出。旧参考坐标系只作诊断。780 行并非 780 个独立案例。
4. 旧 fast 已有帧截距/偏差变量。无扰动时改写已经约 4 cm，不能把这个量归因于注入的几毫米误差。当前输入坐标系下 fast 的平移/平移+旋转自身响应约 4.79/4.83 mm，不再支持先前 3.09→5.02 mm 的强旋转归因。
5. 下一轮仍保留三端构造，但先在同输入上分清局部平行表面近似、法向估计与真正新增求解变量的作用。盲估计未实现，新的倾斜/关联算子未实现，独立真实几何收益未测得。
6. 执行入口为 `repair_pilot_v2.py`；每次唯一目录，旧 CSV/输出/片区拒绝覆盖。协议、源码快照、哈希与核验记录随新运行保存。

以下原始清单保留供定位来源，旧合成/旧评价目录只用于历史审计。

主机 `liekkas`。大文件已经在 `/srv`。不要重下 ETH3D，不要改旧检查点。

## 下一轮直接读这些文件

数据准备不用重做：

- ETH3D 压缩包和哈希：`/srv/slam-research/grf/map-denoise/datasets/multiscan-pilot-v1/eth3d/downloads/`，记录在 `runs/.../logs/eth3d_download.json`
- 已解包 raw/clean：`.../eth3d/extracted/{courtyard,delivery_area}/{raw,clean}/`
- 片区 NPZ/JSON：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/patches/`（6 块都可用：da_wall / da_junction=同轴第二板 / da_thin / cy_wall / cy_junction / cy_thin）
- 抽取报告：`.../patches/EXTRACT_REPORT.json`
- 27 个合成案例：`.../synthetics/`，预留种子不要动
- 统一读写：本目录 `schema.py`、`eth3d_io.py`、`adapt.py`、`perturb.py`
- 预实验协议和结果：`PILOT_PROTOCOL.md`、`PILOT_RESULTS.csv`、`runs/.../pilot/`
- UCL 阻塞证据：`logs/ucl_probe.json`，`datasets/.../ucl/downloads/BLOCKER.json`
- TUM 元数据：`datasets/.../tum_metadata/TUM_TLS24_METADATA.json`
- GPU 小张量记录：`logs/gpu_check.json`
- 旧算子只读：`/home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1/operators.py`

## 三条算法线都还在

同一批案例接着做，不要因为旧方法输赢删掉某一端。

- 简单端：`y_i = mu[k_i] + b[frame_i] + epsilon_i`，联合估计残差噪声、帧偏差和分层。
- 桥接端：再加同帧共享局部倾斜 `a[frame_i]^T u_i`。
- 复杂端：局部三维表面、扫描位姿和射线可见性一起建。

输入接口已经是带 `scan_id` 的片区。估计器只能看见点、帧和提供的 σ。参考位姿、扰动逆变换、层标签在 `evaluation/`。

## 还缺什么

- UCL E57 / IFC：官方 OneDrive 要登录，本轮没有文件。
- TUM 站点子集：只有链接，没有点云。
- 独立真实几何：本轮尚未测得。不要用 clean 或 BIM 顶替。
- 噪声盲估计、倾斜项、新的关联求解器：本轮都没写。
- 第三站重叠：ETH3D 这两个场景实际只有 2 站。
- GPU 算子：只做了小张量核验，没有 CUDA 滤波核。

## 本轮没跑的

- TUM 任何下载
- UCL 镜像或账号下载
- 全场景优化
- 在预留合成种子上调参
- 旧环境安装 pye57
- 显卡驱动更换
