# 数据准入报告

主机 `liekkas`，2026-09-12。片区按几何和重叠选，不按方法分数选。

## ETH3D：完成

官网只下了 `delivery_area` 和 `courtyard` 的 training laser scans，raw + clean，大约 1.44 GB 压缩包。链接来自 https://www.eth3d.net/datasets，许可写的是 CC BY-NC-SA 4.0。本地 SHA256 记在 `runs/.../logs/eth3d_download.json`，这不是官网签名。

每个场景正好 2 站：`scan1.ply`、`scan2.ply`，外加 `scan_alignment.mlp`。PLY 停在扫描仪局部坐标，世界坐标只乘一次 MLP 位姿。位姿检查过：齐次底行、旋转正交、行列式接近 1。0.1 m 体素里最多 2 站，没有第三站重叠。

clean 和 raw 是同一扫描仪来源。clean 是清理版，不是另一台设备测出来的毫米真值。点序和点数都对不上，不能按下标配对。

本轮片区用 clean。墙板来自重叠区的竖直直方图，不是方法输赢。没有独立薄层真值，所以不编薄层标签。`delivery_area` 没有够尖的正交第二峰，`da_junction` 写成同一墙上的第二条局部板（1591 点，800/791，约 0.30×2.12×4.61 m），字段是 `same_axis_second_slab`。`courtyard` 的 `cy_junction` 是正交交叉框（1600 点）。`cy_thin` 有两站，但 385/29 偏得很厉害，点数 414，仍在 300–3000。六块都可用。

## UCL Basic Corridor：失败（登录墙）

官方页 https://indoor-bench.github.io/indoor-bench/ 只有 OneDrive：`https://1drv.ms/u/s!AuBuZXi2cbwhnCe-A4jdfGqHRgzM`。匿名 Graph/OneDrive 接口返回 401 或 share id 格式不对；无 cookie 跳到 `login.live.com`。没有公开直链 zip。

证据：`logs/ucl_probe.json`，`datasets/.../ucl/downloads/BLOCKER.json`。

没有下 E57，没有编站号，没有用不明镜像。旧 Open3D 环境里没有 pye57，也没有给旧环境加包。IFC 即使以后拿到，也只是人工建模，论文还写了不少厚度是任意设的，不能当独立毫米真值，也不能当评分器。

## TUM-TLS-24：未执行下载

只核了官网 https://tum2t.win/datasets/pc-tls 的元数据。页面分开给了 subsampled、original、完整 TUM-TLS-24。完整包大约 95 GB，本轮禁止下。元数据在 `datasets/.../tum_metadata/TUM_TLS24_METADATA.json`。以后若用，要重新核版本、站号、跨模型 Z 偏移 0.7551 m 和许可。注册残差不能当成独立绝对精度。

## 合成对照：完成

27 个精确案例已经生成，目录 `runs/.../synthetics/`。三种子 912101 / 912113 / 912127。预留种子 912201 / 912211 / 912223 本轮不用。目标每例 8 站、768 点，σ = 1 mm。有单层重影、双层 2/4/8 mm、以及两种物理解释分不开的对照。前层不透明，后层不会无条件穿出来。层标签只在 `evaluation/`。

## 真实双层证据

ETH3D 这两个场景只有 2 站重叠。重叠直方图能切出墙板和局部板，不能证明存在独立测得的真双层。合成双层是构造出来的，不是现场第二层。本轮没有独立真实几何评分。
