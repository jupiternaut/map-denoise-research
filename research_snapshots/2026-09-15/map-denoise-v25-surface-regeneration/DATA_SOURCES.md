# 只读来源和坐标契约

主机必须是 `liekkas`。以下路径已在任务准备时核验；仍须执行 preflight.py。
不要广泛扫描其他研究目录。不得把本索引中的激光参考直接加入算法输入对象。

## 旧工程（只读、按需读取）

`/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`

首先读取：

- `field_budget_v24/REPORT.md`：同预算场比较与无稳定 alpha 增量。
- `loss_alignment_v23/REPORT.md`：简单照片选择失败与相机修复。
- `reconstruction_v22/REPORT.md`：原重建/独立参考来源和评价范围。

可复用模块：

- `loss_alignment_v23/calibration.py`：COLMAP 二进制解析。
- `loss_alignment_v23/run.py` 的 `setup()`：**只参考相机/尺度/重投影检查**。
- `reconstruction_v22/operator.py`：旧点移动对照，不是新再生实现。
- `published_outputs_v2/eval_reference.py`：核对读取与官方掩膜语义，不能直接沿用旧评价行。

不要运行旧 prepare_images.py/prepare_calibration.py/run.py：旧输出写入、下载重复和主程序副作用须避开。
复制或改写必要适配器到新工作区，记录源文件哈希和改动说明；不能污染旧环境。

## 算法可见的本地观测

公共数据根目录（复数 datasets）：
`/srv/slam-research/grf/map-denoise/datasets`

| 内容 | scan24 | scan37 |
|---|---|---|
| 照片目录，各 49 张 | `loss-alignment-v23/scan24/image` | `loss-alignment-v23/scan37/image` |
| COLMAP | `loss-alignment-v23/scan24/sparse/0` | `loss-alignment-v23/scan37/sparse/0` |
| 旧归一化/物理尺度 | `real-closure-v21/cameras_geosvr_linked.npz` | `reconstruction-v22-scan37/cameras.npz` |
| 旧重建网格 | `published-outputs-v1/scan24_mesh.ply` | `reconstruction-v22-scan37/scan37_mesh.ply` |

COLMAP 文件为 `cameras.bin`、`images.bin`、`points3D.bin`，照片按文件名对应，不靠任意排序猜相机。
新适配前核对 49 张数量、文件名、宽高、针孔相机类型，不能静默解释未知模型。

## 坐标转换（以实际文件复核）

- 作者网格/稀疏点使用归一化坐标；物理几何为 `p_phys = scale_mat * p_norm_h`。
- 物理坐标投影：`uv ~ P_colmap * inverse(scale_mat) * p_phys_h`。
- 使用 COLMAP 像素内参。`camera_mat` 不是像素内参，不能直接取出投影。
- 不用独立激光参考做 ICP、位姿修正或挑尺度。
- 裁图/缩放后同步更新像素坐标/内参；检查 round-trip 和稀疏轨迹。
- V23 已测稀疏重投影中位/P95：scan24 0.356/1.794 px；scan37 0.441/1.696 px。
  新实现应复核同样口径，P95>5 px 先排接口错误；通过不等于毫米几何正确。
- 范围/单位报告清楚，输出统一为物理 mm；不要再把 `.ply` 默认假定为米。

## 评价端专用（不得进入 fit/selector）

| 内容 | 绝对路径 |
|---|---|
| scan24 激光参考 | `/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference/stl024_total.ply` |
| scan24 mask | `/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference/ObsMask24_10.mat` |
| scan37 激光参考 | `/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37/stl037_total.ply` |
| scan37 mask | `/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37/ObsMask37_10.mat` |

已记录参考 SHA256：

- scan24：`963f2893d40d72957acdeb0affcd8aaa888408b180f5a62b62af747554ba4665`
- scan37：`dcd290f8d6bee24b51fa6df7d0fa3cf017e2c46d9edea0897dc935af2cc56c55`

本目录分离是编程约束而非沙箱。评价端能读参考不意味着构造端应读取它。
未来若要声称强隔离，应另外实施宿主权限；本轮不以搭沙箱阻止真实几何实验。

## 旧运行（只读）

- `/srv/slam-research/grf/map-denoise/runs/reconstruction-v22-qayc8gft`
- `/srv/slam-research/grf/map-denoise/runs/loss-alignment-v23-9g6_i29x`
- `/srv/slam-research/grf/map-denoise/runs/field-budget-v24-uvg3jss3`

相机与照片清单可读 V23 的 `scan24_PROVENANCE.json` / `scan37_PROVENANCE.json`。
V22 缓存评价有 out[keep]、out−q；不适用于变点数输出。
要用旧方法比较新 ROI，应在新工作区重建该 ROI 的旧方法输出；不要拼接不同 ROI 的均值。

## 环境准备时发现

- `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python` 存在，Python 3.12.13。
  可只读执行已装功能；不向该环境 pip install。
- `/home/grf/.local/bin/uv` 存在，可在本工作区建立 `.venv`。
- 当前 PATH 未发现 colmap/OpenMVS 命令，这不证明全机绝无安装。有限定位后再决定独立安装。
- 12 个逻辑 CPU；RTX 3060 Ti 8 GiB。准备时 GPU 仅约 908 MiB 空闲，不是运行时承诺。
- 准备时 /home 可用约 37 GiB。按 AGENTS.md 控制下载和本工作区体积。

## 官方来源入口（使用前核验版本、许可证、下载大小）

- GeoSVR 作者实现：<https://github.com/Fictionarry/GeoSVR>
- 作者现成网格：<https://huggingface.co/Fictionary/GeoSVR/tree/main/meshes_complete/DTU>
- DTU 官方数据入口：<https://roboimagedata.compute.dtu.dk/>
- COLMAP 官方实现：<https://github.com/colmap/colmap>
- COLMAP dense tutorial：<https://colmap.github.io/tutorial>
- Pixelwise View Selection 原文：<https://demuc.de/papers/schoenberger2016mvs.pdf>

只为当前实现补必要文献，不先写综述。官方入口能打开不等于具体文件已获取。
新场景按输入/来源可用性选，下载进工作区 downloads；来源、许可、字节数、哈希写 manifest。
