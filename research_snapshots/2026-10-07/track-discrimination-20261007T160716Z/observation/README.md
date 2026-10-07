# 原始图像轨迹提取封存

本目录提取器运行于 `liekkas`，只消费根目录 `REQUESTS.json`、`PROTOCOL.json` 及 REQUESTS 列出的六张原始照片。`LOCK.json` 在首次执行前锁定源码、配置、请求和最终协议；首次 synthetic 测试和真实提取均一次通过，没有锁定后修补。未读取旧候选坐标、源深度图、激光、旧评价或旧决策。

执行命令：

```sh
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python /srv/slam-research/grf/map-denoise/runs/track-discrimination-20261007T160716Z/observation/extract.py
```

匹配使用 PIL 灰度转换与 `(width,height)` 双线性缩放、K_half/R/center 相机模型、亚像素双线性采样的 9×9 未扭曲 patch NCC。每个固定参考像素沿完整公知参考相机 Z 域以 1 mm 间隔分别搜索两张源图。保留全部超过 NCC=0.6 的连通采样段，以及每段内全部局部极大值；平顶局部极大值使用平台中点。没有 top-k 截断。

随后以每个实测 source1 局部峰像素为固定射线和固定图像 patch，在 source2 中直接沿 source1-source2 极线扫描完整公知 Z 域。这里改变的是 source1 射线上未知位置，source1 观测像素保持固定；没有使用待评价候选的投影制造循环。直接匹配同样保留全部阈值段和局部峰。

主 `star_intervals` 使用参考到源的实测峰对、共同射线区间及三视图正深度/三角化/重投影检查。`cycle_intervals` 是同一 star 轨迹的子集，额外要求直接 source1-source2 匹配与原参考-source2 峰坐标相容，并用参考、source1、两项 source2 观测共同三角化。额外观测继续使用相同 1.5 px 重投影预算。直接搜索只有 source1→source2 一向，未要求反向循环。

`star_full_intervals` 保留所有参考-源阈值段相交的辅助集合。主 star 的有限峰表示及几何检查删除的辅助区间在每行 `diagnostics.full_star_intervals_not_represented_by_accepted_nodes` 中明示。`independent_union_intervals` 是源匹配并集辅助集合。

区间由采样连通段向外扩半步再加协议 1 mm padding，并与采样的像素容差射线支撑区间相交。该扩张是工程约定。1 mm 网格、局部峰代表、未扭曲 patch、像素容差和 padding 不保证连续对应空间的穷尽性或真实层覆盖率；这些不是校准置信集或完整物理可行集合。空集明确为 `empty_unknown`，不推断遮挡、原点正确或错误。

## 封存结果

21 个固定请求中主 star 非空 6 个，cycle 非空 3 个，辅助 full star 非空 6 个。保存了 90 个参考-源阈值段、164 个参考-源峰、77 次直接源-源完整扫描、493 个直接阈值段和 612 个直接峰。通过几何检查的主 star 轨迹 28 条，cycle 轨迹 13 条。5 个请求的 full star 部分区间未进入有限主 star 表示。cycle 始终包含于 star，代码逐行验证该不变量。

cycle 非空请求为 scan118/query117、scan122/query48、scan122/query60。这只是观测构造结果，不评价几何收益或真实层身份。

`TRACKS.json` 包含每条轨迹的观测像素、NCC、节点来源、区间、共同三角化位置、参考深度和每视图重投影误差。`SCANS.json` 保存每次扫描的完整 Z 网格、投影像素、NCC（非法采样为 null）和过阈值索引，可复算阈值连通段与峰提取。原始曲线文件约 7.8 MB。

`SELFTEST.json` 的五项检查全部通过：合成三相机三角化、全部歧义段与平顶峰保留、合成平面原图的直接循环覆盖已知深度、非法文件读被拒绝、候选字段被 schema 拒绝。`SEAL.json` 锁定输出和六张照片哈希，并记录提取/序列化用时 0.722 s（不含 Python/依赖导入）。

提取期间执行了 Python audit-hook 文件白名单和严格请求字段检查；非法候选/深度/评价路径的负测试确实触发 PermissionError。该控制不属于 OS 沙箱，也不能证明对恶意本机原生代码的绝对隔离。
