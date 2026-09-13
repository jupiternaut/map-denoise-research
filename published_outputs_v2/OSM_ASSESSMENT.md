# OSM 数据如何获得，以及本轮重建结果是否可导入

核对日期：2026-09-12。OSM 按 OpenStreetMap 理解。本轮没有向 OSM 上传、创建账号、申请导入或联系社区。

## 数据是什么，如何取得

OSM 的基础对象是带经纬度的节点、道路/建筑轮廓等线面、关系及标签；不是密集 Gaussian PLY 或扫描网格。
数据来自贡献者实地测绘、GPS 记录、许可允许的影像描绘和经过审核的外部地理数据导入。

- 小区域：Overpass Turbo 查询建筑、道路等要素，可导出 GeoJSON 或 OSM 数据。
- 城市/区域：Geofabrik 下载区域 `.osm.pbf` 或其提供的 GIS 图层。
- 三维建筑：可用 footprint、height、building:levels、building:part、roof:shape 等属性构建简化几何；属性缺失时展示器可能采用默认值，不能把默认高度当实测高度。

“OSM 建筑看起来平整”不等于其细部测量优于摄影测量：规则化几何本来就没有表示所有门窗、薄构件和表面细节。

## 当前不能声称达到 OSM 验收标准

OSM 没有一个对所有点云通用的“Chamfer 小于 x mm 即验收”的标准。官方导入要求关注资料准确性、来源可核验、许可兼容、标签转换、与现有数据融合和社区审查等。

|本轮数据|当前状态|
|---|---|
|room Gaussian 模型|局部室内重建；未建立经纬度/地理定位，不是可直接上传的 OSM 要素|
|DTU scan24|受控物体场景；可测局部几何，不是地理建筑对象的验收样本|
|Courthouse mesh|具备建筑场景，但本轮没有核验地理配准、提取建筑轮廓/高度、与现有 OSM 要素消重|
|许可|论文代码或结果公开不自动等于底层测量数据具有 ODbL 兼容的导入许可；本轮没有完成授权链审计|
|社区导入流程|未开展，用户没有授权上传|

毫米级局部表面误差不能证明绝对地理定位准确；大模型内的局部片区也不能代表整栋建筑精度。
当前结论应是“尚不具备宣称可导入 OSM 的证据与交付形式”，不是“已证明精度一定低于 OSM”。

研究上可将 OSM 作为外部位置/建筑轮廓先验，但不能把其所有轮廓、高度当作独立精密几何真值。
不应为了通过一个并不存在的统一 OSM 点云门槛，再把地图去噪课题扩成完整制图生产线。

## 官方依据

- [OSM 初学者指南](https://wiki.openstreetmap.org/wiki/Beginners%27_guide)
- [OSM 数据精度讨论](https://wiki.openstreetmap.org/wiki/Accuracy)
- [可核验性原则](https://wiki.openstreetmap.org/wiki/Verifiability)
- [导入指南](https://wiki.openstreetmap.org/wiki/Import/Guidelines)
- [OSM 版权与许可](https://www.openstreetmap.org/copyright)
- [Overpass Turbo](https://wiki.openstreetmap.org/wiki/Overpass_turbo)
- [Geofabrik 区域数据下载](https://www.geofabrik.de/data/download.html)
- [Simple 3D Buildings](https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings)
