# 受控成像机制实验

## 目标与方法

使用解析针孔相机与不透明平面的最近正向交点渲染灰度图。世界纹理由连续正弦函数定义，像素采用3×3面积采样；没有从真值深度手工构造代价曲线。三个固定随机种子、十种场景、六种方法，共180次双源支持估计。单位为毫米。

六个方法共用父目录support.py。每个深度要求两个源视图同时通过NCC≥0.6，保留全部连通深度区间，区间扩展半网格步长+1mm。固定候选为450/540/600/660/900，原输出900；按支持区间长度均匀分布的均值选择最近候选，空支持保留原输出。真值只在渲染和观测/选择封存后的评估阶段使用。

隔离范围是函数接口与数据使用：score_support只接收图像、相机、中心像素、固定深度网格和方法，不接收真值或候选。外层score_all载入的FIXTURES.json也包含世界定义，但评分代码只使用图像路径与相机字段。因此这里不是“整个评分进程无法访问真值”的强隔离。SELECTIONS.json中的point_gain是到支持均值的绝对深度距离改善，单位mm，并非平方P风险改善值；在这一维固定候选实验中，最近候选与改善正负判定相同，但数值不可当作mm²报告。

环形对照保持中心平面、中心纹理和所有相机不变，仅将900mm背景由常量替换为纹理。其余负对照分别破坏纹理强度、纹理唯一性、源可见性、正平面假设和共同相机标定。

## 结果

表内“含真值/精确选择/空支持”均为三个种子的计数；非目标长度是离600mm超过10mm的支持总长度，按种子平均。

| 场景 | 方法 | 含真值 | 精确选择 | 空支持 | 平均选择误差 mm | 平均非目标长度 mm | 目标表面掩码比例 |
|---|---|---:|---:|---:|---:|---:|---:|
| single_plane_good | translation_full9 | 3/3 | 0/3 | 0/3 | 60.0 | 236.0 | 1.000 |
| single_plane_good | translation_center3 | 3/3 | 1/3 | 0/3 | 40.0 | 186.7 | 1.000 |
| single_plane_good | translation_connected9 | 3/3 | 0/3 | 0/3 | 60.0 | 238.0 | 1.000 |
| single_plane_good | plane_full9 | 3/3 | 0/3 | 0/3 | 60.0 | 253.3 | 1.000 |
| single_plane_good | plane_center3 | 3/3 | 1/3 | 0/3 | 40.0 | 203.3 | 1.000 |
| single_plane_good | plane_connected9 | 3/3 | 0/3 | 0/3 | 60.0 | 255.3 | 1.000 |
| ring9_flat | translation_full9 | 3/3 | 3/3 | 0/3 | 0.0 | 74.0 | 0.111 |
| ring9_flat | translation_center3 | 2/3 | 2/3 | 1/3 | 100.0 | 0.0 | 1.000 |
| ring9_flat | translation_connected9 | 2/3 | 2/3 | 1/3 | 100.0 | 0.0 | 1.000 |
| ring9_flat | plane_full9 | 3/3 | 3/3 | 0/3 | 0.0 | 72.7 | 0.111 |
| ring9_flat | plane_center3 | 0/3 | 2/3 | 1/3 | 100.0 | 0.0 | 1.000 |
| ring9_flat | plane_connected9 | 0/3 | 2/3 | 1/3 | 100.0 | 0.0 | 1.000 |
| ring9_textured | translation_full9 | 3/3 | 1/3 | 0/3 | 40.0 | 94.7 | 0.111 |
| ring9_textured | translation_center3 | 0/3 | 0/3 | 2/3 | 220.0 | 6.7 | 1.000 |
| ring9_textured | translation_connected9 | 0/3 | 0/3 | 2/3 | 220.0 | 6.7 | 1.000 |
| ring9_textured | plane_full9 | 3/3 | 1/3 | 0/3 | 40.0 | 142.0 | 0.111 |
| ring9_textured | plane_center3 | 0/3 | 0/3 | 2/3 | 250.0 | 18.7 | 1.000 |
| ring9_textured | plane_connected9 | 0/3 | 0/3 | 2/3 | 250.0 | 18.7 | 1.000 |
| ring25_flat | translation_full9 | 3/3 | 3/3 | 0/3 | 0.0 | 92.7 | 0.309 |
| ring25_flat | translation_center3 | 2/3 | 2/3 | 1/3 | 100.0 | 24.0 | 1.000 |
| ring25_flat | translation_connected9 | 2/3 | 2/3 | 1/3 | 100.0 | 4.0 | 1.000 |
| ring25_flat | plane_full9 | 3/3 | 3/3 | 0/3 | 0.0 | 92.0 | 0.309 |
| ring25_flat | plane_center3 | 2/3 | 2/3 | 1/3 | 100.0 | 25.3 | 1.000 |
| ring25_flat | plane_connected9 | 2/3 | 2/3 | 1/3 | 100.0 | 6.7 | 1.000 |
| ring25_textured | translation_full9 | 3/3 | 3/3 | 0/3 | 0.0 | 82.7 | 0.309 |
| ring25_textured | translation_center3 | 2/3 | 2/3 | 1/3 | 100.0 | 26.7 | 1.000 |
| ring25_textured | translation_connected9 | 1/3 | 2/3 | 1/3 | 100.0 | 2.0 | 1.000 |
| ring25_textured | plane_full9 | 3/3 | 3/3 | 0/3 | 0.0 | 81.3 | 0.309 |
| ring25_textured | plane_center3 | 2/3 | 1/3 | 1/3 | 120.0 | 40.0 | 1.000 |
| ring25_textured | plane_connected9 | 1/3 | 2/3 | 1/3 | 100.0 | 8.0 | 1.000 |
| source_occlusion | translation_full9 | 1/3 | 0/3 | 1/3 | 170.0 | 36.0 | 0.889 |
| source_occlusion | translation_center3 | 2/3 | 0/3 | 0/3 | 90.0 | 192.0 | 1.000 |
| source_occlusion | translation_connected9 | 1/3 | 0/3 | 1/3 | 170.0 | 36.0 | 0.887 |
| source_occlusion | plane_full9 | 2/3 | 0/3 | 1/3 | 140.0 | 38.7 | 0.889 |
| source_occlusion | plane_center3 | 2/3 | 0/3 | 0/3 | 90.0 | 218.7 | 1.000 |
| source_occlusion | plane_connected9 | 2/3 | 0/3 | 1/3 | 170.0 | 40.7 | 0.887 |
| weak_texture | translation_full9 | 0/3 | 0/3 | 3/3 | 300.0 | 0.0 | 1.000 |
| weak_texture | translation_center3 | 0/3 | 0/3 | 3/3 | 300.0 | 0.0 | 1.000 |
| weak_texture | translation_connected9 | 0/3 | 0/3 | 3/3 | 300.0 | 0.0 | 1.000 |
| weak_texture | plane_full9 | 0/3 | 0/3 | 3/3 | 300.0 | 0.0 | 1.000 |
| weak_texture | plane_center3 | 0/3 | 0/3 | 3/3 | 300.0 | 0.0 | 1.000 |
| weak_texture | plane_connected9 | 0/3 | 0/3 | 3/3 | 300.0 | 0.0 | 1.000 |
| periodic_repeats | translation_full9 | 3/3 | 0/3 | 0/3 | 60.0 | 276.7 | 1.000 |
| periodic_repeats | translation_center3 | 3/3 | 0/3 | 0/3 | 60.0 | 230.7 | 1.000 |
| periodic_repeats | translation_connected9 | 3/3 | 0/3 | 0/3 | 60.0 | 287.3 | 1.000 |
| periodic_repeats | plane_full9 | 3/3 | 0/3 | 0/3 | 60.0 | 285.3 | 1.000 |
| periodic_repeats | plane_center3 | 3/3 | 0/3 | 0/3 | 60.0 | 229.3 | 1.000 |
| periodic_repeats | plane_connected9 | 3/3 | 0/3 | 0/3 | 60.0 | 276.7 | 1.000 |
| slanted_plane | translation_full9 | 3/3 | 0/3 | 0/3 | 60.0 | 69.3 | 1.000 |
| slanted_plane | translation_center3 | 3/3 | 1/3 | 0/3 | 40.0 | 208.7 | 1.000 |
| slanted_plane | translation_connected9 | 3/3 | 0/3 | 0/3 | 60.0 | 67.3 | 1.000 |
| slanted_plane | plane_full9 | 3/3 | 0/3 | 0/3 | 60.0 | 62.7 | 1.000 |
| slanted_plane | plane_center3 | 3/3 | 1/3 | 0/3 | 40.0 | 200.7 | 1.000 |
| slanted_plane | plane_connected9 | 3/3 | 0/3 | 0/3 | 60.0 | 64.7 | 1.000 |
| common_camera_bias | translation_full9 | 0/3 | 3/3 | 0/3 | 0.0 | 170.7 | 1.000 |
| common_camera_bias | translation_center3 | 0/3 | 2/3 | 0/3 | 20.0 | 176.7 | 1.000 |
| common_camera_bias | translation_connected9 | 0/3 | 2/3 | 0/3 | 20.0 | 175.3 | 1.000 |
| common_camera_bias | plane_full9 | 0/3 | 3/3 | 0/3 | 0.0 | 182.0 | 1.000 |
| common_camera_bias | plane_center3 | 0/3 | 2/3 | 0/3 | 20.0 | 182.7 | 1.000 |
| common_camera_bias | plane_connected9 | 0/3 | 2/3 | 0/3 | 20.0 | 183.3 | 1.000 |

## 配对有效性与边界

全部6组配对的参考中心3×3像素最大差为0，真值深度与表面标签逐像素相同。源图因像素积分和边界遮挡仍可能混合不同表面；这些变化本身是观测机制的一部分。

EVALUATION.json逐例记录参考掩码真实目标表面比例、每个源视图可见的目标掩码比例、中心是否被遮挡、正平面假设的世界坐标误差、真值处双源NCC、全部支持区间和选择结果。这些诊断不进入掩码、评分或选择。

封存后的FOOTPRINT_DIAGNOSTICS.json进一步审计源图的完整采样足迹：双线性插值的四个源像素，再包括每个像素的3×3积分子射线。在3×3目标片的plane_connected9真深度采样中，两源的目标表面平均贡献都只有83.73%，9个采样中8个混入背景、仅1个纯目标；5×5目标片平均贡献91.20%，25个采样中14个混合、11个纯目标。上述比例来自实际几何和采样权重，与纹理种子无关。虽然每个被选参考射线属于目标、真值投影点在两个源均可见，完整源像素足迹仍跨越边界。这是“点的所有权”与“实际成像足迹的所有权”之间的实质差别。

## 解释与下一步

本次结果不支持plane_connected9在这些受控条件下普遍优于full9。十二个边界场景中，plane_full9含真值12/12、精确选中600mm为10/12；plane_connected9分别为3/12和6/12。窄目标片加入背景纹理后，plane_full9的平均选择误差从0升至40mm，plane_connected9从100升至250mm。该配对说明外部区域可以改变中心目标的决策，但没有显示当前连通掩码构造消除了这种影响。它也给出了为何需要完整成像足迹闭合条件的具体反例。

single_plane_good仅表示几何为正确单平面、纹理强度足够，不表示纹理唯一。该场景的plane_connected9真深度双源NCC约0.998，三个种子都含600mm，却全部选660mm：其它被接受区间的长度使支持均值偏移。因此“非空”“包含真值”“支持局部化”和“候选排序正确”是不同事件，不能相互替代。

负对照同样保留全部结果。弱纹理使全部18次方法运行空支持；周期纹理在全部18次运行中包含真值，但全部选择660mm。遮挡场景的右源中心实际不可见，plane_connected9仍在2/3种子接受600mm，显示双源高相关本身不能证明目标身份。斜平面的plane_connected9完整参考掩码最大世界坐标偏差约53.98mm，虽含中心真值仍全部选择660mm。共同相机偏差让全部18次运行排除600mm；一些方法却因多个错误区间的均值相互抵消而恰好选择600mm，不能据此将支持视作正确或将标定问题视作已修复。

这些是公开设计的机制场景，不是随机自然场景样本。连接灰度掩码只表示可观测的所有权假设；即使掩码全部属于目标，重复纹理与共同相机误差仍可产生错误而一致的支持。正平面投影只在参考支撑实际近似该平面时有物理依据。固定NCC阈值不保证不同掩码大小拥有同等误匹配率。

应将本机制证据与真实回放并列解释，保留失败场景。不能据此宣称目标身份得到证明、支持具有校准覆盖率，或真实最近激光误差必然改善。下一步若继续，应预注册独立的几何/可见性条件及新场景，不根据这些结果反复调整阈值。

## 文件与封存

- PROTOCOL.json：评分前协议。
- FIXTURES.json、fixtures/*.npz/png：全部世界定义、相机、渲染浮点图、深度/表面标签和预览。
- RENDER_LOCK.json：渲染源码、协议和场景清单哈希。
- OBSERVATIONS.json、curves/*.npz、OBSERVATIONS_LOCK.json：候选/真值无关的支持与原始曲线，及共享评分源码哈希。
- SELECTIONS.json、SELECTIONS_LOCK.json：评估前封存的候选选择。
- EVALUATION.json、SUMMARY.json、PAIRING_CHECKS.json：原始逐例评估、聚合和配对检查。
- footprint_diagnostics.py、FOOTPRINT_DIAGNOSTICS.json：封存后的成像足迹诊断，不改变任何图像、评分、阈值或选择。

上述具体解释在观察结果后加入本报告；锁定的run_mechanism.py、PROTOCOL.json、图像、曲线与选择均未据此调整。
