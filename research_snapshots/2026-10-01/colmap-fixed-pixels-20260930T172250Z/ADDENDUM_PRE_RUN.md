# 运行前接口定稿（优先于初稿中的通用 COLMAP 角点描述）

未执行MVS、未读取本轮GT。两名并行核查者及主执行者核对官方4.2.1源码后定稿：

1. **MVS专用相机使用数组整数格点 K**。`model.cc`→`image.cc` 原样传K；`patch_match_cuda.cu::ComputePointAtDepth` 计算 X=depth*(col-cx)/fx，Y=depth*(row-cy)/fy，CUDA取纹理才加0.5。`fusion.cc`同样以[col*d,row*d,d,1]回投。因此本轮已预缩小的灰图对应cx=388,cy=290；若通用SfM格式/操作应使用cx=388.5,cy=290.5，不混用。这里是适配官方稠密内核的输入基底，不修改库。深度数组D[v,u]与旧数组P反投影直接对应。
2. 主参考图深度端点严格匹配旧实际扫描格点：scan24 [559,786]mm；scan37 [581,865]mm，而非REBUILD中未取整的分位数。COLMAP连续采样，旧法1mm格点加抛物插值，搜索离散化本身属算法差异。
3. photometric主臂：只算ref，4Q源，filter=True。geometric主臂：独立workspace先给5图分别算unfiltered photometric，再只给ref算filtered geometric。ref使用上述相同range；Q的range从ref全图四角×深度两端的8个视锥顶点转到Q相机Z取min/max。没有GT、稀疏3D或额外照片。避免缓存使过滤和未过滤photo混用。
4. 官方GPU PatchMatch没有用户seed选项；gpu_mat_prng.cu固定使用线程id作为curand_init种子。取消“3个外部seed”的无效调用；2场景各跑一次主臂，scan24 photometric额外独立工作目录复跑一次，作重复一致性而非随机泛化实验。
5. 核心算法参数均为4.2.1默认：window_radius=5,num_iterations=5,num_samples=15；只固定gpu_index=0,cache_size=2GiB,num_threads=2以控制资源。max_image_size=-1禁止库再次缩图。
6. 几何臂使用空间传播及源深度一致性，CPU旧法是5×5前向平面扫描；不是只换求解器的消融，而是同照片同相机同查询的成熟方法基线。

来源：
- https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/patch_match_cuda.cu
- https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/image.cc
- https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/model.cc
- https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/fusion.cc
- https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/gpu_mat_prng.cu
