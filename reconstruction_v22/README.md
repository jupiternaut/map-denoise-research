# V22：逐点局部曲面重建与多尺度修正

这不是宣称具有全局最优性或任意场景泛化的新滤波器。它是对 V18 两项具体假设的可运行替换：整片平行面 → 随位置变化的局部曲面；全残差硬投影 → 带启发式可靠性阻尼的修正。

## 算法本体

对每点 i，以自身以外的近邻建立 PCA 坐标，在 k=32/64/128 三个支持范围分别拟合：

\[
z_j=\theta_0+\theta_1u_j+\theta_2v_j+\theta_3u_j^2+\sqrt2\theta_4u_jv_j+\theta_5v_j^2+e_j.
\]

二次项加固定微小正则。平方项使用旋转相容的基，保留刚性坐标变化的一致性。每个查询点有自己的坐标、法向和二次系数，不把1024点统一压成两张平行面。

各尺度预测给出一个法向修正向量。用估计预测方差进行加权，将其投影到 k64 法向得 d_i；跨尺度分歧为 s_i²，残差尺度为 v_i，预测方差代理为 t_i，则：

\[
\alpha_i=\frac{v_i}{v_i+t_i+s_i^2+\varepsilon_i},\qquad
p_i'=p_i+\alpha_i d_i n_{i,64}.
\]

这里的 v、t、s 都来自输入拟合，并非独立测量标定。它们是启发式系数，不是经过校准的后验概率。所有尺度可能共享错误偏差；剔除自身拟合行不等于彻底消除输入相关性。

## 具体接口

`operator.py` 中 `construct(points, ks=(32,64,128))` 返回：

- local_plane64：相同逐点支持的平面投影控制。
- quadratic64：二次图面直接投影。
- multiscale_full：多尺度预测，不阻尼。
- multiscale_consensus：多尺度预测＋逐点阻尼，预定主候选。

输入为 N×3 实数坐标（至少9点），输出保留相同点数与行序。算法不接收GT、真实sigma、扫描标签或参考文件。法向随查询点而变，仅沿各自法向移动。不是网格编辑API。

`run.py` 添加整体位移RMS匹配的常数阻尼控制、identity、冻结V18、官方APSS/RIMLS。旧scan24作暴露开发诊断，新scan37为一场景确认；详见 [PROTOCOL.md](PROTOCOL.md)。

## 复现

使用 liekkas 原项目目录和已有依赖，不修改旧环境：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s reconstruction_v22 -p 'test_*.py' -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B reconstruction_v22/run.py
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B reconstruction_v22/verify.py /绝对路径/新运行目录
```

扫描文件尚不存在时，`prepare_scan37.py` 下载单场景公开重建、独立参考与原作者相机；已完成时无需再运行。历史中断的部分文件保留并命名，不自动删除。

## 已知限制与相关工作

这是经典 MLS／局部多项式回归思路，不把“局部二次＋多尺度”本身宣称为原创原理。

- [APSS（2007）](https://cgl.ethz.ch/research/past_projects/apss/)
- [RIMLS（2009）](https://www.labri.fr/perso/guenneba/publi/RIMLS_eg09/index.php)
- [Manifold Approximation by Moving Least-Squares Projection](https://arxiv.org/abs/1606.07104)

断裂边、相邻薄面、关联错误、稀疏法向与相关系统误差仍可能失效。不通过更多平滑自动解决浮点伪几何，也不由单场景成绩宣称全局泛化。
