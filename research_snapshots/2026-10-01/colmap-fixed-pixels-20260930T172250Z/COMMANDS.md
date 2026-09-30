# 执行与复现

目标主机 `liekkas`；运行目录 `/srv/slam-research/grf/map-denoise/runs/colmap-fixed-pixels-20260930T172250Z`。

```bash
uv venv --python /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421
uv pip install --python /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -r requirements.txt
```

已执行（不要在已封存工作目录重新覆盖运行）：

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_colmap.py prepare
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B freeze_setup.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -u -B run_colmap.py run
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B evaluate_mvs.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B normalize_scopes.py
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B plot_results.py
```

环境：Python3.12.13，pycolmap-cuda12 4.2.1，cuda-toolkit12.9.2.0，nvidia-cuda-runtime-cu12 12.9.79，nvidia-curand-cu12 10.3.10.19，numpy2.5.3，scipy1.18.1，pillow12.3.0；绘图使用原open3d-019的Matplotlib。GPU为RTX3060Ti8GB，驱动595.91.07。

完整重新运行请在本项目runs下新建命名目录，复制本轮源代码/契约（不复制workspaces输出），保留OLD/BASE指向原只读运行；在新目录运行prepare、freeze_setup、run、evaluate。首次prepare后freeze_setup只更新GPU进程检查的源哈希并封存评价脚本，无需再次安装已有环境。程序通过`__file__`定位新目录。输出文件排它创建，防止重跑静默覆盖。

准备脚本不读GT。评价只在118项预测封存以及旧候选seal都通过后读取官方参照。空间图片是所有固定像素，未按效果选ROI。
