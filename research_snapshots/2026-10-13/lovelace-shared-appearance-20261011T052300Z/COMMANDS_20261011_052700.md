# 复现入口

主机liekkas；新实验目录：

```bash
cd /srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z
```

作者测试/独立核验器测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
```

非确认样例（600模型mm、99011种子，不写确认结果）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 run_shared.py --smoke
```

锁定科学代码、完成审查后，一次正式确认；目录已存在则拒绝覆盖：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 run_shared.py --output outputs/confirmation-v1
```

确认结束后，独立进程核验：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 shared_verification.py --output outputs/confirmation-v1
```

四臂共用54配置、486初态，共1944决策。两个原照片臂不读材料样本；校准臂独自使用三个额外样本，仅信息对照。单CPU、GPU=0、确认1800秒上限。旧nuisance/hybrid/challenge目录只读。不改点云默认、不发布仓库。
