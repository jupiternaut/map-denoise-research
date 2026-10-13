# A4 复现入口
精确目标：liekkas，本目录 appearance-1。标准库 Python；不需要GPU、API key或新依赖。

## 完整前置夹具
```bash
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1/test_suite.py
```

## 正式有限批次（只对全新目录）
```bash
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1/execute_appearance.py
```
默认输出 outputs/appearance-v1。已有输出会在任何写入前被拒绝。
主程序36冷求解/324保存动作和独立核验共用正式开始时刻起3600秒，不能重新调用核验来重置期限。
旧外包/动作复用而非新求解。主RUN_STATUS COMPLETE不单独表示阶段完成，
STAGE_STATUS COMPLETE与verification/VERIFY PASS同时存在才可交付。

## 通过后配对分析
```bash
python -B /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1/analyze_appearance.py --save
```
该命令只读取已核验的结果/参照与外包，生成analysis/PAIRED.json；不读取私有truth。
目标文件只创建一次，重复分析可去掉--save打印，不覆盖已有证据。

## 原始工件
- outers.json：36新外包及108分支追踪
- decisions.json：324新动作（先保存读回完整再评分）
- reference/results.json：324冻结共享动作+162 KEEP参照
- coverage.json：36条深度/生成分支/重复九材料覆盖
- SOURCE_LOCK.json、INPUTS.json：字节封存
- BUDGET.json、STAGE_STATUS.json、verification/VERIFY.json：预算及独立核验
所有报告是当前固定B80合成族的机制对照，不是自然场景迁移证据。
