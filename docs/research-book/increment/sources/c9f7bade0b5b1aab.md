# 来源快照

原始路径：`/srv/slam-research/grf/research-skill-comparison-20260930T065500/refine-logs/EXPERIMENT_RESULTS.md`

# ARIS 构造结果

目标主机 liekkas；目录 `/srv/slam-research/grf/research-skill-comparison-20260930T065500`。仅使用 `experiment-plan` 的问题—证据—执行结构，构造并实际评估八个 standalone 算法。所有评价只用 scan24/37 开发数据，单 CPU 线程、每案180s上限，同一 prepared 数据/抽样/权重/留出折/metric/routing。没有改变候选几何，没有回放GT访问，没有API/GPU/新包安装。

## 开发结果

| 案 | 数学机制 | score | native | minus3 | plus3 | 时间s | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| P01 | shared joint ExtraTrees | .266070 | -.056874 | .351618 | .301747 | 29.419 | valid |
| P02 | own-only shared ExtraTrees | **.268883** | -.057917 | .368819 | .294853 | 12.442 | valid，最高新案 |
| P03 | shallow contrast HGB | .263585 | -.109147 | .354130 | .310971 | 51.006 | valid |
| P04 | invariant common + antisymmetric advantage | .258424 | -.077255 | .338799 | .301985 | 74.312 | valid |
| P05 | shared contrast benefit/harm | .261703 | -.127449 | .331440 | .329972 | 93.271 | valid |
| P06 | weighted quadratic ridge | .227579 | -.173199 | .300919 | .296735 | 6.935 | valid |
| P07 | exchangeable utility-weighted 3-way classifier | .161248 | -.321143 | .225651 | .262582 | 83.894 | valid |
| P08 | shared local-plane latent correction | .235613 | -.112919 | .245844 | .336344 | 30.070 | valid |
| common seed | independent candidate benefit/harm HGB | .269922 | -.101631 | .337879 | .334063 | 55.257 | 共同基线，不计八案 |
| KEEP | 不移动 | 0 | 0 | 0 | 0 | 共同检查 | 非恢复成就 |

最高新案 P02 的 KEEP macro MSE=2.292259372 mm²，output=1.675910670 mm²；移动34,383/78,598（macro movement .437520870），helpful/harmful=24,915/9,468。native MSE 从1.034428299增到1.094339487（5.79%退化），minus3从2.519036842降到1.589967056，plus3从3.323312975降到2.343425469。开发scalar winner不等于deployable winner。

## 锁定与共同seed

八案全部有效，失败/timeout=0，总 evaluator wall time=381.348173s。最高新案按冻结开发metric选择P02，保存 `aris/best.py` 与union of presealed train24/train37样本的最终模型、selection.json。最终refit不是新搜索提案，不提供额外OOF结果，也不读取回放参考。

共同seed=.269922051 比P02=.268882618高.001039433。因此此臂新构造没有超过共同seed；按允许fallback的workflow选择可保留common seed。这里将“最高新案产物”与“有效起点fallback”显式区分，不强迫采用略低分的新算法。seed的模型/推断设置由主代理统一封存。

## 主张解释

- C1：八个新案均相对KEEP有正开发总体score，但原生条件均退化；这证明构造了可运行且有部分恢复效用的选择器，不证明同时保留所有有用几何。
- C2：P01→P02删除上下文后分数略升、时间下降，随机树下上下文有益未获支持；其他joint结构也没有超越P02。按证据选择更简单结构。
- 平滑ridge很快但分数更低，分类概率不能代替绝对utility排序，物理平面近似在整体metric下失败。这些负反馈保留，不删失败机制历史。

## 范围与公平性

只做了单次functional pilot，同一八案上限不等于语言模型tokens、推理context或总算力相等。该代理的精确语言模型token计数没有可用逐臂测量，报告为 unavailable，不由源码行数或时间估算。模型/预算未切换；没有外部付费fallback。主代理在本臂后段提供ordinary的进度摘要，本臂未读取ordinary源码或复制其算法，P05–P08结构在此之前已写或由本臂机制失败与观测metadata驱动；这种共享上下文也限制严格受控workflow因果解释。

六场景此前暴露，因此即使之后回放也只是development replay。目前未进行回放、独立确认、deployment-default修改或Git push；不主张ARIS一般优越。

## 最终产物核验（2026-09-30 07:21:57）

`aris/best.py` 是P02原源码的字节相同副本，sha256=`ea02a6c010c8ae7d617b2598125e26d9abf7132d1ddf286c53350c4f3afbb4ba`。`aris/model.pkl` sha256=`b5fb317610390b25025a045876d3c6cda51098edd7176e6d646075ee588ef2ca`，72棵树、225输入维，首树root sample count=32000（16000行×两个共享候选）。`aris/selection.json` 将模型、源码、开发反馈、prepared data和训练indexes绑定；fit indices hash=`e2b65692af0fc8a64e8e389223c320d359b8afdb8e19c59d341f966c6315c690`。

所有八案源码hash仍与评价时相同，prepared文件hash验证通过。保存模型读回后32行x/geometry纯观测推断shape=(32,2)、全部finite；交换A/B输入/geometry后输出也严格交换。此核验不是新评价提案，没有读取gains/losses/回放GT，也没有新增score。

最终coordinator第一次fit与pickle成功，随后因相对program路径转换selection字段失败；将路径规范为absolute并只读回已有pickle，恢复selection，无重新fit。该实现错误不属于提案无效/timeout；八案仍8 valid、0 invalid。原fit elapsed没有成功写入，selection明确fit_seconds=null，不编造精确时长。文件/模型没有删除或替换。
