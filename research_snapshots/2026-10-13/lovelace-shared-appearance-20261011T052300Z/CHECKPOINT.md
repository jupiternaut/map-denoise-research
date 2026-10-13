# 共享外观实验检查点

## 身份

host `liekkas`，绝对目录 `/srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z`。
源来源GitHub `https://github.com/jupiternaut/map-denoise-research`，记录commit `bd8d05aad65e2c55cd9cd288beae4ef949606984`。本轮不是Git操作，不声称该目录已发布。

## 当前交付

一次锁定确认完成且独立核验PASS。旧hybrid动作不变；shared_union仅恢复跨像素/视图三个未知材料的一致性。432偏移实例115改善、317同分、0恶化，MAE下降22.35%；54正确初值KEEP。所有54真深度覆盖，负控72KEEP。

## 不可改写的输入与科学源

历史nuisance/hybrid/challenge目录只读；46文件摘要在`outputs/confirmation-v1/SOURCE_LOCK.json`。不改确认PROTOCOL/THEORY/solver/runner/verifier/tests；正式结果及核验已存在，入口拒绝覆盖。

初始研究合同的“未运行”是确认前快照。最新已测状态由REPORT、EXPERIMENT_RESULTS及TRACKER记载，不追改确认前协议。

## 下一步边界

消融计划是未来工作，不算已运行；先补组件/完整预算对照，再引入有公开误差界的跨视图外观变化。任何新实验使用新协议、新输出目录，不为本次确认换阈值。不改真实点云默认；没有授权新增GPU/API/软件依赖、上传远端或作外部消息。

## 已知限制

模型为已知分区、三个灰度材料、已知不透明投影成像与有界噪声/相机误差的合成装置；不是任意真实纹理。核验程序共享历史几何内核；函数信息分离不是OS盲测；计时共用缓存。未决盒的保留不证明它们有真实场景，条件非恶化只覆盖合同内的标量深度误差。

## 复现

见COMMANDS.md。确认目录已存在，禁止重复执行确认或核验来覆盖；如需再现，先声明独立新输出而不是把再现写作第二次确认。没有未完成的本轮后台作业。
