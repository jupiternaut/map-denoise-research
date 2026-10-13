# A9只读前沿审计

独立代理读取36个both checkpoint与已封存评价表，无求解、无旧文件修改。

118个informative未改善全部落在retained_tolerance+FEASIBLE_WITNESS叶，每个恰由一叶覆盖，宽15/32。none40、partial78；39个位于hull端叶、79个在内部。90个informative分支全部tree_complete，173个保留叶全部为容差可行。

48个negative未改善均落在deadline叶，涉及26个不同叶。全部both中唯一UNKNOWN在p016b80e0/full节点50，随后拆分保留，不包含偏移现任且不是hull端。

解释：不能把118个未改善归因于超时。新实验应缩小区间外包络、细分相机盒；GT分组仅用于此事后审计，不选择正式观察。
