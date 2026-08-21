# Final14 Mono × CEC 深度因子实验冻结协议

## 目的

Final14 已经证明 CEC 在 natural-direction mixed Novel/Revisit 查询中有效。本实验不再
选择阈值或修改方法，而是回答两个尚未被严格拆开的归因问题：

1. 当 NavDP 查询控制器完全不读取 simulator metric depth、只使用 frozen LingBot-Map
   的 causal monocular sidecar 时，CEC 相对 native 和 raw-DINO 是否仍有价值；
2. mono 与 metric 的差异究竟来自 NavDP 局部控制，还是来自 CEC 的接管收益。

## 冻结总体与边界

- 总体：Attempt 7 的 natural-direction Final14，21 histories、10 scenes，每条一个 Novel
  和一个 Revisit 查询；manifest SHA-256 为
  `7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a`。
- 这是已经读过结果的 consumed attribution，不是新的论文确认集。
- Goal-A 历史最初由 metric-depth NavDP 生成。五臂都只重放完全相同的 causal RGB
  历史；mono/metric 因子只作用于 query controller。因此结果不得写成“全程 mono”。
- 21/21 histories 的 online-A 长度均不少于 40 帧，范围 115–364，中位数 166。
  query 时 mono scale 必须处于 causal first-40 active 状态，禁止 bootstrap 或未来帧。

## 五个严格配对臂

| arm | query depth | 历史接管 |
|---|---|---|
| `mono_native` | monocular sidecar | 无，exact native NavDP |
| `mono_raw_fixed` | monocular sidecar | raw-DINO top-1 fixed bearing |
| `mono_cec` | monocular sidecar | CEC 认证后的 fixed bearing |
| `metric_native` | metric request | 无，exact native NavDP |
| `metric_cec` | metric request | CEC 认证后的 fixed bearing |

五臂共享 scene、query、Goal-A RGB replay、seed、NavDP/MemNav 进程及 query budget；arm
顺序按 history index 循环平衡。运行时不读取 Novel/Revisit 标签。

## 主检验和次级归因

预注册主检验：

- `mono_cec - mono_native`；
- `mono_cec - mono_raw_fixed`。

都分别在 Novel、Revisit、以及两者平衡合并的 42 queries 上报告 successes、配对
gain/loss、exact McNemar p 和 scene-cluster bootstrap 95% CI。

次级传感器归因：

- `mono_native - metric_native`；
- `mono_cec - metric_cec`；
- `(mono_cec-mono_native) - (metric_cec-metric_native)`。

交互项只作归因性描述，不冒充新的独立确认。

## Fail-closed 审计

- 每个 mono plan 必须报告 `monocular_sidecar`，且
  `metric_depth_sensor_consumed=false`；
- 每个 mono plan 必须带 causal first-40 scale receipt，frame index ≥40，同一 query
  scale receipt hash 唯一，且 whole-episode cache 未被读取；
- 每个 metric plan 必须实际读取 metric depth，且不出现 mono receipt；
- 对 CEC 全拒绝的 query，必须与同 depth 的 native 臂逐 plan seed、selected trajectory
  hash 和物理轨迹完全相同；
- success 由独立 verifier 从 raw final distance `<1.0 m` 重算；
- 任一字段不满足即不汇总。

## 冻结运行参数

- max query steps：600；execution horizon：8；success distance：1.0 m；
- terminal U-turn / visual refinement：off；
- retrieval oracle、gate oracle、trajectory oracle：off；
- graph rescue、CDEC rescue：off；
- scene cluster bootstrap：100,000 次，seed 2026081902。

本协议冻结于任何新增 Final14 mono query 结果产生之前。
