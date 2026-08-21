# MDTEC 单目短程读出 × CEC 长程读出组合协议（冻结草案）

状态：`conditional_on_gate_d_authorization`。本协议在 Gate D 全量结果与独立
verifier 出现前写定；只有 Gate D 工程授权通过才允许提交。若 Gate D 不通过，本协议
作废且不得运行。

## 1. 唯一问题

在同一个 `raw_lingbot_depth_first40_v1` 单目 NavDP controller 上，打开既有 CEC
是否仍能提高 supported-Revisit 成功率，同时保持其 reject 时的逐动作 native fallback？
这不是重新选择 CEC 阈值，也不是新的论文确认集。

## 2. 冻结人口

- 上游：Fresh160 的 immutable 20-scene/160-episode manifest；
- 选择规则：按上游 manifest 的每场景 episode 顺序取前 2 条；
- 总体：20 scenes / 40 episodes；不得按可构造性、支持度或结果再筛选；
- 数据角色：已消费 integration population，只能证明两个已冻结组件能否组合；
- blind/development：均不得读取。

## 3. 两臂与唯一差异

两臂共同使用：

- 同一 MemNav/LingBot server 与同一 NavDP server；
- `navdp_depth_source=monocular_sidecar`；
- first 40 causal RGB frames 输出 zero depth，随后使用一次冻结的 first-40 scale；
- 同一 NavDP checkpoint、goal/current RGB、diffusion seeds、FIFO、500-step budget、
  execution horizon 8、1.0 m success radius；
- 同一条 raw-monocular Goal-A rollout，经 SHA-256 绑定的 shared trace 重放；
- 不读取 Novel/Revisit role 标签。

两臂仅在 Goal-B 的长程读出不同：

1. `raw_native`：`hybrid_route=native_sidecar`。RGB 仍写入同一 LingBot stream，
   NavDP 仍消费同一 sidecar depth，但所有动作均为 native ImageGoal；
2. `raw_cec`：既有 `certified_relocalization`、既有 top-8/proof/certificate、既有
   `verified_bearing_v1_fixed_2.5m`；reject 时调用与 `raw_native` 相同的 native
   ImageGoal endpoint。

CEC retrieval、候选顺序、LightGlue/PnP、certificate 阈值、固定半径和生命周期全部
冻结，不允许依据 Gate D 或本实验结果修改。

## 4. 配对与审计

- scene 内 arm 顺序二阶平衡；
- Goal-A source 只运行一次，两个 B arm 的 A trace SHA 必须完全一致；
- 每个 raw plan 必须报告 `metric_depth_sensor_consumed=false`；
- frame index <40 的 depth 必须逐像素为零；有效 scale 的 frame 40+ 必须实际进入
  active raw depth，且整条 episode 只有一个 scale receipt SHA；
- CEC reject 的请求 seed、返回 seed、selected trajectory 与 `raw_native` 对应动作必须
  完全一致；
- certificate runtime/transport failure 必须为 0，否则 fail closed 且本次组合审计无效；
- 结果由独立 verifier 从 per-plan JSON 与 metric CSV 重算。

## 5. 报告，不设新的选择门

主要报告 Goal-A 成功后 conditional Goal-B：两臂 SR、SPL、paired gain/loss、exact
McNemar 与 scene-cluster bootstrap 95% CI；同时报告 joint A→B。Intent-to-treat 的 40 条
全部保留，A 失败不允许静默丢弃。

这是兼容性/组合性实验，不用它重新决定 CEC，也不以显著性决定是否保留已经由 Final14
和 HM3D 确认的 CEC。允许的结论仅为：短程 raw-depth readout 与长程 certified-bearing
readout 在同一因果单目 stream 上能否共同工作。

