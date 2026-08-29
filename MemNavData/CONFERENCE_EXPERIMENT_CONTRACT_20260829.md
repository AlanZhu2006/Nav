# CEC 会议实验统一契约（2026-08-29）

本文是会议 Table 1、Table 2 与两组消融的统一口径索引。逐实验的 immutable
population、hash、seed 和作业信息仍以对应 protocol JSON 与 independent verifier
为准；本文不替代它们，也不手工汇总未封存结果。

## 1. 方法名只允许以下含义

| 论文名称 | 冻结定义 | 唯一允许变化 |
|---|---|---|
| NavDP native | 冻结 NavDP ImageGoal policy；Table 1/2 使用 causal LingBot 单目深度 sidecar | 无 episodic-memory authority |
| NavDP + CEC | 与同表 native 完全相同的 NavDP、输入、seed 与预算 | certificate accept 时额外输入单位 bearing，经固定 `2.5 m` residual；reject 时执行逐请求相同的 native ImageGoal |
| ViNT native | 冻结 ViNT ImageGoal controller 及其原生请求 | 无 episodic-memory authority |
| ViNT + CEC | 与同表 ViNT native 完全相同 | 第一个已认证 bearing 只通过若干个不超过 `30 deg`、零平移、每次读取 fresh observation 的物理转向兑现；reject 时 exact native ViNT |

`Raw fixed`、`old geometry`、`finite-PnP`、role oracle、Pi3X、CDEC 和 GCT 只属于消融或
负结果，不得混入 Table 1 主方法行。

## 2. CEC 的冻结干预边界

CEC 读取当前 Goal image 与此前已经真实观察到的 causal RGB history：

1. frozen DINO 做 temporally diverse top-8 address proposal；
2. SuperPoint + LightGlue 建立局部对应，Fundamental-MAGSAC 检查两视图支持；
3. historical LingBot monocular depth 将历史关键点 lift 到 3D，PnP-RANSAC 产生相对位姿 witness；
4. operational certificate 要求 PnP inliers `>=16`、query/reference hull coverage 均
   `>=0.05`、reprojection RMSE `<=2 px`；
5. accept 时丢弃单目平移尺度，只释放单位 bearing；reject 返回 `⊥`，不释放替代动作。

CEC 不是第二个 planner，不读取 Novel/Revisit role，不使用 simulator depth、未来帧或
ground-truth pose。正常 reject 与 geometry-stream/runtime failure 必须区分：前者 exact
fallback，后者 fail closed 且不能计作正常 reject。

## 3. 统一输入与因果历史

- Full-Mono NavDP 的 dense input 是 frozen LingBot streaming relative depth，尺度只由前
  40 个 causal observations 与已知相机安装高度估计一次，随后 immutable；控制与 CEC
  的 metric-depth sensor read budget 都为 0。
- Memory 是显式 causal RGB buffer 及其逐帧 receipt，不是 expert trajectory，也不是
  LingBot KV cache。历史写入顺序必须和机器人实际 observation 顺序一致。
- Leg 3 的 `actual_ab` history 必须是完整 actual-mono A 段与 B 段 observation concat；
  receipt 中的 A/B step counts、逐帧 trace 和 replay decision frames 必须一致。
- 同一 paired comparison 的 native/CEC 共享 query、历史、controller checkpoint、seed、
  最大步数和成功半径。不同 controller 之间的绝对 SR 不是 paired estimand。

## 4. Query 与 role 契约

- 每条 history 配一个 unsupported Novel query 和一个 supported Revisit query；runtime
  role visibility 始终为 `none`。
- Novel 要对完整可用 history 满足 protocol 指定的低共视门；Revisit 必须由实际历史
  observation 周围的受控位姿扰动构造，并满足冻结共视区间。
- 任何已消费 query 的 JPEG SHA 或 pose+yaw identity 都不得包装成 fresh query。
- Smoke outcome、partial formal outcome 和旧 query outcome 均不得用于选 population、
  改阈值或挑方向。

## 5. Table 1 统一统计契约

- 数据集内每个 controller 分别估计 `CEC - native` 的 paired effect；禁止把 NavDP 与
  ViNT 的绝对 SR 当成方法优劣比较。
- 每条 query 最大 600 steps，成功半径 1 m；报告 Novel、Revisit 与 balanced overall
  的 SR，并保留 SPL、path length、steps、final distance 作为次要量。
- 主检验是 two-sided exact McNemar；不确定性为 100,000 次 scene-cluster bootstrap
  95% CI。必须同时报告 paired gain/loss、takeover 与 exact-fallback 审计。
- HM3D Table 1 的 frozen population 是 28 histories、21 scenes 的 fresh-query、
  scene-overlapping population。MP3D 必须先独立通过 construction power gate，才能运行
  同协议四行；不得用 Final14 metric-depth 数字填 Full-Mono 表。

控制协议源：

- `hm3d_table1_controller_portability_protocol_20260829.json`
- `mp3d_table1_fullmono_source_expansion_protocol_20260829.json`

## 6. Table 2 统一统计契约

- 固定 sequence 为 actual-mono Novel-A -> Novel-B -> mixed Novel/Revisit-C；A/B prefix
  在任何新 C outcome 之前封存。
- Leg 3 native/CEC 使用同一 A+B prefix；报告 `C | successful A,B` 的 paired Novel 与
  Revisit effect，同时单独报告 factual A/B source waterfall。
- 不得把只对 successful A/B prefix 构造出的 C 结果写成 unconditional three-leg joint
  SR。Leg 1/2 的 factual completion 与 Leg 3 的 treatment effect 必须分栏呈现。
- Prospective construction gate 是 `>=16 histories`、`>=10 scenes`、front/side/rear
  各 `>=3`；未过门不得降低阈值或运行 controller。

控制协议源：`hm3d_table2_leg3_mixed_role_protocol_20260829.json`。

## 7. 结果开放与失败处理

每条正式链必须依次具备：immutable source receipt、smoke、formal aggregate、独立从
episode receipts 复算的 verifier、最终 seal。只有 seal 后才允许读取 SR/SPL/final
distance；运行期间只可查看 scheduler state、进程退出码和不含 outcome 的结构完成数。

基础设施错误必须 fail closed、保留日志、精确修复并重跑完整 paired unit。不得把 runtime
failure 计作 certificate reject，也不得复用失败链的单臂数字与修复链拼表。

## 8. 当前允许的论文 claim 边界

- 可以：CEC 在冻结 controller 内的 Revisit utility、unsupported-query exact fallback、
  second-controller transfer、HM3D/MP3D 分别通过同协议后的 cross-dataset replication。
- 不可以：formal safety guarantee、CEC 显著超过 raw memory 的 Revisit ceiling、
  NavDP-vs-ViNT superiority、oracle bearing 是 deployable method、conditional Leg 3 被包装
  成 end-to-end joint SR。
- 真机闭环和 20--50 m length buckets 在各自冻结协议完成前保持 pending，不能用
  no-motion smoke 或 executed path length 替代。
