# HM3D full-mono 多段记忆累积实验（冻结协议）

冻结时间：2026-08-24 04:27:50（Asia/Shanghai）。机器可读双生协议：
`MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json`。冻结时尚未产生或读取本协议的
`C/B2/C2` 导航结果。

## 1. 只回答一个核心问题

本实验不是再证明一次“Revisit 有用”，而是检验记忆能否在实际运行中继续增长：

```text
A（Novel，既有 actual-mono 事实轨迹）
  → B（Novel，actual-mono native 只运行一次并封存）
  → C（回到 A 历史中的受控 Revisit）
  → B2（回到刚刚走过的 B）
  → C2（再次回到 A）
```

主比较发生在 B2：`all_prior` 可访问事实 A+B，`initial_leg_only` 的候选上限永久锁在
A 结束帧。两臂在 C 结束前使用相同历史边界、相同随机种子和相同动作；只有 B2 开始时
是否允许检索新增 B 记忆不同。因此 B2 的配对差异才是“在线记忆累积”的直接效应。

## 2. 为什么 B 真的是当前 agent 的 Novel

父 population 是 2026-08-20 已独立验证的 HM3D actual-online full-mono 运行：196 条
Goal-A source、131 条 actual-mono A 成功、130 条可物化 A 历史，分布在 46 个非空
scene clusters。父 manifest SHA-256 为
`a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5`。

对每条 recipient A 历史，B 目标只能取自同场景另一条 actual-online A 历史的最后一个
事实 RGB observation 及该帧的精确位姿（不是把控制器动作后的 endpoint 错配给前一帧
图像）。它满足：

- donor 与 recipient 不同；
- recipient A 末端到 B 的测地距离为 2--9 m；
- B 对 recipient A 全历史的最大共视严格小于 0.10；
- B 与后续 C 的测地距离为 2--9 m。

donor 只用于生成一个真实相机分布的目标图，运行时不提供 donor 身份、轨迹、位姿或
role。若有多个 donor，按预冻结的距离、共视、episode rank 排序；允许同场景 donor 被
重复使用，但统计始终按 scene 聚类。

## 3. C、B2 与 factual-B population

C 是 recipient A 历史附近重新渲染的受控视图，不是历史 JPEG 原样重复；其 A-history
最大共视在 `[0.55,0.90]`，平移 0.2--0.8 m、朝向差 12--45°。

B 只由冻结 mono native NavDP 跑一次：`monocular_sidecar`、600 steps、8-step execution
horizon、1 m success radius，禁止 simulator metric depth。失败 B 完整进入 attrition
ledger；成功 B 写成 canonical trace 并哈希封存。B2 重用同一 B 目标图，但只有当它被
实际 B 轨迹支持（max covis ≥0.20）时才进入查询 population，≥0.50 另报 strong support。
同时从实际 B endpoint 到 C 的测地距离仍须在 2--9 m，避免把“成功后停在目标半径边缘”
偶然变成过短的下一段。
这些都是 `C/B2/C2` 之前的事实前缀/可构造性条件，不读取任何后续 query outcome。

所有满足条件的 history 全部保留，不替换、不放宽阈值。预期目标是至少 24 histories、
15 scenes；若达不到仍报告完整结果，但明确标为 underpowered。

最终 query population 不只保存 B 的轨迹哈希，还复制并封存该次 factual-B 的原始
completion/depth audit。独立 verifier 会重新读取 A、B 的原始 NavDP plans，逐条核对
`monocular_sidecar`、单目 depth receipt、固定 causal scale receipt 和零 simulator
metric-depth；因此“full-mono”不是仅由聚合 summary 自报。

## 4. 三个同机配对臂

三个臂都先逐帧重放完全相同且哈希一致的 A、B factual RGB，并重建相同的 mono NavDP
FIFO 与 CEC long-term memory：

1. `all_prior`：C 只看 A，B2 可看 A+B，C2 可看全部因果历史；
2. `initial_leg_only`：C/B2/C2 的检索上限都锁在 A；
3. `forced_reject_native`：记录同样的 CEC shadow proof，但从不授予接管，逐动作执行
   frozen mono native NavDP。

role 仅在离线分析 sidecar 中存在，从不传给 controller。每个 goal 必须开启且只开启
一个新的 CEC goal session，long-term memory 保留，NavDP short FIFO 在 C 之前按冻结合约
清空。`all_prior` 与 `initial_leg_only` 必须在同一张 GPU、同一组常驻 MemNav/NavDP/CEC
进程中先后执行，每臂前完整 reset；偶数 population index 先 all-prior，奇数先
initial-only。这样主比较不会混入项目已经实测过的跨机器 CUDA 路径漂移。
`forced_reject_native` 随后在同一 allocation/GPU 上用新的 fail-closed hub 运行。

## 5. 预注册指标

Primary 是共同成功且逐动作相同的 C prefix 后，B2 上 `all_prior - initial_leg_only` 的：

- paired gain/loss 与风险差；
- two-sided exact McNemar；
- scene-cluster bootstrap 95% CI；
- both-success 子集的 paired steps/path（效率 co-primary）。

Secondary 才是 C、C2、三查询 prefix-survival、joint SR、all-prior 对 forced-native，以及
B2 是否实际采用 factual-B anchor。B2 分叉以后 C2 起点可能不同，因此 C2 不冒充严格
动作配对的主证据。

## 6. 允许的结论边界

若独立 verifier 通过，本实验可以证明 full-mono CEC 在 previously Novel 的 B 被亲历后，
能否把它转化为后续可调用的 episodic memory，并给出多目标连续运行的逐段 SR 与生存
曲线。

这 54 个 HM3D scenes 已在 2026-08-20 实验中使用，因此本实验是 consumed-scene 的
lifelong accumulation mechanism confirmation，不是新的 fresh-scene generalization；也
不是训练结果、Novel 导航改进或形式化安全证明。
