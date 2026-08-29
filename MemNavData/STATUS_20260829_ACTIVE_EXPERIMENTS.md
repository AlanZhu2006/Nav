# 2026-08-29 会议实验活跃总账

更新时间：2026-08-29 13:29（Asia/Shanghai）。本文件是当前调度与证据边界的
唯一简表。任何 active rollout 都不读取 partial SR、SPL、final distance 或逐臂
outcome；只有 independent raw-file verifier 通过并写入最终 seal 后，才允许打开结果。

## 1. 当前结论先行

- **HM3D 的 ViNT 两行已经成为正式正结果。** 在冻结的 28-history、21-scene、
  56-query mixed Novel/Revisit population 上，CEC 对 Novel 做到 0 takeover 和 exact
  fallback；Revisit 从 `3/28` 提升到 `19/28`，paired `+16/-0`，风险差
  `+57.14 pp`，exact McNemar `p=3.05e-5`。Overall 从 `6/56` 到 `22/56`。
- **上一轮 NavDP 的 `14/56 vs 14/56` 不是方法零结果。** 它有 2,657 个
  runtime-failure plans，全部由 authority endpoint 契约不匹配触发，随后按设计
  exact fallback。它只能作为 fail-closed 基础设施审计，不能进入性能表。
- **正确的 NavDP 完整双臂修复已经提交。** 新 smoke 会把任何 endpoint/runtime
  failure 直接判为任务失败；不再允许基础设施错误被统计为正常 reject。
- **MP3D 第二数据集仍处在 outcome-blind query 构造阶段。** smoke 已完成，正式
  controller 四行尚未提交，也没有新 MP3D SR。
- **HM3D lifelong B2 仍在等待 GPU 优先级。** 它是会议 Table 2 的补充机制实验，
  不是 powered confirmation。

## 2. HM3D Table 1：跨 controller 正式结果与修复

### 2.1 冻结比较对象

- 28 条 actual-mono causal histories；
- 21 个 fresh HM3D scene clusters；
- 每条一条 Natural Novel 与一条 Revisit query，共 56 queries；
- runtime 不读取 Novel/Revisit role；
- 每个 controller 内 native 与 CEC 同 query、同 seed、同 checkpoint、同输入模态、
  同步数预算并严格 paired；
- NavDP 与 ViNT 的绝对 SR 不作为 controller superiority 的 paired estimand。

### 2.2 ViNT：已封存、可报告

运行链：原数组 `16526731`，唯一非科学崩溃 cell 18 精确重跑 `16540176`，
replacement aggregate/verifier `16540207/16540208`。独立 verifier SHA-256：
`89a5e85f7899e283ce8924fd1777e98b061ceb6577641cb6811b4c5f6b02bf32`。

| Role | ViNT native | ViNT + CEC | paired gain/loss |
|---|---:|---:|---:|
| Novel | 3/28 | 3/28 | 0/0；0 takeover，exact fallback |
| Revisit | 3/28 | 19/28 | +16/-0 |
| Overall | 6/56 | 22/56 | +16/-0 |

Revisit 风险差为 `+57.14 pp`，exact McNemar `p=3.0517578125e-5`；overall 风险差
为 `+28.57 pp`，scene-cluster CI 为 `[+20.0,+36.54] pp`。CEC 在 27/28 Revisit
queries 获得接管权，在 0/28 Novel queries 接管。它首次给出强闭环证据：CEC 的
proposal--witness--authority 接口不只适用于 NavDP。

### 2.3 NavDP 旧链：为什么无效

旧链先后修复了 byte-identical JPEG 的 causal transaction cache 和 aggregator
读取错误字段两个基础设施缺口。analysis-only replacement `16542535/16542536` 与
joint seal `16542548` 均已完成；只有在 seal 后才打开结果。

打开后发现 `mono_native = mono_cec = 14/56`、CEC takeover 为 0，并且所有 CEC
请求累计 2,657 个 runtime-failure plans。原始 receipt 的确定性错误是：

```text
RuntimeError: certificate endpoint used wrong authority policy
```

原因是 transaction-aware NavDP server 与旧 MemNav authority endpoint 被拼在一起；
后者不接受/回显 evaluator 冻结的 `strict_certificate` policy。因此 evaluator 对每次
异常都安全地退回 native。ViNT 在同一 population 上能接受 27/28 Revisit，进一步
排除了“历史没有几何 witness”这一解释。

### 2.4 NavDP 正确修复：已正式排队

新 server overlay 只组合两个已经 receipt-bound 的父组件：

- Final14 strict-authority MemNav server/policy agent；
- full-mono transaction-aware NavDP server。

checkpoint、DINO/LightGlue/PnP、certificate 阈值、2.5 m residual、query、history、
seed、arm order、600-step budget 与 1 m success radius 均未改变。完整 native/CEC
双臂必须同进程重跑，不能复用旧 native。

新增三层硬门：history runner、aggregator、independent verifier 都要求
`runtime_failure_plans == 0`。正常 certificate reject 仍合法并 exact fallback；
endpoint/proof runtime failure 会终止 cell 并阻断全部统计。

- task bundle：
  `hm3d_table1_navdp_authority_transaction_repair_3d11c13df616cfa0`；
- task receipt：
  `3d11c13df616cfa0a61f56ccd238b1cc5e8c983b14cc31db12778bc30593989a`；
- server overlay receipt：
  `ef4f30de3103d7af742137d8c63790e0f107afb880ad0650f9f98c649c05472d`；
- 远端隔离预检：8/8 authority/transaction contract tests 通过；所有 bundle SHA、
  endpoint symbol、bash syntax 与 Slurm test-only 通过；
- smoke：`16543736`；
- 54-rank formal array（并发 2）：`16543737`；
- aggregate / independent verifier / replacement joint seal：
  `16543738 / 16543739 / 16543740`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/navdp_authority_transaction_repair_20260829T052519Z_3d11c13d`。

最后一次只读调度检查：smoke 因 `Priority` 等待；下游均按 dependency 等待。尚未
读取或产生这轮 NavDP outcome。

## 3. MP3D Table 1：第二数据集构造

MP3D 官方 90 scenes 已被 train/development/blind/Final14 使用，因此不能声称
fresh-scene。冻结的合法 claim 是：

> reused scene/history，new outcome-blind query 的跨数据集、跨 controller replication。

协议从 Fresh160 的 20 scenes / 40 actual-mono histories 出发，不读取旧 Goal-B
outcome；每条 history 尝试构造一个 `covis<0.10` 的 unsupported Novel 和一个
`0.55<=covis<=0.90` 的 Revisit，并用 JPEG SHA 与 pose+yaw 同时排除旧 consumed
queries。prospective power gate 要求至少 20 histories、12 scenes，且
front/side/rear 各至少 4。只有 independent construction verifier 授权后才允许提交
NavDP/ViNT 四行 controller rollout。

- construction task bundle：
  `mp3d_table1_new_query_6813d50153b355e5`；
- smoke：`16542589`，`COMPLETED 0:0`；
- 20-cell construction array：`16542590`，并发 2；
- finalize / independent verifier：`16542591 / 16542592`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_new_query_20260829/construction_20260829T050401Z_6813d501`。

最后一次调度检查时，cells 0--8 与 10 已完成，9 与 11 正在运行，12--19 等待 array
并发槽。这里的“完成”仅表示构造进程正常退出，不代表 query 合格；正式分母必须等
finalize + verifier。未提交 controller rollout，未读取任何新 SR。

## 4. HM3D lifelong / Table 2

上游 factual-C integrity barrier 覆盖冻结 22 histories / 15 scenes；最终 17 条
accepted histories 可进入 B2。两者属于不同阶段的分母，不能混写。

- node-affinity repair bundle：
  `hm3d_lifelong_node_affinity_repair_ddd01842308dfa37`；
- resume launcher：`16540396`，已完成；
- true-stack smoke：`16540468_[0]`，等待 `Priority`；
- node-affine formal launcher：`16540469`，等待 smoke dependency。

这条线要补的是会议 Table 2 中 retained history 随 leg 积累的机制证据。它仍明确
标为 underpowered；即使结果方向漂亮，也不能用 17 条样本冒充 powered confirmation。
完整 Table 2 仍缺同-prefix 的 Leg-3 Novel 对照和统一的 Leg-1/2/3 分母/joint 定义。

## 5. 与会议清单的精确对账

| 会议交付物 | 当前状态 | 下一道门 |
|---|---|---|
| Table 1 HM3D 四行 | ViNT 两行成立；NavDP 旧链判为 infrastructure incident | `16543736--16543740` 全链 seal |
| Table 1 MP3D 四行 | outcome-blind construction 正在运行 | `16542592` verifier/power gate 后才提交四行 |
| Table 2 HM3D by leg | factual population 已封，B2 等待 | B2 verifier；随后只补 Leg-3 Novel 缺口 |
| Depth ablation | 已有 Gate C/D、Final14 factorial 与 full-mono 证据 | 先审计能否同 population 重组，禁止拼不同分母 |
| CEC mechanism ablation | Raw/CEC/known-role 底层证据大部分已有 | 统一导出 retrieval accuracy、FA/FR；不急着重跑 |
| Real robot | 软件合约与 no-motion transport 已有 | 仍缺冻结 protocol 下的 paired autonomous trials |
| Trajectory-length bins | 当前 2--9 m 为主，长距离桶不可构造 | 低优先级，主表完成前不启动 |

## 6. 当前最优执行顺序

1. 等 HM3D NavDP smoke 先验证 authority + transaction 真正在线贯通；只有它通过，
   54-rank formal 才会启动。
2. 完成 MP3D construction verifier。若 power gate 失败，诚实报告 constructibility；
   不降低阈值凑分母。若通过，立即提交同协议四行。
3. 利用空闲 GPU 让 lifelong B2 自然启动，但不为抢队列取消 Table 1。
4. HM3D 与 MP3D Table 1 seal 后，再生成会议主表并决定是否需要 Table 2 Leg-3
   Novel 新 rollout。
5. 不再启动新 learned module、GOAT 适配、length-bin 或额外 controller 支线；它们
   现在都不在论文关键路径上。

## 7. 工程与审计状态

- 共享 `alantorch` ControlMaster 正常，远端身份严格为 `yz11502`；本轮所有 HPC
  操作均通过共享 SSH 的交互 PTY 完成。不能再把 no-PTY/SCP channel 的挂起误报为
  Slurm controller down。
- 主提交器已永久改为固定 authority+transaction 组合 overlay，避免未来重新使用
  transaction-only endpoint。
- NavDP canonical direction 字段已从 Novel query 的
  `assigned_direction_stratum` 读取；不再访问虚构的 history-level 字段。
- MP3D/HM3D aggregators、verifiers 与 sbatches 已参数化，但 dataset claim scope
  必须显式给出。
- 当前仓库修改尚待完整回归、commit 与 push；在此之前不能把工作区状态称为 release。

权威提交收据：

- `HM3D_TABLE1_NAVDP_AUTHORITY_TRANSACTION_REPAIR_SUBMISSION_20260829.json`；
- `MP3D_TABLE1_NEW_QUERY_SUBMISSION_20260829.json`；
- `HM3D_TABLE1_NAVDP_ANALYSIS_PATH_REPAIR_SUBMISSION_20260829.json`。
