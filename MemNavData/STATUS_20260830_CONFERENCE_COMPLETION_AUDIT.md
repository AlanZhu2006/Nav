# 2026-08-30 会议实验完成度审计

本文件只回答两个问题：会议清单现在完成到哪里，以及下一份 GPU/真机时间应该花在哪。

## 1. 当前完成度

| 会议项目 | 状态 | 最准确结论 |
|---|---|---|
| Table I：跨 controller / dataset | **完成** | HM3D 与 MP3D 上 NavDP、ViNT 的 native/CEC 四行均已 sealed、独立复算；只作 controller 内 paired claim |
| Table II：HM3D continual by leg | **部分完成** | factual A/B 与 multi-goal evidence 已有；强制 Leg-3 Novel/Revisit pair 在 8 histories / 6 scenes 停于 constructibility gate，无 policy SR |
| Real robot | **未完成** | transport、hash、fail-stop 与启动框架可用；尚无冻结的 paired autonomous outcome |
| Depth ablation | **完成** | 同 Final14 query population 的 metric/mono/zero 与 CEC 对照完成；Goal-A history 仍是 metric replay，须保留边界 |
| CEC mechanism | **完成** | proposal、finite-PnP witness、strict authority 与闭环已统一复算；authority 行为成立，阈值特定 SR superiority 未显著 |
| Length buckets | **未完成，低优先级** | 当前 query 全在 0--20 m；要填表必须另建 20--50 m benchmark |

## 2. 8 月 30 日新增的两项硬结论

### 2.1 Table II 的缺口不是多跑 5,000 次 sampler 能解决

13 条 Novel attrition 已执行 195,000 次尝试。6,660 个候选通过全部前置几何门后，全部
因与完整 A+B history 的 covis `>=0.10` 被拒；side 单独有 2,847 个。因此这是二元
role-pair 在长历史后的构造性稀缺，不是方向 sampler、CUDA 或预算失败。

### 2.2 CEC 的核心贡献是 authority，不是另一个 matcher

Final14 supported Revisit 的 DINO top-8 address coverage 为 21/21；但 finite PnP pose
会授权 18/21 unsupported Novel。严格 certificate 把 Novel authorization 降至 2/21，
同时 Revisit 保持 21/21 authorization、20/21 SR。闭环 CEC/raw 是 28/42 vs 23/42，
`+5/-0, p=.0625`；CEC/finite-PnP 是 28/42 vs 25/42，`+4/-1, p=.375`。论文应强调
proof-before-control，而不是夸大阈值 superiority。

## 3. Table II 怎样进一步完成

优先选择是保留 HM3D constructibility null，并把 continual 表改成真正可辨识的两层：

1. factual prefix survival / leg waterfall；
2. 在相同 factual C prefix 后，只改变 `all_prior` 与 `initial_leg_only`，检验新完成的 B
   是否成为以后可调用的记忆。

当前 HM3D B2 job 正是第二层，sealed population 为 17 histories；它完成后仍必须标为
underpowered external mechanism evidence。现有 MP3D 18-episode retained-history
dose response 可保留为内部证据，但不冒充会议要求的 HM3D powered table。

若作者团队坚持原始 Table-II 二元矩阵，不能修补现有 22 条。必须另冻至少约 44 条新的
successful A+B prefixes，或改用更大场景数据集；然后原封不动地重用 `<0.10`、
`[0.55,0.90]` 与方向 gate。这个扩样会消耗大量 factual A/B rollout，且论文增益低于
完成真机，因此当前不把它列为 P0。

## 4. 现在的优先级

1. **P0：真机 paired campaign。** 这是会议主交付物中唯一完全没有 outcome 的高优先级
   项；先校准独立 arrival/path evaluator，再跑冻结的 20 pairs。
2. **P1：等待 HM3D B2 node-affine smoke/formal。** 不改 source node、partition 或 replay
   contract；它受跨节点 RGB hash 约束，不能随意迁移到另一张卡。
3. **P1：论文同步。** 激活统一 mechanism table，写入 Table-II constructibility boundary，
   不修改已经冻结的 abstract/introduction。
4. **P2：20--50 m length benchmark。** 只在真机与核心表完成后启动。

## 5. 当前 HPC

截至 2026-08-30 北京时间早晨：

- `16540468_[0]`：HM3D B2 true-stack smoke，`h100_tandon`，1 小时，因
  `QOSMaxGRESPerUser` 等待；scheduler estimate 为 2026-08-31 19:55（北京）；
- `16540469`：CPU deferred launcher，正确等待 smoke dependency；
- 没有新的 B2 SR，不能读取 partial result。

该 smoke 必须在 factual-B/C 的原 node 上 exact replay；因此不能为了更快排队随意转到
L40S 或 5090。若取消/重提，必须保持 node affinity、immutable bundle 和未读 outcome
边界。
