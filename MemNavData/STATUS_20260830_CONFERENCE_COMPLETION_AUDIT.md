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

截至 2026-08-30 11:08（北京时间）的再次核验：

- `16540468_[0]`：HM3D B2 true-stack smoke，`h100_tandon`，1 小时，因
  `Priority` 等待；调度器当前给出 `StartTime=2026-08-31 07:55 EDT`，即北京时间
  2026-08-31 19:55。这个估计依赖 `gh005` 上现有长任务释放，不能视为启动保证；
- `16540469`：CPU deferred launcher，正确等待 smoke dependency；
- 没有新的 B2 SR，不能读取 partial result。

该 smoke 被显式绑定到 `ReqNodeList=gh005`。复核时该节点四张 GPU 已全部分配给两个
长任务，Slurm 因此暂时无法给出 backfill 起点；这不是代码、依赖、QOS 或 controller
故障。B2 复用的 factual-B/C RGB 在该节点生成，跨节点 replay 已知会改变渲染哈希，
所以不能为了缩短排队把任务迁移到另一台 H100/L40S/5090，也不应取消后无修改重提。
它保持为非阻塞、underpowered mechanism evidence，论文和真机关键路径继续独立推进。

## 6. 投稿关键路径补充审计

当前不再缺新的 controller、matcher 或 learned head。下一份时间应按以下顺序使用：

1. **正式真机 outcome：**先完成独立的 scale-free arrival calibration 与 held-out
   confirmation，再执行已经冻结的 20 个 native/CEC pairs；现有 transport、no-motion、
   手动干预或 powered debugging trace 均不能填入结果表。
2. **实现与叙事对齐：**长程 memory 是 causal RGB/descriptor address 与 LingBot
   geometry witness 的组合，不是 LingBot KV cache 单独完成内容寻址。可以保留
   “one causal RGB stream, two time scales, one frozen policy”，但不能把 state-only
   retrieval 写成已实现能力；candidate-free GCT 的负结果恰好说明显式 address 仍必要。
3. **论文系统证据：**补一张基于冻结 episode 的 qualitative trajectory/evidence 图，
   并在最终 RTX+Jetson 共驻配置上报告 end-to-end first-use/cached latency。已有 eager
   microbenchmark 只证明实现等价与速度/显存交换，不替代目标设备测量。
4. **唯一值得考虑的新仿真：**若真机完成后仍有算力，再冻结一个新的 powered
   mixed-role population，比较 always-on raw、finite-PnP 与 strict CEC；这是目前比
   length bins、更多 controller 或重复 Leg-3 sampling 更直接的统计缺口。

以下支线不再进入投稿关键路径：GOAT/Replica 适配、X-NavDP、Pi3X/CDEC 继续训练、更多
controller smoke、20--50 m length buckets，以及在现有 22 条 A/B prefix 上重复采样。

## 7. 真机 formal 软件边界更新

截至本次审计，真机仓库 `AlanZhu2006/MemNav-RealWorld` 已连续完成两项关键修复：

- `d658aed`：增加显式 `mono_native / mono_cec` authority arm。Native 臂保持相同
  causal-monocular depth 与目标输入，但跳过 certificate 和 direct-local bearing；不再用
  “CEC 恰好 reject”冒充 native 对照；
- `7a2f827`：增加 outcome-blind paired-campaign verifier。预注册 JSON 永远保持结果空白，
  SR/SPL、`+gain/-loss` 与 exact McNemar 只能从 40 个 finalize 后的 hash-sealed run 独立
  复算；verifier 同时检查 Odin `S_i/L_i/P_i/SPL_i`、场景/目标/dataset 绑定、显式
  authority mode，以及 native 臂零 CEC takeover。

相关非 ROS 单元/协议测试为 `146 passed`；公开 release verifier 为 `failures=0`。三个依赖
ROS `message_filters` 的 Jetson-side collection tests 无法在当前桌面 conda 环境收集，这一
点属于环境边界，不是此次代码回归。当前 plan-only verifier 读出 40 个结构合法的注册
run 和 0 个 outcome，并明确列出四个 scene registry 尚未冻结；因此依旧没有可报告的
真机 SR/SPL。剩余的真实 P0 已压缩为物理工作：独立 arrival 标定/held-out confirmation、
候选时刻 pose 收据、scene freeze，然后才执行 20 个 matched pairs。
