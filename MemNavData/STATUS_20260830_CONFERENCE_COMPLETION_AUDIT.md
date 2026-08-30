# 2026-08-30 会议实验完成度审计

> **13:02 CST no-fallback execution addendum.** 作者要求继续完成原始会议清单，不能把
> constructibility null、低样本 smoke、旧结果或删表当作完成。为此，Table II 不再停在
> 下文第 3--6 节所述的“保留 null”方案。一次早于任何扩样 factual-B/Leg-3 outcome、且已
> 独立复算的 Natural-B construction audit 封存了 84 个额外候选，覆盖 53 个 source
> recipients / 30 scenes（front/side/rear `19/14/51`，max covis `<.10`）。完整执行协议
> `hm3d_fullmono_lifelong_natural_b_expansion_execution_protocol_20260830.json` 保持原
> `.10/.55/.90` support 门、16 histories / 10 scenes / each-stratum>=3 的 Leg-3 门、
> 600-step budget、mono-native/mono-CEC arms 全部不变。
>
> 已提交的正式链为 smoke `16591442` -> 54-scene materialization `16591445` -> seal
> `16591447` -> independent verifier `16591452` -> factual-B deferred launcher
> `16591458`。Smoke 已在 H100 `gh003` 上 `47 s` 正常完成，复现 scene 0 的 `2`
> candidates / `2` recipients，且 `navigation_outcomes_read=false`；formal array 已启动。
> 后续链只有在逐级 verifier 通过后才会依次提交：84 次 exactly-once factual-B、84 个
> factual-prefix support checks、原始 22 + 扩样 supported prefix 的 exact union、完整
> Table-II Leg-3 construction、power/strata gate、以及同 prefix 的 mono-native/mono-CEC
> paired policy evaluation。任何一级不足都 fail closed，并继续扩 source population；不
> 降阈值、不挑结果、不把失败链或 B2 支线替代表格。
>
> 权威 submission receipt：
> `HM3D_NATURAL_B_EXPANSION_EXECUTION_SUBMISSION_20260830.json`；bundle receipt SHA：
> `1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282`；run root：
> `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7`。
> 在最终 policy independent verifier 出现前，Table II 仍是“未完成”，不得读取 partial
> SR 或把 84 个候选写成 navigation population。
>
> 提交后静态审计在任何 factual-B 输出出现前发现首个 deferred launcher 的 shard size
> `1` 与冻结 raw-file verifier 的原始 max-`2` 合约冲突。旧 launcher `16591458` 以
> elapsed `0 s` 精确取消；materialization 与其 verifier 保留不动。修复只把 shard size
> 恢复为 `2` 并从 sealed shard manifest 动态派生 array 长度，replacement launcher
> `16591609` 继续依赖同一 `16591452`。修复 bundle SHA 为
> `49ca32aae755d5f67c913480adc1b15c23ba4ed7ea0aee9dab065f32d8669665`；
> `factual_b/`、`prefix_fragments/` 与 Leg-3 outcomes 在修复前均不存在。这是零 outcome 的
> infrastructure repair，不改变候选、threshold、controller 或统计口径。

> **15:40 CST full-completion addendum.** 作者进一步明确：原会议 Table II、Table III
> 与真机目标必须按原定义完成，不能再以 constructibility null、旧结果、低样本 smoke、
> 删项或降门槛代替。当前执行因此分成三条互不替代的链：
>
> 1. **Table II expansion。** 46-shard、84-candidate factual-B array `16592875` 正在
>    完成最后分片；正式运行中出现的 `exit=2` 是 gh012 上服务已监听但旧 `ss` 探针
>    假阴性，`exit=1` 是 Habitat/EGL transport crash。两类都不作为 navigation outcome。
>    只按 completion 文件存在性与 byte hash 选择缺失 identity 的修复 launcher
>    `16594267` 已冻结，等待原数组 `afterany` 后精确补齐；不读取 SR、不改候选或门。
> 2. **Table III actual-mono length benchmark。** 已在不读取导航结果的 100-scene HM3D
>    capacity graph 上冻结 125 个 reserves；三个 bin 的预评估容量分别达到
>    `16 histories / 16 scenes`、`16/16`、`16/12`。首次 factual-A gate 在策略执行前
>    暴露 capacity navmesh 与 evaluator runtime rebake 不一致；第二次证明 pinned
>    navmesh 可用后，又在 evaluator import 前暴露 immutable overlay 漏装本地依赖。
>    两次均没有 policy outcome。v3 bundle 递归封存 36 个本地 import dependencies，
>    并在提交前于远端容器执行真实 `eval_2leg_habitat.py --help` import preflight。
>    正式 factual-A jobs 为 `16595631 -> 16595636`；完整下游 DAG 为
>    `16595695`（125 constructions）-> `16595706`（48-history population seal）->
>    `16595711`（独立 population verifier）-> `16595720`（48 paired histories / 96
>    queries）-> `16595722`（SR/SPL/McNemar aggregate）-> `16595726`（raw verifier）。
>    强化 raw-row verifier `16595792` 再从全部 192 arm-role metric rows 独立复算 SR、
>    SPL、gain/loss、exact McNemar 与 authorization counts。全部依赖均为 `afterok`；
>    `partial_results_allowed=false`、`fallback_completion_allowed=false`。
> 3. **Real robot。** 软件已冻结 4 scenes x 5 matched blocks（20 pairs / 40 autonomous
>    rollouts），但 outcome 仍为 0/40。该项唯一剩余依赖是现场：独立 arrival/path
>    calibration、四场景 registry 和有人值守的 paired motion。静态、no-motion 或手动
>    接管 trace 均不会补位。
>
> 截至本 addendum，Table I、Depth、CEC mechanism 已完成；Table II 与 Table III 在正式
> HPC 链中；真机仍等待物理执行。论文只会接收最终 independent verifier 通过的数字，
> Abstract/Introduction 保持作者当前版本不动。

> **16:10 CST Table-III native-crash repair addendum.** v3 factual-A gate
> `16595631_0` 在 `gh009` 上完成 pinned-navmesh 验证、服务启动和约 18 分钟真实策略推理
> 后，Habitat evaluator 收到 native `SIGABRT (-6)`；作业未 OOM、未超时，stderr 为空，
> 且没有写出 `completion.json`。依赖的 remainder 和原下游 DAG 因此全部取消，任何
> partial artifact 都没有进入 population 或 SR。Clean repair 沿用完全相同的 125 个
> candidate identity、seed、GLB/navmesh hash、模型、600-step contract、长度桶和门限；
> 唯一运行时加固是 `PYTHONFAULTHANDLER=1`，并将 GPU 调度固定到已稳定使用的项目
> A100 partition，以便 native crash 若重现时留下可审计栈。新的正式 factual-A gate/
> remainder 为 `16596239 -> 16596273`，run root 为
> `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_execution_20260830/formal_20260830T080030Z_8ff97ca6`。
> 重接的完整 DAG 是 `16596354 -> 16596365 -> 16596374 -> 16596376 -> 16596377 ->
> 16596380 -> 16596401`；最后一级仍从 192 个 raw arm-role rows 独立复算所有正式数字。
> 新收据继续声明 `partial_results_allowed=false`、`fallback_completion_allowed=false`、
> `threshold_relaxation=false`。

本文件只回答两个问题：会议清单现在完成到哪里，以及下一份 GPU/真机时间应该花在哪。

## 1. 当前完成度

| 会议项目 | 状态 | 最准确结论 |
|---|---|---|
| Table I：跨 controller / dataset | **完成** | HM3D 与 MP3D 上 NavDP、ViNT 的 native/CEC 四行均已 sealed、独立复算；只作 controller 内 paired claim |
| Table II：HM3D continual by leg | **正式运行中，powered gate 与 construction verifier 已通过** | exact union 为 41 supported A+B histories / 20 scenes；41/41 construction 完成后封存 20 个可构造 C histories / 13 scenes，A100 修复链 smoke 正在运行 |
| Real robot | **未完成** | transport、hash、fail-stop 与启动框架可用；尚无冻结的 paired autonomous outcome |
| Depth ablation | **完成** | 同 Final14 query population 的 metric/mono/zero 与 CEC 对照完成；Goal-A history 仍是 metric replay，须保留边界 |
| CEC mechanism | **完成** | proposal、finite-PnP witness、strict authority 与闭环已统一复算；authority 行为成立，阈值特定 SR superiority 未显著 |
| Length buckets | **正式运行中** | 125 个 actual-mono reserves 已冻结并执行；完成 receipt 修复后将严格封存三个桶各 16 histories，再跑 96 个 paired queries |

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
- `ed80bbf`：关闭 exact-goal formal 启动门。每轮必须提供 registry 中的 scene/run ID、
  external frozen goal、goal SHA 与 sealed-dataset SHA；source、Jetson installed goal、RTX
  active goal 或 dataset manifest 任一不一致都会在运动仍锁止时失败。Novel/Revisit 不再
  对应两个 launcher 分支，而只是同一种 frozen goal 是否获得 causal-history support；
  runtime 不读取 role。Survey 自动候选只保留为 lifelong/engineering demo。

相关非 ROS 单元/协议测试为 `146 passed`；公开 release verifier 为 `failures=0`。三个依赖
ROS `message_filters` 的 Jetson-side collection tests 无法在当前桌面 conda 环境收集，这一
点属于环境边界，不是此次代码回归。当前 plan-only verifier 读出 40 个结构合法的注册
run 和 0 个 outcome，并明确列出四个 scene registry 尚未冻结；因此依旧没有可报告的
真机 SR/SPL。剩余的真实 P0 已基本压缩为物理工作：独立 arrival 标定/held-out
confirmation、scene goal/start/path freeze 与 Odin 现场验收，然后才执行 20 个 matched
pairs。若 formal 全部使用 external frozen goal，自动 candidate 的 SportModeState pose
收据不再阻塞这张主表，只阻塞另行报告的自动 lifelong target-selection demo。

## 8. 16:40 CST 原始会议目标无 fallback 执行更新

本节覆盖上文早期的“低优先级/可删项/constructibility null 可替代”等排期判断；旧文字仅
作为过程审计保留。当前唯一完成标准仍是原会议清单：Table II、Table III 与真机正式
双臂都必须得到预注册规模、完整原始行和独立 verifier，不能以 smoke、旧结果、低样本、
删项或降低门限补位。

### Table III：正式数组继续，缺失身份已自动化精确修复

- A100 gate `16596239` 已完成；正式 125-candidate factual-A array 为 `16596273`。
- index `2` 在任何 completion receipt 写出前因两个独立数组的算术端口碰撞退出；这不是
  policy outcome。其余已完成项不会重跑，数组仍继续执行全部冻结 reserves。
- 运行器现统一使用节点本地、持有到两个 server 退出的 TCP pair `flock`；collection 和
  paired-query 均不能再由调用方注入算术端口。修复提交对应 Git commit `94b44dd`。
- outcome-blind exact-repair launcher `16597086` 已以 `afterany:16596273` 提交。它只按
  `completion.json` 与 sidecar SHA 的字节完整性冻结 missing indices，完全不反序列化
  `reached_A`；半成品先逐字节归档，只在 A100 上重跑缺失 identity。repair 完成后必须
  先验证 `125/125` receipt，才会重新提交 125 constructions、48-history population、
  96 paired queries、aggregate 与 raw-row independent verifier。实现 commit 为
  `977c49a`；本地相关契约测试 `23 passed`。

### Table II：H100 native crash 与科学结果分离，转 A100 精确补缺

- 第一轮 outcome-blind repair array `16596509` 的 cell 0 在 gh009 运行 18:28 后由
  Habitat evaluator `SIGABRT (-6)` 退出，没有 completion receipt；这与此前 H100 原生
  crash 指纹一致，不是模型失败。该数组已有一个 failed cell，原 `afterok` finish 已不
  可能成立。
- 新 A100 repair launcher `16597232` 已先以 `afterany:16596509` 提交；随后才精确取消
  旧数组剩余 cell 与失效 finish `16596510`，避免继续占用 H100。新 launcher 会再次按
  receipt 完整性枚举所有仍缺失 history，并归档被取消/崩溃的 partial output；候选、seed、
  600-step budget、controller、certificate 与统计门限均不改变。
- 新 repair tag 为 `transport_repair_a100_20260830`，GPU 固定 `a100_tandon`，并使用同一
  生命周期端口分配器；提交 receipt 明确
  `navigation_outcomes_read_at_submission=false`、
  `scientific_thresholds_changed=false`、`fallback_completion_allowed=false`。实现与
  receipt commit 为 `a7209c8`。launcher 已完成并由独立 verifier 确认仍缺 24 histories、
  分为 11 个 scene groups，`navigation_outcomes_read=false`；正式 A100 repair/finish 为
  `16597248 -> 16597249`。

### 真机与论文

- 真机仍为正式 `0/40`，不是失败数字而是尚未物理执行；软件侧 formal CLI 修复已推送
  `MemNav-RealWorld main` commit `9d83edb`。完成仍需用户在场、Tailscale 认证、独立
  arrival/path calibration、4 个场景冻结与 20 对自主 rollouts，任何 no-motion 或手动
  trace 都不会替代。
- 在 Table II/Table III 最终 verifier 出现前不读取 partial SR。论文 Abstract 与
  Introduction 保持作者版本不动；最终仅把 verifier-sealed 数字写入会议矩阵与表格。

## 9. 17:45 CST 运行进展与定性素材闭环

### Table II 已越过 factual-B 基础设施故障

- A100 exact-repair array `16597248` 的 11 个 scene-group cells 已全部
  `COMPLETED`，耗时 5:16--13:11，均写出正常 completion；此前 H100 的
  `SIGABRT` 没有在 A100 repair 上复现。
- finish `16597249` 已 `COMPLETED`，随后启动 84 个 factual-prefix construction
  `16598398`、population verifier `16598399` 与 union/Table-II launcher `16598400`。
  17:45 CST 时 construction 已推进至 index 76，剩余 `77--83` 等待 GPU quota。
- 这仍不是最终 Table II 结果：只有 population verifier、C-query paired array、summary
  与 raw-row independent verifier 全部通过后才能读取 SR/SPL。当前没有查看任何 partial
  outcome，也没有缩小 population。

### Table III 原数组继续，失败项保持为待精确补缺 identity

- `16596273` 仍按 `%2` 执行冻结的 125 factual-A candidates。旧 immutable wrapper 中
  仍可能出现 pre-receipt server-start 失败；截至本次审计已看到 indices
  `2/23/40/42/43` 在 completion 前退出，其日志没有形成可用 policy outcome。
- 不对这些 identity 做删除、换样或事后解释；`16597086` 会在整个原数组结束后按
  receipt/SHA 完整性一次性冻结全部 missing indices，再由带动态端口锁的新 wrapper
  精确补跑。其后仍须得到 `125/125` factual receipts 才能进入 48-history/96-query
  正式表。

### Motivation failure case 不再是空缺

从已经冻结并消费的 Final14 中选择了一个**仅用于定性展示**、不用于模型或门限选择的
paired history。它同时满足：

- unsupported Novel：mono Native/Raw/CEC 为 `success/failure/success`；CEC 从第一步
  拒绝 memory，完整 query trace 与 Native 逐点相同；
- supported Revisit：mono Native/Raw/CEC 为 `failure/success/success`；CEC 选择历史
  anchor 后获得 529 个 PnP inliers、query/reference coverage `34.0/36.1%`、RMSE
  `1.09 px` 并授权 unit bearing；
- Novel 的局部匹配虽有 56 个 PnP inliers，但 reference coverage 仅 `3.8% < 5%`，因此
  operational certificate 正确 abstain。这个案例直接展示“同一历史既可能有用，也可能
  干扰；proposal 不等于 control authority”。

可复算生成器：

- `MemNavData/build_final14_qualitative_case.py`；
- `MemNavData/build_final14_qualitative_video.py`。

内部 hash-sealed 原始素材与 receipt：

- `.diagnostics/conference_qualitative_20260830/history_002/`；
- `.diagnostics/conference_qualitative_20260830/rendered/`。

已生成 300-dpi PNG、vector PDF 和 8 秒 H.264 MP4；图中不暴露 scene identity，数值标注
读取 formal receipt，重新绘制的 CUDA matcher 仅用于 inlier 线条并单独记录其轻微数值
差异，不新增任何 scientific row。论文将该图前置为 Fig. 1，并在完全不修改
Abstract/Introduction 的前提下压缩重复的实验协议、limitations 与 caption；Tectonic
重新编译为严格 8 页，正文占 1--6 页、References 从第 7 页开始。会议交付物没有被删掉
或移入不存在的 supplementary。

## 10. 18:26 CST powered Table-II gate 与 A100 policy 固定

本节是当前权威运行状态，覆盖上文仍把 Table II 写成 22-history constructibility null 的
历史描述。

### Table II 已形成预注册规模的 result-blind population

- 84 个 factual-B rollout、84 个 factual-prefix check 与独立 population verifier 已全部
  完成；扩样产生 `19 histories / 13 scenes`，其中 `15` 条为 strong support。
- 与原始 sealed population 做 exact union 后，Table-II population 为
  `41 histories / 20 scene clusters`；population SHA 为
  `341f8102102948fc6467895833c91c9816eb8c5fa9f4505422e8fccf4d3cc0b5`。
- union verifier 明确记录 `target_met=true`、`result_blind=true`、
  `leg3_query_navigation_outcomes_read=false`。因此这不是旧 8-history/6-scene null 的包装，
  而是已经越过原始 `>=40 histories / >=15 scenes` power gate 的新 population。
- hidden-role C-query construction array `16599077_[0-40%4]` 正式运行；18:26 CST 已推进至
  indices `23--26`。它完成后依次执行 finalize `16599078` 与 independent verifier
  `16599079`。在 verifier 前仍不读取任何 query policy outcome。

### 正式 policy evaluation 只固定基础设施，不改变实验

H100 上已两次出现 Habitat evaluator native `SIGABRT`，而相同 identity 的 A100 exact
repair 已全部完成。为避免已经 powered 的 Table-II policy 链再次掉入相同基础设施故障，
在 construction verifier 触发前完成了 A100-only replacement：

- 新 immutable launcher bundle：
  `hm3d_table2_policy_a100_4e11647ad496f6f9`；receipt SHA
  `4e11647ad496f6f9e0dd0938e58a026a3d7a7098828091659069c5270811add8`；
- 新 launcher `16600350` 继续只依赖原 independent verifier `16599079`；
- 旧 mixed-partition launcher `16599080` 在新 job 已被 Slurm 接受后才取消，状态为
  `CANCELLED`；没有运行过 policy；
- replacement receipt：
  `HM3D_TABLE2_POLICY_A100_REPLACEMENT_SUBMISSION_20260830.json`。

替换仅增加 `--partition=a100_tandon`。41-history population、Natural Novel/Revisit C、
`mono_native/mono_cec` 两臂、actual-A/B history、600-step budget、hidden role、certificate
门限、aggregate 与 raw independent verifier 均不变；没有删行、降低阈值、读取 partial
SR 或允许 fallback。

### Table III 与真机仍未完成

- Table III 原 125-candidate array `16596273` 已运行到 index `71`；`72--124` 继续排队。
  精确补缺 launcher `16597086` 仍等待原数组 `afterany`，不会删除任何失败 identity。
- 真机目标仍是 4 scenes x 5 matched blocks = 20 pairs / 40 autonomous rollouts。Jetson
  当前离线，且正式运动必须作者现场确认；因此该行仍是“尚未执行”，不能用 plan-only、
  no-motion、手动 trace 或仿真替代。

## 11. 18:55 CST Table-II 完整会议口径与最终 launcher

本节覆盖第 10 节中仍指向 `16600350` 的 launcher 状态；科学 population 与 policy
contract 没有变化。

### 41/41 construction 已完成，正式 policy 尚未读取

- `16599077_[0-40%4]` 的 41 个 hidden-role C-query construction 全部完成；finalizer
  `16599078` 与独立 construction verifier `16599079` 均以 `exit=0` 完成。41 条是输入的
  supported A+B pool；Natural Novel/Revisit 两个查询都能按冻结方向约束构造的最终
  population 是 `20 histories / 13 scene clusters`，超过预注册 `>=16 / >=10` 门。
- construction 只建立查询与历史，不运行 `mono_native/mono_cec`，因此截至本节仍没有
  partial policy SR 被读取。
- A100 launcher `16601041` 成功建立第一条 policy DAG，但 smoke `16601072` 在两个模型
  server 正常启动后、evaluator import 阶段暴露旧 task bundle 漏封
  `MemNavData.cec_handoff_contract`；formal 及其下游自动取消，正式 policy rows 为 `0`。
- 第一轮 exact repair 封存了 handoff contract 并通过 evaluator import preflight，但 smoke
  `16601475` 在约 100 秒 memory replay 后、第一次在线 `/memory_step` 才触发另一个已知
  namespace 问题：新版 server 延迟导入 `monocular_depth_runtime`，却解析到旧 base bundle
  中不含 `bind_monocular_depth_transaction` 的同名文件。formal rows 仍为 `0`。
- 第二轮 exact repair 使用新 run root `policy_runtime_repair_v2`，显式封存当前 mono-depth
  transaction runtime。smoke `16601690` 已完整跑完 native 的两个 80-step queries，CEC
  arm 在 reset 时暴露同类 namespace fallback：旧 `certified_relocalization_runtime` 缺少
  authority-policy 两个字段。formal rows 仍为 `0`。
- 第三轮不再逐文件猜依赖，而是显式绑定已完成 Table-I 正式闭环的完整
  authority-transaction closure `82e71f19...`。八个关键 runtime 文件与当前仓库 SHA
  逐字节一致；生产环境检查同时约束 evaluator import、mono transaction API、certificate
  authority contract 与 PnP module provenance。launcher `16602096` 已生成 smoke `16602104`、
  formal `16602105_[0-53%4]`、aggregate `16602106`、raw verifier `16602107` 与内含修正版
  meeting verifier `16602108`。截至 20:03 CST，smoke 等待 A100 配额。
- 当前 repair immutable bundle：
  `hm3d_table2_policy_import_repair_e8c4530f3ffc4bfe`，receipt SHA
  `e8c4530f3ffc4bfe2307b81203bb921815616ace412a43f877e46bd471955773`。

### 新增 post-seal meeting verifier，防止 Table II 漏报 Leg-1/Leg-2

原 policy verifier 的 estimand 有意只覆盖
`C | successful, supported factual A+B`。它不能单独填完整会议 Table II。新增的独立
meeting verifier 只在最终 policy raw verifier 通过并封存后运行，联结以下哈希绑定输入：

- actual-mono Goal-A source：`131/196`；
- result-blind factual-B candidate rollouts：`54/183`；上游 materialize 了 `130` 条 eligible
  成功 A history，实际 183 个 B candidates 覆盖 `67` 条唯一 A，且存在重复 candidate；
- 其中进入 Leg-3 构造的 supported A+B histories：`41`；
- 最终 policy verifier 逐 role 复算的 Natural Novel / Revisit / balanced-all Leg-3。

这里 `183` 个 B 是为同一批成功 A prefix 冻结的不同 factual-B candidates，部分 A 会被
重复使用。因此 verifier 明确禁止把 `131/196`、`54/183` 与条件 C 机械相乘成一个不存在的
unconditional three-leg joint cohort；它报告的是会议要求的逐段 factual waterfall 和
同 prefix 条件 C 效应。这是补齐可辨识口径，不是 fallback，也不修改任何 rollout。

修正版 verifier 已用全部真实 sealed upstream receipts 和一个临时零值 policy receipt 做
pre-policy dry-run，独立复现 `131/196, 54/183, 130 eligible, 67 covered, 41 supported,
20 constructible`；没有读取真实 Leg-3 outcome。

相关本地回执与测试：

- `HM3D_TABLE2_POLICY_AUTHORITY_CLOSURE_V3_SUBMISSION_20260830.json`；
- `independent_verify_hm3d_table2_meeting_result.py`；
- synthetic provenance / escaped-identity / absolute-sidecar 三项回归测试全部通过。

### Table III 同步状态

原 125-candidate factual-A array 已启动到 index 84：当前 `69 COMPLETED / 13 FAILED /
2 RUNNING`，其余保持 array throttle 排队。13 个失败 identity 不删除、不替换；exact-repair
launcher `16597086` 等待原数组 `afterany` 后按 completion receipt 与 byte hash 精确补齐
同一身份，再封存三个长度桶各 16 histories。

## 12. 20:38 CST Table-II 正式运行与 Table-III directed-geodesic 精确修复

### Table II 已进入正式 policy array

- runtime-closure v3 smoke `16602104` 在 A100 上以 `exit=0` 完成，用时 `10:56`；此前三次
  smoke 暴露的 handoff、mono-depth transaction 与 authority-policy namespace 缺口均未
  复现。
- 正式 array `16602105_[0-53%4]` 已启动，后续 aggregate `16602106`、raw verifier
  `16602107` 与 meeting verifier `16602108` 仍保持依赖锁。运行中只查看 scheduler 与
  infrastructure log，不读取任何 partial SR。

### Table III 原数组完成，19 个缺失身份全部保留

- 原 factual-A array `16596273` 已结束：125 个冻结 identity 中 `106` 个形成带 SHA 的
  completion receipt，`19` 个在 receipt 前退出。这个计数来自文件完整性，不解析
  completion outcome。
- 失败日志指纹证明旧的单一“transport collision”归因不完整：index `2` 是端口占用；
  另外 18 个 identity 是 `capacity/factual Goal-A geodesic changed`。它们不是导航失败，
  也不能通过再次运行旧代码解决。
- 对同一 pinned navmesh 的 outcome-blind 复算定位了根因：capacity 收据测量
  `query_start -> first_goal`，factual Goal-A 实际走
  `first_goal -> query_start`。Detour 的端点投影使这两个有向测量在 18 条中相差
  `0.064--1.641 m`；旧 collector 错误地用反向距离校验正向收据。
- 修复保留原 `0.05 m` 门槛，并在收据的原方向重算；反向距离只用于实际 Goal-A 路径。
  Candidate、goal、三个 length bins、seed、step budget、controller、success radius 与模型
  均未改变。冻结契约为
  `hm3d_table3_directed_geodesic_repair_contract_20260830.json`。
- 已取消会确定性重现错误的旧 launcher `16597086`。新 immutable bundle 为
  `hm3d_table3_actual_mono_a_transport_repair_3f481a0da0b5b9d0`，receipt SHA
  `3f481a0da0b5b9d00aa9ec7b0c71f3b45c02ed561c79a736a3a3ca38d2a5d66d`。
  Launcher `16603024` 从 completion receipt+SHA 冻结全部 19 个 missing identities，提交
  A100 exact-repair array `16603035` 与 finish verifier `16603036`。重复 launcher
  `16603049` 因 immutable repair plan 已存在而在导航前拒绝，不产生污染。

### 真机软件链恢复到 formal-ready、运动仍锁止

- Workstation、GitHub 与 Jetson 已同步到 `MemNav-RealWorld` commit `d81a201`；设备工作树
  干净。新增 camera-recovery 只允许 fail-closed 操作。
- Jetson 实机发现 RGB-D 帧已 stale 约 212 秒。调用
  `/navdp_camera_recovery/restart` 后验证新 RGB/depth `11/10` 帧，恢复后
  `rgbd_age=0.027 s`，并继续保持 `enabled=false, estop=true, cmd=0`。
- 这关闭的是采集基础设施风险，不是导航结果。Odin1 当前未连接，arrival/path 标定和
  4 scenes x 5 matched blocks 的 `40` 次正式 rollout 仍未执行，不能由旧工程轨迹替代。

## 13. 20:53 CST 两条正式仿真链同步推进

### Table II：38/54 formal queries 完成，零失败

- `16602105_[0-53%4]` 当前为 `38 COMPLETED / 2 RUNNING / 14 PENDING / 0 FAILED`。
- aggregate `16602106`、raw verifier `16602107` 与 meeting verifier `16602108` 仍由
  `afterok` 锁住；未读取任何 partial policy outcome 或 partial SR。
- 正式数值只会在三个最终节点全部 `exit=0` 后进入论文；否则只允许对原 identity 做
  基础设施精确修复，不删行、不换场景、不降门槛。

### Table III：directed-geodesic repair 继续运行，下游完整 DAG 已冻结提交

- `16603035` 的 19 个原 identity 当前为
  `14 COMPLETED / 3 RUNNING / 2 PENDING / 0 FAILED`；完成审计 `16603036` 仍等待该数组。
- 26 项本地提交前测试通过。新的下游 immutable bundle 为
  `hm3d_table3_actual_mono_downstream_a6583b0b516371de`，receipt SHA
  `a6583b0b516371de60f43e46a070b1711a8697a46f54ee0ae6c930a70d8ade2e`。
- 已提交完整正式链：construction `16608231_[0-124%4]` -> population finalize
  `16608333` -> independent population verifier `16608340` -> paired query
  `16608351_[0-47%4]` -> analysis `16608364` -> independent result verifier
  `16608376`。首节点严格依赖 repair verifier `16603036`，因此不会读取不完整 factual-A。
- 冻结规模保持为 `125 candidates -> 48 histories -> 96 role queries -> 192 paired
  arm-role rows`，三个长度桶各 16 histories；`partial_results_allowed=false`、
  `fallback_completion_allowed=false`、`threshold_relaxation=false`、
  `smoke_substitution=false`。旧的已取消下游回执保留作审计记录，未覆盖。

## 14. 21:17 CST Table-III final-run-root binding repair

第 13 节记录的第一条 downstream DAG 在首批 construction 暴露了一个提交器默认值错误，
随后被完整取消；它不是科学结果，也没有任何 query arm 运行。

- `16603035` 的 19/19 exact repairs 全部 `COMPLETED`，finish verifier `16603036` 以
  `exit=0` 验证最终 factual-A run root 的 125/125 completion receipts。
- 第一条 downstream receipt 错把 `A_RECEIPT` 默认指向早先、已 supersede 的
  `formal_...8c8f8c25`，而实际完成数组 `16596273` 属于
  `formal_...8ff97ca6`。首批 construction 因旧 root 中 completion 缺失/歧义在导航查询前
  退出；整链 `16608231/16608333/16608340/16608351/16608364/16608376` 已取消，未产生
  policy query outcome。该 receipt 保留用于失败审计。
- 修正版提交器默认并哈希绑定
  `HM3D_TABLE3_ACTUAL_MONO_A_SIGABRT_REPAIR_SUBMISSION_20260830.json` 与
  `HM3D_TABLE3_ACTUAL_MONO_A_DIRECTED_GEODESIC_REPAIR_SUBMISSION_20260830.json`，要求二者
  同时指向原数组 `16596273`；它还独立校验 repair completion 的 125 个 byte receipts。
  已完成的 Slurm job ID 不再被误当作活跃 dependency，而以不可变 completion receipt
  作为更强的启动门。
- constructor 不再用 scene-prefix glob 搜索 factual completion；它现在按冻结的
  `history_index + scene + episode` 解析唯一绝对 identity，并验证 completion sidecar。
  Candidate、role、distance bin、threshold、controller、seed、budget 和 success definition
  全部未改。27 项测试通过。
- 正确正式链为 construction `16609158_[0-124%4]` -> population finalize `16609170` ->
  independent population verifier `16609184` -> paired query `16609194_[0-47%4]` -> analysis
  `16609203` -> independent result verifier `16609207`。其 run root 是
  `formal_20260830T080030Z_8ff97ca6`，wrapper SHA 为
  `2d8d08ff5a65da0ad00e7372fd756174b88958ed4cf74076e6605c951b38c3fe`。
- 同期 Table II 为 `50 COMPLETED / 3 RUNNING / 1 PENDING / 0 FAILED`；仍未读取 partial SR。
- 旧的 retained-history node-affine `16540468`（N=1 smoke）与依赖 launcher
  `16540469` 自 8 月 28 日起始终未运行。它们服务于 17-history underpowered mechanism
  线，文档已明确禁止用来填 powered Table II；为避免抢占同一用户 GPU 配额，21:25 CST
  精确取消这两个 pending jobs。没有删除既有数据或正式结果，也没有改变任何会议 population。

## 15. 21:45 CST Table II powered result sealed

- policy array `16602105` 54/54 完成，aggregate `16602106`、raw verifier `16602107` 与
  meeting verifier `16602108` 全部 `COMPLETED 0:0`；结果只在三个节点通过后首次读取。
- factual waterfall 为 Novel-A `131/196`，Novel-B `54/183`。B 的 183 个 candidates 覆盖
  67 个唯一 A prefixes，故禁止把两个阶段率相乘成 unconditional joint。
- 在 20 histories / 13 scenes 的 `C | successful, supported factual A+B` population 上：
  Novel native/CEC 均为 `4/20`（`+0/-0, p=1`）；Revisit 为 `8/20 -> 17/20`
  （`+10/-1, p=.01171875`）；balanced all 为 `12/40 -> 21/40`
  （`+10/-1, p=.01171875`, risk difference `+22.5 pp`）。
- balanced SPL `.1572 -> .4236`，Revisit SPL `.1422 -> .6750`。Runtime role hidden；
  Novel takeover 1/20，Revisit takeover 18/20，21 queries 为全部拒绝后的 byte-exact fallback。
- meeting verification SHA：
  `a3b4adf9f5c29cab775da30fc19fd60704201070b87c35aa755ffe1e34457f50`；
  `verified=true`，无 fallback completion、threshold relaxation 或 partial outcome。
- 完整结果与 claim boundary 见
  `HM3D_TABLE2_POWERED_MEETING_RESULT_20260830.md`。

## 16. 22:07 CST Table II 逐段 SPL 补充审计完成

- 原会议清单要求逐 leg 同时报 SR、SPL 与有效分母；primary meeting verifier 已封存 A/B
  factual SR，但没有汇总其 SPL。该缺口不能用空值冒充完成。
- 新增只读 post-seal verifier `independent_verify_hm3d_table2_stage_spl.py`，哈希绑定
  primary meeting verifier，并逐条复核 `196` 个 actual-mono Goal-A raw metrics/traces 与
  `183` 个 result-blind factual-B raw metrics/traces。它没有重跑 policy、没有选择 prefix、
  没有读取下游 query 来改变 population。
- HPC CPU job `16609841` 在 `00:00:32` 内 `COMPLETED 0:0`。独立复算得到：Leg-1
  `SR=131/196=.66837, SPL=.60106`；Leg-2 `SR=54/183=.29508, SPL=.19211`。
- 封存 verifier SHA 为
  `321f40534e1d467f4bf47b58414ac09f714b54c4b032460ee67235debd9f97ee`；A/B raw
  artifact-set SHA 分别为
  `1a4ccff0f4bfb67d18cd010f7ef0a921d2cdd4e8c526c3e84e07c08462800876` 与
  `16a9daba00f0cd59f50be319d9c781ece0f9474cf1a72ce73ee1a6c39c117691`。
- 至此 Table II 的会议字段（逐段分母、SR、SPL、Leg-3 hidden-role Novel/Revisit paired
  contrast）均已有封存证据；仍不得把分层 waterfall 相乘为 unconditional 3-leg joint。

## 17. 23:50 CST Table III 历史来源边界与 causal-survey 正式链

### 原 actual-mono source gate 的结论只到 constructibility

- `125/125` factual-A 与 `125/125` construction receipts 已完整。按
  `0--20/20--30/30--50 m` 三桶，factual-A reached 为 `8/40, 4/40, 3/45`，满足至少
  40 帧的可用 history 为 `8/40, 2/40, 2/45`；最终可构造 role pair 只有
  `2/40, 0/40, 0/45`。
- 该链没有生成任何 query-policy metric 或 completion。它证明原设计把长度分析条件在
  “NavDP 先自主完成一个同样长的 Goal-A”上后，source population 无法达到三个桶各
  `16 histories / 10 scenes`；它不是 CEC 的 SR 结果，也不能以 2 条短桶 history 填表。

### 新链测量历史跨度，不冒充 actual NavDP Goal-A

- 新协议 `hm3d_table3_causal_survey_protocol_20260830.json` 保留原三个 query geodesic
  bins、每桶 16 histories / 10 scenes、每 scene 每桶至多 2 条、Novel/Revisit 距离匹配、
  `covis < .10 / >= .55`、hidden role、mono-native/mono-CEC、固定 controller、success
  radius 与按距离增长的预算；唯一改变并明确披露的是 history source。
- History 是沿冻结 geodesic、按 `0.0376 m / 4.5 deg` 上限生成的物理有序 causal RGB
  survey。构造阶段使用 simulator pose/depth 生成与审计 query；运行时 replay 只向
  LingBot/CEC/NavDP 发送 RGB，既不发送 pose，也不读取 survey depth。它回答的是
  “相同 controller 在不同历史/查询跨度上能否利用被观察过的目标”，不是
  “NavDP 能否先自主完成 20--50 m Goal-A”。若会议单元格坚持后一个 estimand，这条链只能
  作为额外 controlled length analysis，不能改名替代。
- Construction-only DAG 为 `16610812_[0-124%4] -> 16610818 -> 16610823`；run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_20260830/formal_20260830T150042Z_3d811e0f`。
  23:50 CST 时 `35/125` completion 已写出，均属于首个短桶，其中 `30` 条可构造、覆盖
  `30` scenes；无 failed array cell。其余因账户 GPU QOS 当前只给一张 A100 而串行推进。
- 正式 query 尚未提交。只有 `16610823` 写出并哈希封存
  `verified=true, formal_policy_evaluation_authorized=true`，且三个桶严格各 16 histories 后，
  `submit_hm3d_table3_causal_survey_queries_hpc.sh` 才能提交 `48 histories x 2 roles x 2 arms
  = 192` 个 raw rows；不存在 smoke、partial 或降低门限入口。
- 为避免语义漂移，aggregate/verifier 已拆成 survey 专用文件：
  `analyze_hm3d_table3_causal_survey.py` 与
  `independent_verify_hm3d_table3_causal_survey_result.py`。后者重新读取全部 metric、plan、
  replay 与 completion artifacts，复核 RGB-only depth receipts、role 未前传、四个
  arm-role replay 的 history equality、reject exact fallback、SR/SPL、paired gain/loss、McNemar 与
  scene-cluster bootstrap。相关纯契约回归测试当前 `54 passed`。

### 00:30 CST 后续预检

- construction 已推进到 `40/125` completion，仍为 `0` failed task；array 任务 40/41
  正在 A100 上运行。此时中长桶尚未开始，因此没有 query SR，也没有提前读取结果。
- query-side 独立 verifier 进一步改为从 `query_result.final_goal_dist_m <= 1.0 m`
  重新计算 success，并逐 plan 重算 certificate accept 与 runtime failure；所有 plan 必须
  继续记录 `role_label_visible=false`。全拒绝 query 除 diffusion seed、trajectory SHA 与
  executed trace 外，现在还要求完整 `query_result` 相同。
- artifact seal 不再只哈希“去重后的 digest 集合”，而是哈希排序后的
  `relative_path + digest` 清单；正式规模必须恰好是 48 个 completion 与 288 个原始
  metric/plan artifacts。这样同内容文件、文件缺失或路径替换都不能被集合去重掩盖。
- 与 Table-III、shared-role、Full-Mono 及 controller contract 相关的扩展本地回归为
  `110 passed`；两套 runtime Python 编译、全部相关 sbatch/shell 语法与
  `git diff --check` 通过。Habitat 依赖文件仍由锁定 habitat Python 做 `py_compile`，不向
  memnav 测试环境临时安装 quaternion。

### 00:45 CST result-blind 容量补充链

- 在 `49/125` construction receipts 时，短桶为 `34/40` constructed，中桶前 9 条为
  `2/9`；其余是 covis、受控 Revisit 或 survey-frame navigability 的构造失败。此时仍未
  提交任何 query policy，因此这里只能用于判断预注册人口门是否有容量风险，不能形成
  方法结果或选择 policy 超参。
- 为避免跑完后降低 `16 histories / 10 scenes` 门限，已在读取任何 query outcome 前冻结
  `hm3d_table3_navmesh_capacity_replenishment_protocol_20260831.json`：保持三个距离桶、
  2 m role 距离匹配、60 度 bearing 分离与每 scene 每桶最多 2 条，只把 result-blind
  navmesh sampling 改成独立 seed `20260831`、`512 points/scene`。它不会 render，不加载
  NavDP/CEC，也不能授权 policy evaluation。
- CPU geometry DAG 已提交：smoke `16612322`、100-scene array `16612340`、finalize
  `16612357`、independent verifier `16612387`；immutable bundle receipt SHA 为
  `7d4e0756994404da0da7f7dbbbe8fb402cca1e033fe1c474cfe43528dbb60516`。Smoke 只验证
  geometry runtime，不是结果替代。若原 125 条本身通过，补充容量不会改变冻结人口；若
  不足，只允许按新的 verified geometry ledger 追加去重候选并重新做 16/16/16 独立人口
  gate，仍禁止删除旧 receipts、改阈值或读取 policy outcome。

## 18. 2026-08-31 01:20 CST Table III append-only power closure

### Result-blind navmesh capacity 已完整封存

- `16612322 / 16612340 / 16612357 / 16612387` 均为 `COMPLETED 0:0`；独立 verifier
  返回 `verified=true`，并确认没有读取 query-policy 或 navigation-policy outcome，仍未
  授权 policy evaluation。
- 三个桶的去重 geometry capacity 分别为 `200 histories / 100 scenes`、
  `102 / 54`、`30 / 15`；在原冻结的每桶 `16 histories / 10 scenes` 与每 scene 至多
  2 条规则下，三个桶均有足够候选容量。capacity summary SHA 为
  `9faa519882bd0641e6ded9b6d2042e333dfc2c352007197392d9334974820c4d`，独立
  verifier SHA 为
  `b21b16ddd9acd941748177e5b927441e94f73867f8c8cc6219ed8f63caa977190`。

### 扩样规则先于完整 census 冻结，且只允许追加

- 新的 selection protocol 要求完整读取全部 `125/125` base construction receipts，保留
  每一条 base candidate 及其失败收据；只有最终不足 `16 histories` 或 `10 scenes` 的桶
  才从上述 verified capacity ledger 追加全部未使用候选。顺序固定为 scene-round-robin，
  不删除、不替换、不改阈值，也不读取任何 policy outcome。
- CPU plan freezer / independent verifier 已作为依赖链提交为 `16614442 -> 16614445`，
  run root 为
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e`；
  source-bundle SHA 为
  `a8dd232ef5e64347a7e0ca529d24eedb6cbbe3864a883f02f7260023b85f5eca`。
  两个 CPU job 仍只因 base array 依赖而 pending。

### 当前调度与后续正式链

- base construction 已写出 `61/125` 哈希收据，其中 `40` 条 constructed、`21` 条
  geometry-ineligible；所有收据继续记录 `query_policy_outcomes_read=false`，没有 query
  SR 被生成或读取。余下 array `16610812_[61-124%4]` 的精确 pending reason 是
  `QOSGrpGRES`；本账号没有重复 GPU 作业。按 HPC 手册应等待 `gpu48` 总额度释放，不能
  临时换未验证 QoS 或重复提交。
- 已实现并测试 expansion construction -> merged finalizer -> independent population
  verifier。它动态执行 verified expansion plan 的每一条 candidate，并且最终仍硬要求
  `16/16/16 histories`、每桶至少 `10 scenes`、48 histories / 96 queries；base 的 125 条
  completion ledger 全部进入最终 provenance。
- merged query 链已提前完成但尚未提交：`48` 个同进程 paired elements，每条同时执行
  Novel/Revisit × mono-native/mono-CEC，共 `192` 个 raw arm-role rows。扩样 scene 可能不在
  原 54-scene plan 中，因此 v2 runner 以 sealed merged manifest 自身作为 scene-index
  namespace，并把 analyzer 与独立 verifier 显式绑定到
  `hm3d_table3_causal_survey_population_verification_v2_20260831`。不存在 smoke、partial、
  旧结果替代或降低门限入口。
- 新链补齐 shared-SSH `yz11502` 身份 gate、local/remote immutable bundle self-test、
  exact composed-runtime import smoke 与所有 `afterok` 依赖。全部 Table-III 本地测试目前
  `44 passed`；相关 Python 编译、shell/sbatch 语法检查均通过。

## 19. 2026-08-31 01:55 CST Table III dependency-held formal handoff

- base array 的外部状态未变：`16610812_[61-124%4]` 仍以 `QOSGrpGRES` pending；Slurm
  估计的最早启动时间是 `2026-08-31 04:20 EDT`，该估计不构成资源保证。没有重复提交
  base，也没有切换未验证 QoS。
- 为避免完整 base census 后再等待人工接力，当前有效的 CPU-only deferred launcher 是
  `16614853`，严格依赖 expansion-plan independent verifier `16614445`，同时绑定 base
  population verifier `16610823`。本地提交收据是
  `HM3D_TABLE3_CAUSAL_SURVEY_DEFERRED_V2_SUBMISSION_20260831.json`，SHA 为
  `84ccc8daf2e24a287bf869bffa929ecb8347b8e9dae31b287587ab17c4783c08`；远端 immutable
  bundle SHA 为
  `b03fd773b66de29821c8699c794efd3187fc0edd22c8629186f35cb49472852f`。
- launcher 只能执行两个 fail-closed 阶段。第一阶段要求完整 `125/125` base receipts，
  动态执行 verified append-only expansion 的每一个 candidate，再经 merged finalizer 与
  independent verifier 硬验证三个桶严格 `16/16/16`、每桶至少 10 scenes。第二阶段只有在
  该 verifier `formal_policy_evaluation_authorized=true` 后才能提交 query；它保留最大预算
  的正式 history 作为 full-stack gate，再以 `afterok` 提交其余精确 47 histories，最终仍
  要求 48 completions、96 queries 与 192 raw arm-role rows。gate 本身属于最终总体，不能
  用作 smoke 或 partial 替代。
- expansion plan 合法为空时，表示 base 自身已经通过全部冻结人口门，而不是构造失败。
  launcher 此时不创建伪 expansion，而是在 `16610823` 独立验证原始 base population 后走
  同一 maximum-budget gate 与 48-history query contract；非空时才走 append-only merged
  population。两条路径互斥，均禁止旧总体替代、删 receipt 或降低门限。首版 launcher
  `16614810` 未覆盖这个边界；在它仍为未运行的 `Dependency` 状态时被精确取消，并由
  `16614853` 替换，因而没有留下运行输出或 policy row。
- 第一次远端 composed-runtime preflight 在任何 `sbatch` 写操作前拦住缺失的
  `MemNavData.cec_handoff_contract`。修复没有改变方法或协议：新 bundle 显式包含
  `cec_handoff_contract / controller_portability_contract /
  monocular_depth_runtime / certified_relocalization_runtime`，并绑定此前 Table-I/II 验证过的
  authority runtime closure
  `82e71f19ee7f4e5233fae499633ce5a233c9c036bb41b9e2bf7d4f0f18effd7d`。
  第二次 preflight 强制复核各 module 的 `__file__` 来源与 delayed mono-depth transaction
  API 后通过；相关本地回归为 `58 passed`。
- 截至本节写入时，`16614853` 仍为 `Dependency`；它没有提交任何 policy GPU job，未生成
  或读取 partial SR。该结果若完成，论文中仍只能称为 controlled causal-history-length
  analysis，不能重命名为 actual-NavDP-prefix 结果。

## 20. 2026-08-31 02:02 CST Table III recovery closure and live state

- Slurm 现场先在 `61/125` 后释放三张卡，array 61--63 均以 `0:0` 完成；随后 64--65
  也完成。当前为 `66/125` base completion receipts：`41 constructed / 25
  geometry-ineligible`。余下 `16610812_[66-124%4]` 精确 pending reason 又回到
  `QOSGrpGRES`；
  `16610818 -> 16610823 -> 16614442 -> 16614445 -> 16614853` 均保持
  dependency-held。Slurm 的 start estimate 已发生变化，因此不把它写成完成时限或资源保证。
- 尚无 Table-III query policy job。上面的 construction receipt 只含构造状态；没有 SR、SPL
  或 arm outcome 被用于改变 population、门限、长度桶或后续作业图。
- 对已封存 capacity 的逐 fragment 复核澄清了一个容易误读的数字：summary 中
  `200/102/30` 是按最终每 scene 每桶最多两条计算的 geometry-gate capacity；不可变的
  100-scene fragments 实际保留短/中/长 `800/307/87` 个确定性 triads。Expansion freezer
  会在 deficient bin 中对这些 triads 做 identity 去重后追加全部未使用候选。因此长桶除了
  45 条 base candidates 外，最多还有 87 条当前 seed 的 result-blind rendered-construction
  尝试；不需要改变 covis、距离、方向、scene breadth 或最终 `16/16/16` 门。
- 自动 deferred bundle 已经不可变，不能用本地修改偷换正在排队的正式链。为保证万一需要
  人工恢复时仍执行同一协议，两个 recovery submitter
  `submit_hm3d_table3_causal_survey_queries_hpc.sh` 与
  `submit_hm3d_table3_causal_survey_merged_queries_hpc.sh` 现也显式绑定并哈希验证相同 runtime
  closure；远端 preflight 检查 CEC overlay、LingBot PnP 来源与 delayed mono-depth
  transaction API 的 module provenance。
- base recovery 不再直接并发提交 `0-47`。它和 merged/deferred 路径一样，先选冻结总体中
  computed step budget 最大的 population element 作为完整 full-stack formal gate；只有
  `afterok` 才提交其余精确 47 条。该 gate 保留在最终 48 条总体中，不能作为 smoke 或
  partial result。
- 更新后的 shell/sbatch syntax、`git diff --check` 与同一组契约回归均通过，结果为
  `58 passed`。这些改动只修复恢复路径的运行时闭包和提交顺序，不修改冻结 population、
  controller、threshold、query、budget 或 success definition，也没有触发新 job。
