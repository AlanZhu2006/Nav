# Double-Revisit NRR pilot（2026-08-12）

## 问题与协议

本 pilot 检验的是：同一条在线因果记忆能否连续支持两个不同的 Revisit，而不是把旧
Novel→Novel→Revisit 数据改标签。

协议 `multileg_v5_double_revisit_20260812`：

- `A = initial ImageGoal`，`B = Revisit`，`C = Revisit`；
- B、C 都采样自 leg A，但两者的 stride-1 GT anchor 至少相隔 32 帧；
- C 的完整 covisibility curve 覆盖 leg A+leg B，且整个 expert leg B 必须满足
  `max covis <= 0.10`，避免生成数据中的最近历史捷径；
- Goal B/C JPEG 都必须与 metadata pose/yaw 的 Habitat 重渲染逐字节一致；
- 首段 yaw 均匀采样；不足、非有限值、错误锚点、错误帧索引或不满足 hard negative
  均 fail closed。

评测三臂：

1. native：A/B/C 全部冻结 NavDP；
2. known-role direct：A 用 NavDP，B/C 都用 raw-DINO top-1 + LingBot metric pose +
   mixed NavDP residual；这是有角色标签的 reference，不是可部署 selector；
3. certified：A/B/C 都自行做 open-set SuperPoint+LightGlue/PnP certificate，accept 才把
   scale-free bearing 交给 mixed NavDP，否则精确 fallback native。

## 四条固定数据及审计

场景与 seed 在生成/策略运行前固定，没有按策略成败重采样：

| scene | seed | geo A/B/C (m) | 初始 heading offset | B/C GT anchor | C 对 expert-B 最大 covis |
|---|---:|---:|---:|---:|---:|
| e9zR4mvMWw7 | 20260840 | 5.043 / 3.625 / 3.307 | +123.08° | 45 / 89 | 0.0000 |
| gxdoqLR6rwA | 20260841 | 8.297 / 5.250 / 3.501 | -150.63° | 112 / 56 | 0.0829 |
| dhjEzFoUFzH | 20260842 | 8.523 / 4.940 / 2.778 | -149.36° | 120 / 39 | 0.0782 |
| gTV8FGcVJC9 | 20260843 | 3.581 / 3.437 / 3.180 | -1.50° | 39 / 77 | 0.0000 |

四条均由独立 Habitat simulator/pathfinder 重算并通过：`4/4 valid`；首帧和 A 终点位姿、
B/C 重渲染 JPEG、Parquet/RGB/depth 连续索引、anchor 归属、anchor gap、expert-B hard
negative 全部通过。前三条在一次 outer attempt 内接受；gTV 使用预设预算的第 5/10 次
outer attempt 接受，没有放宽条件。

## 冻结闭环结果

公共设置：RTX 4090；官方冻结 `navdp_checkpoint.ckpt`；每 leg 最多 600 steps；
`exec_horizon=8`；确定性 diffusion seed；NavDP short memory carry；position SR 1 m。

四条 native 的 A 为 `2/4`。gxdo、dhj 在 A 失败，B/C 因果 censor，不能计为 Revisit
失败或成功。有效分母是 e9z 与 gTV：

| scene / arm | A | B given A | C given AB | B steps/path | C steps/path | joint SPL |
|---|---:|---:|---:|---:|---:|---:|
| e9z native | 1 | 0 | censored | 600 / 21.350 m | — | 0 |
| e9z known direct | 1 | 1 | 1 | 99 / 3.454 m | 221 / 5.667 m | 0.813 |
| e9z certified | 1 | 1 | 1 | 161 / 4.861 m | 471 / 17.228 m | 0.432 |
| gTV native | 1 | 0 | censored | 225 / 7.227 m | — | 0 |
| gTV known direct | 1 | 1 | 1 | 118 / 3.897 m | 203 / 6.407 m | 0.773 |
| gTV certified | 1 | 0 | censored | 234 / 7.095 m | — | 0 |

因此，在仅有的两条 A-success episode 上：

- native B：`0/2`；
- known-role B：`2/2`，且随后表面上的 C：`2/2`；
- certified B：`1/2`；e9z joint 成功，gTV certificate 持续接管但 B 仍失败。

这些是 `N=2` integration/mechanism evidence，不能报告为正式 SR、增益或显著性。

## 最重要的因果审计：表面双 Revisit 含在线 B 轨迹捷径

生成器保证的是 expert leg B 对 C hard-negative；但在线策略执行的 B 轨迹与 expert leg B
不同。逐帧读取真实 memory trace 后发现，known-direct 的 C 在两条成功 episode 中都选择了
在线 B 段的新帧，而不是 A 段：

- e9z：在线 A 结束于 frame 150；C 选择 frame 174；
- gTV：在线 A 结束于 frame 76；C 选择 frame 95。

这说明 `A→B→C` 的任务级成功是真实的，但不能据此主张“系统保留并调用了 A 中两个不同
旧地点”。为隔离该问题，新增 fail-closed `--double_revisit_c_history initial_leg_only`：它只把
C 的 retrieval candidate ceiling 收紧到在线 A 的最后一帧，绝不扩大默认因果候选范围；B 的
seed、steps、路径和结果保持逐项相同。

| scene | C 全历史 | C anchor | C 仅 A 历史 | C anchor | 受限后终距 |
|---|---:|---:|---:|---:|---:|
| e9z known direct | success, 221 steps | B 段 174 | fail, 600 steps | A 段 93 | 4.105 m |
| gTV known direct | success, 203 steps | B 段 95 | fail, 600 steps | A 段 42 | 1.537 m |

严格旧记忆口径因此是 `C|AB = 0/2`，不是 `2/2`。这是本 pilot 的主要发现。

e9z certified 的 C 无论全历史或仅 A 历史都未接管：全历史候选 183 因 fundamental inliers
不足拒绝；仅 A 历史候选 93 因 reference hull coverage 拒绝。两臂最终都由相同 native fallback
在 471 steps 成功，因此不能归因给第二次 certificate。

## C-only 因果归因（同日补充）

为区分“raw-DINO 排错 anchor”与“在线 A 中根本没有可执行 C 记忆”，新增仅限严格
double-Revisit 的评测 oracle：A、B 完全照旧；C 强制使用**实际在线 A 轨迹中离 Goal-C
最近的 causal frame**。oracle 使用 Habitat 目标坐标，只是机制上限，不是方法结果。

两条样本的新旧 A/B plans、memory trace、steps、path length 和终距逐项完全相同；唯一变化
是 C anchor：

| scene | raw-DINO A-only anchor / 距 C | raw C | path-nearest anchor / 距 C | oracle C |
|---|---:|---:|---:|---:|
| e9z | 93 / 2.449 m | fail，600 steps，4.105 m | 150 / 1.660 m | fail，235 steps，1.207 m |
| gTV | 42 / 1.776 m | fail，600 steps，1.537 m | 76 / 1.113 m | **success**，285 steps，0.996 m |

严格 `C|AB` 因而从 raw-DINO 的 `0/2` 变为 privileged path-nearest 的 `1/2`。N=2 不能报
SR 或显著性，但机制信息明确：

- 两条 C 都从较好的 anchor 获益，candidate selection 有真实因果作用；
- gTV 证明在线 A 中确实有能完成 C 的旧记忆；
- e9z 改善很大但仍差 0.207 m，并因 stuck 提前终止，说明 ranking 不是唯一瓶颈，后面还有
  视角覆盖、LingBot pose 或 mixed controller 的误差。

### certified A-only 全候选审计

使用与在线 certificate 完全相同的 SuperPoint+LightGlue、1.5 px Fundamental-MAGSAC 和冻结
precheck，扫描所有合法 A 帧，而不是只看 DINO top-8：

| scene | 合法 A 帧 | 全 A precheck-pass | DINO top-8 pass | 最近帧证据 |
|---|---:|---:|---:|---:|
| e9z | 143 | 2（frame 129、147） | 0/8 | frame 150：8 fundamental inliers |
| gTV | 69 | 0 | 未形成 C shortlist | frame 76：11 fundamental inliers |

e9z 的两个可进入 PnP 的帧距 C 分别为 1.834 m、1.672 m，但都不在 DINO top-8。精确重放
在线 A 后，服务器再次给出完全相同的 top-8 `[8,14,93,100,74,18,106,87]`。把 129/147
直接交给完整 LingBot-depth PnP 仍未通过正式 certificate：

- frame 129：Fundamental 28 inliers；PnP `status=ok`，但仅 10 inliers，query/reference
  coverage `0.033/0.002`，拒绝；
- frame 147：Fundamental 19 inliers；PnP 6 inliers，reference coverage `0.013`，拒绝。

gTV 更能说明 hard certificate 的 recall 问题：没有任何 A 帧达到 16-inlier precheck，nearest
frame 76 只有 11 inliers，却已经在真实闭环中把 C 从失败救成成功。因此当前 certified 路线
不是单一的排序问题，而是两层召回损失：DINO shortlist 会漏掉可疑似正确帧，atomic PnP
certificate 又会拒绝一部分实际可用的方向证据。

作为对照，同一系统在第一次 Revisit-B 上并未整体失灵：e9z/gTV 的 certificate 分别选择
frame 80/16，PnP inliers 为 301/223，且每次计划都 accepted。B 与 C 的巨大证据差说明，
问题集中在第二个目标相对**实际在线 A 轨迹**的可观测性，而不是 LightGlue 或服务器普遍坏掉。

### 新发现的 benchmark 定义缺口

生成器把 B/C 定义为 generator expert leg-A 的 Revisit；评测时记忆来自冻结 NavDP 实际跑出的
online leg A。两条路径并不相同。因此“C 在 expert A 上有 GT anchor”并不保证“C 在在线 A
记忆里有强共视 anchor”。本 pilot 中 B 恰好有数百个 PnP inliers，而 C 只有 0--2 个能过
Fundamental precheck，正是这个缺口的直接证据。

所以当前 `0/2` 不能简单归因为模型不会保留两个地点；它混合了：

1. expert-relative goal sampling 与 online-memory observability 不一致；
2. DINO shortlist recall；
3. hard certificate recall；
4. pose/controller 的剩余误差。

## 结论与下一步

当前证据支持：

1. 双 Revisit benchmark、逐腿路由和连续记忆执行链已经完整跑通；
2. known-role Revisit expert 在两条 native-B 失败上都救活第一次 Revisit；
3. 默认在线记忆确实能利用中间轨迹继续完成任务，但会把“长期保留两个旧地点”与“最近路径
   再定位”混在一起；
4. 在真正只允许 A 段旧记忆的严格定义下，raw-DINO 配置为 `0/2`、privileged
   path-nearest 为 `1/2`，不支持立即宣称双旧地点记忆成功；
5. certified 方法对第一次 Revisit 只有 `1/2` 闭环成功，且第二次未通过 certificate，尚未稳定。

不应把当前 generator-relative 配置直接扩到 HPC；上面的归因已经说明，扩大只会把数据定义
缺口、retrieval 和 controller 混成一个更大的失败率。

下一步应先构建 **shared-online-A double-Revisit** 小基准：先冻结并保存一条 native A rollout，
再从这条真实 rollout 的 RGB/pose 中预注册两个时距分离的目标，所有 arms 精确 replay 同一个 A
前缀。B/C 都必须在 online A 内满足预声明的空间与视觉支持区间，C 仍只允许 A history。这样
才能把问题干净地变成“同一在线记忆能否检索并执行两个旧地点”，而不是测试 expert/online
路径是否碰巧重合。

本机先生成 2--4 条并做 native / raw-direct / path-nearest 三臂门检：只有 raw 与 oracle 之间存在
稳定可恢复 gap，且两个目标的 online observability 合同都通过，才冻结 20-scene manifest 上 HPC。

## 代码与证据路径

- 数据生成：`MemNavData/generate_twoleg.py`
- 数据契约：`MemNavData/multigoal_benchmark_contract.py`
- 独立审计：`MemNavData/audit_multigoal_role_symmetry.py`
- 逐腿路由：`MemNavData/multigoal_policy_contract.py`
- 3-leg evaluator：`MemNavData/eval_3leg_habitat.py`
- C-only oracle helper：`MemNavData/double_revisit_diagnostics.py`
- 全 A visual-support 审计：`MemNavData/audit_double_revisit_visual_support.py`
- causal candidate ceiling：`NavDP/baselines/memnav/policy_agent.py`、
  `NavDP/baselines/memnav/memnav_server.py`
- 全部生成/审计/闭环 receipt：`.diagnostics/double_revisit_nrr_pilot_20260812/`

相关 compile、diff whitespace 检查以及 30 项定向回归测试通过。
