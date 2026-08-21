# Shared-online double-Revisit pilot — 2026-08-13

## 结论

真实 online-A 记忆可以在同一 episode 中连续支持两个非精确 Revisit 目标。固定的 4 场景 V1 pilot 中，B 成功 4/4、C 成功 4/4、joint 4/4。

这是管线与机制 pilot（N=4），不是成功率估计，也不作显著性或论文主结果声明。

## 正式 pilot 契约

- A 是原生 NavDP 实际完成的 online rollout，不是 expert replay；共 1059 个物理帧。
- 每次评测逐帧哈希重放同一条 A；NavDP 只恢复原始 decision frames，重放期间 diffusion sample 为 0。
- B/C 均来自 online-A 历史位置附近的受控新视图：平移 0.20–0.50 m、偏航 10–25°，不是历史 JPEG 的像素复制。
- B 自然执行，保持 A 末端的 NavDP FIFO；仅在 B→C 切换时清空 NavDP 的 8-frame 短期 FIFO（`before_c`）。
- MemNav/LingBot 的因果状态不清空；C 的目标检索候选 ceiling 固定在 online-A 最后一帧，禁止把 B rollout 当作 Goal-C anchor。当前位姿编码仍合法地跟踪 B 期间的机器人运动，这不是 Goal-C 目标证据。
- 清 FIFO 后，C 的首个当前观测必须对 Goal-C 共视不超过 0.10；之后 C 自己产生的在线观测可正常进入控制器。
- seed 固定，轨迹选择为 server，所有 oracle override 关闭。

## V1 闭环结果

| scene | A frames / decisions | B steps | B final | B-tail max / endpoint C covis | C steps | C final | joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| gxdoqLR6rwA | 240 / 30 | 184 | 0.988 m | 0.0106 / 0.0000 | 73 | 0.987 m | 1 |
| pLe4wQe7qrG | 405 / 51 | 120 | 0.998 m | 1.0000 / 0.0000 | 47 | 0.981 m | 1 |
| yqstnuAEVhm | 213 / 27 | 159 | 0.974 m | 0.0000 / 0.0000 | 92 | 0.978 m | 1 |
| mJXqzFtmKg4 | 201 / 26 | 157 | 0.998 m | 0.0000 / 0.0000 | 57 | 0.982 m | 1 |

总账：B 4/4；有效 C 输入契约 4/4；C given B and valid input 4/4；joint 4/4。四条 C 的 reset receipt 均为 `algo=navdp`；所有 C 计划的候选 ceiling 均不超过各自 A boundary。实际 C anchors 分别为 92、190、142、127，均来自 A；四场 `graph_subgoal_enabled` 均为 false，B 帧没有通过图路径成为间接子目标。

## 为什么需要 V2 和 `before_c`

原 V1 只约束 B/C 的时间与空间距离。gxdo 的旧样本中，实际 B rollout 对 C 的最大共视为 0.4885，因此 C 会获得新的近程记忆捷径；该数据不能用于双 Revisit 声明。

V2 增加了 A-end→B shortest-path 的视觉 route-negative 约束，4 场景、V0/V1 共 8 对均通过离线上限 0.08。然而 pLe 的真实策略 B 路径仍达到 C 共视 1.0：同一位置的控制器朝向不同于 path-facing proxy。全 360° 排除又会过度保守，因为“某个朝向理论可见”不等于控制器实际观察到。

因此最终采用最小因果隔离：B 按自然状态执行；C 前只清 NavDP 短 FIFO，同时把 Goal-C 的目标检索硬锁在 A 边界。pLe 中 carry 与 `before_c` 的 B plans 和逐帧 rollout 哈希完全一致；B 终点对 C 共视为 0，reset 后 C 使用 A 中的 anchor 190，在 47 帧内成功。相反，`every_goal` 在 B 前也清 FIFO，会使 pLe 的 B stuck（最终 2.086 m），说明它改变了 B 控制条件，不适合作为主协议。

## 可复现资产

- online-A：`.diagnostics/shared_online_a_v0v1_pilot_native_20260812`
- V2 benchmark：`.diagnostics/shared_online_double_revisit_v2_route_negative_pilot_20260812`
- benchmark manifest SHA-256：`95f5cbb311c10f3f6604eca47632cefea4b77b80d9f1e0e6ec93c1056c30786f`
- 4 场景结果：`.diagnostics/shared_online_double_revisit_closed_loop_pilot_v2_20260812`
- 独立 benchmark 审计：`.diagnostics/shared_online_double_revisit_v2_route_negative_pilot_20260812/audit.json`

## 当前能说与不能说

能说：评测构造已经从 expert-frame 假 Revisit 修正为真实 online history；在 4 个场景中，冻结的目标记忆定位与 NavDP controller 能连续兑现两个相邻视角 Revisit，且第二次目标检索未使用第一次 Revisit 新采集的目标证据。

不能说：4/4 不能证明总体 SR 接近 100%；目标对经过可行性筛选，而且当前使用 known-Revisit phase route 与 raw-DINO direct/legacy metric 路径。下一步统计实验必须冻结本页契约，加入 native/cold 或 single-memory 对照，并在不重用这 4 个 pilot 场景的池上配对运行。

## 为什么 2-leg 的 expert/online 差异没有破坏结果，而旧 3-leg 会

这不是“2-leg 天然稳、3-leg 天然不稳”，而是两个协议包含的信息条件不同。

正式 2-leg certificate 实验虽然最初按 expert-A 定义 Goal B，但后续已经在实验自己的真实
online-A trace 上逐帧重渲染审计：A 成功的 `120/120` 条都有
`max covis >= 0.20` 的 Goal-B 支持，`115/120` 还有 `max covis >= 0.50` 的强支持；全部
11 条低于 0.20 的 episode 都是 A 失败，未进入 `B|A` 分母。因此该 2-leg 数据中，A 成功
事实上也筛出了 online-A 确实观察过唯一 Revisit 的 episode；`112/120` 的 certificate 结果不受
expert/online 标签错配驱动。

旧 double-Revisit 则要求同一条 online-A 同时覆盖两个彼此分离的历史地点。到达 Goal A 并不
保证策略沿 expert-A 的同一路径走过，更不保证 expert 路径上的两个 anchor 都被真实相机观察。
旧 N=2 pilot 中，Goal B 在 online-A 上有 `301/223` 个 PnP inliers，而 Goal C 的最近帧只有
`8/11` 个 Fundamental inliers：同一条成功 A 可以充分支持 B，却几乎不支持 C。

3-leg 还多了一种 2-leg 不存在的混杂：中间 B rollout 可能重新看到 C。旧 pilot 的两条表面
C 成功分别选择 frame `174` 和 `95`，但各自 online-A boundary 只有 `150` 和 `76`；把 C
候选收紧到 A-only 后，表面 `2/2` 变成 `0/2`。所以旧数字混合了长期 A-memory、expert/online
可观测性差异和最近 B-memory 捷径，不能解释为长期记忆 SR。

这里的证据强度也必须分开：2-leg 的可观测性结论有 120 条有效 episode；旧 3-leg 的
`2/2 -> 0/2` 只有 N=2，不能估计总体降幅，但已经足以否定旧 benchmark 的因果解释。当前
shared-online 构造正是针对这两个缺口：先冻结真实 online-A，再从中确定 B/C，并把 C 的目标
证据严格锁在 A boundary。

## 冻结的下一步：先补反事实，再决定是否上 HPC

当前 `4/4` 只完成了 `known-Revisit raw-DINO direct + legacy metric residual` 一条方法臂。
它证明执行链可行，但尚未证明目标必须依靠长期记忆，也没有验证 role-free certificate 能否
连续接管两次。下一步先在现有四条 V1 episode 上补齐三个本机臂；所有臂必须复用相同的哈希
online-A、相同 seed、`before_c` 和 A-only C-history：

1. **native/native**：B、C 都由冻结 ImageGoal NavDP 控制，判断当前目标对是否本身就很容易；
2. **memory/native-C**：B 与现有 full-memory 臂使用完全相同的 known-role memory controller，
   到 C 才切回 native。它应与 full-memory 臂拥有逐项相同的 A、B 路径，是检验“第二次旧
   记忆是否真正有因果价值”的主反事实；
3. **role-free certified**：B、C 都运行冻结的 certificate；pass 才输出 scale-free bearing，
   reject 精确回退 native。失败只按 `certificate reject`、`accepted but controller fail` 或
   runtime failure 归因，不在这四条 pilot 上调阈值或半径。

本机 gate 只决定 benchmark 是否有信息，不报告显著性：

- shared-A hash、A boundary、C-tail 和 reset receipt 必须全部通过；
- full-memory 与 memory/native-C 的 B rollout 必须逐项一致；
- 如果 C 在 full-memory 与 memory/native-C 间没有任何 paired discordance，说明当前 V1
  太容易或没有 memory-specific signal，不能原样放大；
- 如果 certified 的主要问题是 fail-closed reject，说明缺 certificate recall；如果 certificate
  已 accept 但闭环失败，才归到 residual/controller；两者不能混为一个 SR；
- 只有出现可恢复的 memory gap 且 protocol 无违规，才冻结正式 manifest。

通过本机 gate 后，正式统计实验应使用未参与这 4 条 pilot、旧 fresh160 和训练/开发决策的
scene clusters。名义目标是每个 cluster 两条 shared-online V1 配对，但 scene 数必须按独立
scene inventory 冻结，不能为了凑 40 条把 episode 数伪装成 scene 数。预先固定目标选择规则，
不按任何方法成败重采样。正式报告：

- `SR_B`、`SR_C|B`、`joint(B and C)`，不再把一个 joint 单值当全部解释；
- full-memory 对 memory/native-C 的 paired C `+/-`：这是多地点持久记忆的主因果检验；
- certified 对 native 的 paired joint `+/-`：这是 role-free 部署系统的主方法检验；
- exact McNemar 与 scene-cluster bootstrap CI；C 输入合同失败单列，不混入策略失败。

标准 strict-v4 `Novel A -> Novel B -> Revisit C` 暂不作为主要 SR 扩样，因为历史结果中
`B|A` 约 20%，大多数 episode 到不了 C。它保留为互补的 **open-set safety stream**：在已有
A-memory、但 Goal B 为 Novel 时测 certificate false takeover；只有 `A and B` 成功的 episode
才评 C。这样 double-Revisit 回答“能否连续使用两个旧地点”，strict-v4 回答“有历史但目标
Novel 时能否安全 abstain”，两者共同验证同一个 certified residual 架构，而不再让 Novel
探索失败淹没 Revisit 结论。

## 2026-08-13 四臂本机因果 gate：已完成

上述本机 gate 已在同一 RTX 4090、同一 MemNav/NavDP server pair 内完成。四场按平衡顺序运行，
所有 arm 都逐帧哈希重放相同 online-A；生成器、checkpoint、V1 manifest、A-only ceiling、
`before_c` reset、确定性 seed 和 oracle-off 设置保持冻结。独立审计未导入 evaluator 的汇总代码，
结果为 `audit_ok=true`。

| arm | B | C given B and valid input | joint | mean B/C steps |
|---|---:|---:|---:|---:|
| full memory：B/C 均 known-Revisit residual | 4/4 | **4/4** | **4/4** | 158.25 / 67.0 |
| role-free certified residual | 4/4 | **4/4** | **4/4** | 153.25 / 77.5 |
| memory-B / native-C | 4/4 | **0/4** | 0/4 | 158.25 / 452.0 |
| native/native | 0/4 | 无有效 C 分母 | 0/4 | 581.0 / — |

最重要的因果对照是 full-memory 与 memory-B/native-C。四个 episode 中，两臂的 B plans、逐帧
物理 rollout 和写入的 memory trace 全部逐项完全一致；差异只从 C controller 开始。full-memory
把 C 从 `0/4` 提高到 `4/4`，paired `+4/-0`，exact McNemar `p=0.125`。因此当前 V1 确实含有
第二次持久记忆的因果信号，不是 Goal C 本身很容易，也不是 B 轨迹差异制造的结果。N=4 仍然
不能作为 SR 或显著性结论。

role-free certificate 在四条 episode 的两次 Revisit 上均完成闭环：B 共 `78/78`、C 共
`40/40` 个 planning decisions 通过 certificate，118 次原因全部为 `certificate_accepted`；没有
reject、accepted-but-fail 或 runtime failure，C 的候选 ceiling 从未越过 online-A boundary。
它与 known-role full-memory 的 joint 为 `4/4 vs 4/4`。这证明冻结的 scale-free bearing 接口可
连续接管两个目标，但也说明这四条经过可观测性筛选的正样本无法测 certificate 的 abstention
边界；该安全问题必须由独立 Novel no-match stream 回答。

全 native 在四条的第一次 Revisit B 就全部失败，所以不能用它构成 C 的配对分母。它仍有两个
作用：证明任务不是冻结 ImageGoal NavDP 自己即可完成，并提供 certified/full-memory 的端到端
joint 负对照。certified 对 native joint 为 `+4/-0, p=0.125`，同样只属机制 gate。

### Gate 后冻结决策

本机 gate **通过**，理由不是“4/4 看起来高”，而是预注册的主反事实出现四个方向一致且零反向
的 C discordance，同时全部因果合同通过。下一步可以扩样，但不能在这四条上继续修改目标扰动、
certificate threshold、2.5 m adapter 或 controller。

正式扩样分成两个互补数据流：

1. **Primary：fresh shared-online double-Revisit**。先在未参与本 pilot 和旧架构决策的场景上
   真实执行并冻结 online-A，再按冻结 V1 规则预注册两个目标。每 scene 目标为两条 episode，
   但最终 scene 数服从下面的独立集合审计，不能预写成 20；四臂仍为本节四臂。主检验是 full-memory C 对
   memory-B/native-C 的 paired `+/-`，部署检验是 certified 对 native 的 paired joint；按 scene
   cluster bootstrap，并单列无合法 B/C pair 的数据构造失败，绝不按方法结果补采样。
2. **Safety：strict-v4 Novel no-match stream**。共享已有 A memory 后，以 Novel B 为主要单位，
   比较 native 与 certified 的 false takeover、B paired SR 和损失；不要求 B 成功才进入
   false-takeover 分母。C 只在 `A and B` 成功时作为次要 Revisit consistency 指标。该流不与
   double-Revisit 的正样本 SR 合并。

只有 Primary 获得足够的有效 C 配对分母，且 Safety 没有不可接受的 Novel harm，才进入更大
HPC 确认；不会直接恢复旧 expert-relative 3-leg，也不会把两个数据流压成一个 joint 数字。

### Scene inventory 停止线

本机 `/home/asus/Research/datasets/mp3d_20scene/assets` 只有 20 个 scene assets，它们与 fresh160
使用的 20 个 scene clusters 相同，且已经参与大量架构决策；因此本机不能产生真正 fresh 的
scene-disjoint 正式结果。canonical 总账同时注明：所谓 526 条 3-leg pool 是 episode 数，真正
同时与旧 20-scene pool 和 train40 不重叠的统计单位只有约 16 个 scene clusters，不能写成
`N=526`，也不能无依据承诺另有 20 个 fresh clusters。

因此下一次 HPC 动作必须先只读生成 **scene-role receipt**：列出 cluster 数、与 pilot/consumed/
train/development 的交集计数，但不读取任何目标或 outcome。若唯一剩余集合属于 final-reserved
或 blind，则在明确授权一次性打开之前停止；不能把 held-out 当成新的 development pool。正式
任务一旦打开 held-out，architecture、certificate、adapter、目标构造和统计脚本都必须保持本页
哈希冻结，只允许修复任务基础设施，不允许根据结果调方法。

### 新增实现与收据

- 逐腿因果 route：`multigoal_policy_contract.py` 的显式
  `known_revisit_leg_indices`，默认行为不变；
- evaluator CLI：`--shared_online_known_revisit_scope both|b_only`；
- 同机四臂 runner：`run_shared_online_double_revisit_gate_local.sh`；
- 独立审计：`audit_shared_online_double_revisit_gate.py`；
- run root：`.diagnostics/shared_online_double_revisit_gate_local_20260813`；
- audit SHA-256：
  `1ad9ccddce69e52dd72f41bd165c93088b93cb09a761e5df199f461536a5586a`；
- source-input receipt SHA-256：
  `0c288babbba7766ea1965ee4c7869499f4503c49e636861fb29ee2d8864239ce`；
- 24 项逐腿路由、shared-online replay 与 goal-switch 回归测试通过；runner 结束后 server 已关闭，
  GPU 回到约 255 MiB / 0% idle。
