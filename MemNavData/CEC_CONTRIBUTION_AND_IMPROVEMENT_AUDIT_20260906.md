# CEC 成果、贡献与改进方向复审（2026-09-06）

本轮问题：所有已有成果能支撑什么？“工程拼接感”是否真实？下一步怎样提升而不重复失败路线？

后续进展：[Pi3X 时序上下文归因与本机小重放](PI3X_TEMPORAL_CONTEXT_ATTRIBUTION_20260906.md)。固定当前图后，仅更换因果桥接帧即可在一个已消费案例复现/消除方向翻转；分析指向 current-to-anchor 更新，而非已证明的目标关系漂移。未形成新 SR 或生产修复。

进一步验证：[连续当前状态与稀疏目标关系](PI3X_PERSISTENT_GOAL_READOUT_RESULT_20260906.md)。同两条已消费 RGB 序列上，连续状态 + 初始固定目标关系将原 accepted 时刻的中位方位误差从 16.28° 降到 2.19°，大于 90° 的次数从 12 降到 0；动态目标对照已实现大部分改善。这支持时序职责分离的研究方向，但不增加正式 SR，不改变 Pi3X 原未通过决定。

补全进展：[四 query / 三 scene 固定观测结果](PI3X_SHARED_STATE_FOUR_QUERY_RESULT_20260906.md)。在全部四条已知异常的 143 个原 accepted 时刻，连续状态 + 动态目标关系将中位误差 12.57°→2.22°、>90° 次数 20→0。固定首次关系不一致优于动态关系；尤其弱支持 Novel 的动态 P95 为 4.04°、固定为 11.20°。因此主发现是连续当前状态与目标定位的分工，不是永远缓存第一次目标，更不是新的 SR 提升。

方式：使用 `icra-human-review` 做作者侧、同上下文复审；读取活动论文、实验报告、关键实现与封存结果。本轮独立复算了本地 fresh full-mono HM3D 的 168 条 arm-query CSV 记录，以及远端 Final14 的 84 个 CEC/Pi3X 计划文件。其他主表数字核对活动稿及已有独立审计，不声称本轮重新运行或重算所有历史实验。

没有改论文、生产代码或冻结结果，没有运行训练、导航评测、提交/取消 Slurm 作业、操作机器人或 commit/push。只新增本报告。远端通过既有 `alantorch` / `yz11502` 共享 SSH 做只读计算，审计子会话已退出，未关闭共享 master。

## 1. 核心判断

方法有可信的跨目标记忆收益；training-free 不是问题所在。但目前最强证据支持的是“因果视觉历史 + 流式几何能够帮助冻结策略”，而不是“每一个 matcher / certificate 组件都带来独立、显著的 SR 增益”。

当前合理贡献与 Abstract/Introduction 基本一致：把一个冻结流式几何模型与同源 RGB 索引组成可查询的 episodic memory，稠密读取当前深度，稀疏读取历史目标相对方向，接入冻结局部策略。不应为了创新性把它重新称为 learned MoE、新定位算法或通用长程规划器。

本轮新增且最重要的发现：**此前 Pi3X 的“69/479 个错误重定位方向”解释混用了最短路切向与目标直线方位。重新按目标方位计量后仍有 20/479 个 >90°，但不能再把全部 69 个称为定位错误。** 见第 4 节。这个问题没有推翻 CEC 的 SR，也没有让 Pi3X 自动通过原来的替换门。

## 2. 成果总账：主结果与机制结果分开

### 2.1 当前最有力的闭环结果

| 结果 | 配对结果 | 可以支持的结论 |
|---|---|---|
| 完整单目 HM3D，28 histories / 21 scenes | native 17/56，raw 28/56，CEC 32/56；CEC 对 native +16/-1，p=0.000274658 | actual-mono A 历史与单目 query 链可以组合；不是只有 query-time mono |
| HM3D / NavDP，28 histories | Revisit 8/28→25/28，+18/-1，p=0.000076294；Novel 6/28 不变 | 同一冻结 controller 上跨目标记忆的效用 |
| HM3D / ViNT，同一 28-history population | Revisit 3/28→19/28，+16/-0，p=0.000030518；Novel 3/28 不变 | 经 controller-specific adapter，记忆证据可迁移 |
| MP3D / NavDP，42 histories / 25 scenes | Revisit 9/42→37/42，+29/-1，p=5.77e-8；Novel 17/42 不变 | 另一数据集的配对复现 |
| MP3D / ViNT，同一 42-history population | Revisit 2/42→24/42，+22/-0，p=4.77e-7；Novel 9/42 不变 | 第二 controller 的跨数据集复现 |
| 两个成功目标后的第三目标，20 histories / 13 scenes | Revisit-C 8/20→17/20，+10/-1，p=0.01171875；Novel-C 4/20 不变 | 累计历史在后续目标仍有用；不是无条件三段 joint |

Table I 的 HM3D 与 fresh full-mono HM3D 是不同 query populations，不能互换 native 分母或成功数。两个 controller 的输入适配也不同：NavDP 保留原始目标图并接收固定半径 PointGoal；ViNT 使用受限转向与认证历史 anchor 图。表中是各自的 memory-on/off 效应，不是完整 NavDP 与完整 ViNT 导航系统的绝对排名。

MP3D 有同一条 weak-support Novel 被两个 adapter 接受，但 binary outcome 未改变；所以“Novel SR 不变”不等于“从不接管 Novel”。

### 2.2 归因与旧正结果

| 证据 | 结果 | 边界 |
|---|---|---|
| Final14 query-depth 五臂 | metric native 11/42，zero native 4/42，mono native 10/42，metric CEC 26/42，mono CEC 28/42 | 共享 metric-A 历史；不能替代完整 mono-A 结果 |
| 单目 first-goal 对照 | mono 27/40，metric 30/40 | 未达到既定非劣效标准；不证明 mono 优于 RGB-D |
| 当前 mono Final14 authority | raw / finite-PnP / CEC = 23/42、25/42、28/42；Revisit 都为 20/21 | strict 对 finite +4/-1，p=.375；对 raw +5/-0，p=.0625 |
| 早期 metric-controller Final14 | CEC 28/42 vs raw 21/42，+8/-1，p=.0391 | 这个正结果应保留；不能用它替代不同深度条件下的 mono 对照 |
| Fresh160 supported Revisit | native 27/120，raw 106/120，CEC 112/120；CEC 对 raw +9/-3，p=.146 | 强支持、高共视，接近饱和 |
| 最初 geometry memory | 4/40→19/40，+15/-0，p≈6.1e-5 | 早期“历史有用”证据 |
| Novel oracle bearing | 28/40→40/40，+12/-0，p=.000488 | 特权局部路线信息上界，不是可部署 Novel 方法 |
| 18 条连续多目标机制 | forced reject / initial-only / all-prior = 4/18、6/18、11/18 | 两端 +8/-1、p=.0391；all-prior 对 initial-only 未单独显著 |

不能将同一 Final14 population 的不同读出、深度、support-band 分析算成多组独立泛化证据。

### 2.3 不应再作为“新建议”重启的路线

- 加宽候选：18/40 vs 18/40；只能说该处理未改善，不证明所有检索都无价值。
- CDEC：top-1 126→128/155，却把 actionable accepts 从 122 降到 115。优化候选标签不等于优化最终几何可执行性。
- Candidate-free GCT：DINO-addressed 18/20 vs full-prefix 5/20，暴露长历史寻址困难；不是学习不可能的证明。
- 小 residual：74/80→76/80，+2/-0，p=.5，未提供稳定升级证据。
- Active glance：最好 25/40，仍低于 native 31/40；不能重新包装原地扫描为已验证改进。
- X-NavDP：21/26 vs 20/26，p=1；这次未证实额外收益，不是普遍控制能力等价。
- Graph rescue：若干内部个案收益，NNR 正式对照没有净增益，不能重新当通用方法。
- Metric distance：固定 bearing 25/28，metric 24/28，+1/-2，p=1；不是证实 metric 更差，只是没有显示额外效用。
- GOAT/Replica：前者存在不同目标、执行和到达合约，后者受当前构造规则限制；都不是已完成的外部方法成功验证。
- Pi3X：存在真实 SR 正结果；其失败解释须按第 4 节修正，不能简单归入“学不出来”。

## 3. 独立复算：短中程还有什么可提升

原始目录：

`.diagnostics/hm3d_fresh_fullmono_mixed_role_20260820/pulled_20260828/evaluation_natural_direction/`

本轮读 84 个 CSV，168 条 arm-query 记录，28 histories、21 scenes；每条 `reached` 与记录末距离 `<1 m` 一致。

| Role | native | raw fixed | CEC | CEC 对 raw |
|---|---:|---:|---:|---|
| Novel | 8/28 | 4/28 | 8/28 | +5/-1，p=.21875 |
| Revisit | 9/28 | 24/28 | 24/28 | +1/-1，p=1 |
| 合计 | 17/56 | 28/56 | 32/56 | +6/-2，p=.2890625 |

CEC 的四条 Revisit 失败全部已有持续 certificate accept，runtime failures 均为 0，全部以 stuck 结束：

| scene / episode | 初始 geodesic | 末距 | accept plans | raw 是否成功 |
|---|---:|---:|---:|---:|
| 7GAhQPFzMot / 0002 | 3.644 m | 2.492 m | 29 | 否 |
| 7GAhQPFzMot / 0003 | 3.314 m | 2.052 m | 25 | 否 |
| XNeHsjL6nBB / 0002 | 3.105 m | 2.720 m | 21 | 否 |
| LEFTm3JecaC / 0003 | 2.322 m | 2.374 m | 30 | 是 |

这排除了“它们都因为初始拒绝记忆而失败”的解释；没有排除错误定位、后来位姿失真、局部路线歧义或 controller 执行限制。不能把 `accepted` 当作 GT 正确。

另一个有用的反证：28 条 Revisits 的首次请求都在后半球，成功的 24 条也如此；四条失败后续后半球请求比例反而较低。仅凭“初始方向在后面”不足以把这四个失败都归因于缺 U-turn。

raw memory 也不是 DINO 独自给出方向。它读取 LingBot 几何/位姿，走未经 CEC 验证的目标位姿读出，再转成固定半径方向。它的高 SR 支持的是“有几何的历史记忆很有用”，并不自动否定 streaming-memory 主贡献。

## 4. 新发现：Pi3X 的定位方向与路线方向被混用

### 4.1 原始计量链

通过正式 run 的 launcher 找到实际运行源码：

`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_learned_handoff_42bd846fcbcc377f/`

其中 `MemNavData/bearing_diagnostics.py` 的 SHA-256：

`0c04cfb3ee1753c4de9ebd0da13ad51a9d413e08d32b7530c2d9761e78cd3838`

`evaluation_geodesic_bearing_error_deg()` 比较预测方向与 Habitat shortest path 的第一个非退化线段。正式 evaluator 仅在 learned accept 时写入这个角度；summarizer / independent verifier 读取它，计数 `>90°`。

但 Pi3X 的实际输出 `_scale_free_bearing(current_c2w, goal_c2w)` 是相对目标位置方向，并非最短路切向。CEC 的 canonical readout 也属于前者。

- 定位方位误差：预测方向 vs 当前到目标的直线相对方向。
- 路线方向差：预测方向 vs 绕障碍的最短路第一段。

两个量都可以测，但后者大于 90° 不能单独证明错误重定位。正确目标在墙另一侧时，可行路线完全可能先背离目标。

### 4.2 本轮原始记录重算

读正式 natural-direction 21 histories 的两个方法、两个 role，共 84 个 plans JSON；使用原 manifest 的 goal floor position 与各自 rollout 中同 step 的 x/z/yaw。没有调用策略、改变轨迹或读取 GT 作为运行时输入。

复算采用：

```text
delta = [goal_x - current_x, goal_z - current_z]
forward = [-sin(yaw), -cos(yaw)]
left    = [-cos(yaw),  sin(yaw)]
target_bearing = normalize([dot(delta, forward), dot(delta, left)])
endpoint_error = acos(dot(normalize(predicted_bearing), target_bearing))
```

另一遍使用 `atan2` 角差复算四个坏 query，结果一致。没有缺失 step/bearing；同一步复算距离与日志的最大差为 `4.44e-16 m`。

| 测量 | Pi3X | CEC |
|---|---:|---:|
| 接受的计划数 | 479 | 449 |
| 对目标直线方位的中位误差 | 4.066° | 2.609° |
| 对目标直线方位的 P95 | 88.249° | 12.825° |
| 对目标直线方位 >90° | 20/479 | 0/449 |
| 涉及的 query / scene 数 | 4 / 3 | 0 / 0 |

原 Pi3X `69/479` 是**对 geodesic 首段**的超 90° 次数；其中 38 次对目标直线方位误差实际 ≤30°。原 69 次里只有 20 次仍属 endpoint >90°，没有新增 endpoint >90° 而原 geodesic ≤90° 的记录。

示例：`8WUmhLawc2A / episode_0000 / Revisit / step 160`，距目标 1.795 m，原计量 104.158°，目标方位误差仅 10.214°。

84 个源文件路径与内容 SHA 按读取顺序组成 JSON 后的 aggregate SHA-256：

`5320f119d86128922f96c1aacc638b6eb1d1b58600bba68310b5c67c17108cf3`

原始 run root：

`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/`

注意：479 与 449 是各方法自身轨迹上的计划级计数，不是 479 个独立 episode，也不是同一状态上的逐帧配对比较。这个分析是事后诊断，不增加正式 SR 样本。

### 4.3 更有信息量的时序分解

| Pi3X query | >90° 次数 | 首次接受方位误差 | Pi3X / CEC outcome |
|---|---:|---:|---|
| 8WUmhLawc2A / 0000 / Revisit | 11 | 4.802° | 失败 / 成功 |
| PuKPg4mmafe / 0001 / Revisit | 3 | 4.281° | 成功 / 成功 |
| PuKPg4mmafe / 0004 / Novel | 4 | 3.275° | 成功 / 成功 |
| V2XKFyX4ASd / 0004 / Revisit | 2 | 3.179° | 成功 / 成功 |

四条都不是一开始就给出灾难方向；异常出现在后续 current-to-goal 更新。这支持优先检查**时序几何一致性 / 重定位更新**，而非直接归咎于首次检索或 proof-head 分类。

它尚未证明这些错误由哪一个内部机制引起：current pose、goal pose、桥接帧选择、视角变化、局部重建坐标系与运动闭环仍需区分。初始方位准确也不等于初始完整相对位姿准确。

原替换决定保持不变：Pi3X 为 19/21 Revisit，CEC 20/21；非劣效未通过，仍存在真实 endpoint 方向长尾。不能事后改写为通过。应修正的是“69 次全部属于错误定位，所以 learned 做不好”的解释。

## 5. 三个真正影响贡献判断的问题

### 5.1 最近简单替代解释尚未彻底排除

活动论文 Table I 清楚证明 memory-on 相对 memory-off 的收益；Table IV 与 fresh full-mono 说明严格验证限制了无支持干预，但当前 mono 总体 SR 对 raw / finite-PnP 的额外优势尚不稳定。

检索、局部匹配、PnP、自认证和 bearing-only 导航都有先例。AnyImageNav 已采用几何 query 与 self-certification，且明确依赖 depth/odometry；ViNT 的完整系统可包含拓扑规划；经典 teach-and-repeat 也用方向修正重走路径。不能声称这些单个构件是空白地带。

合理区别是：本项目的 episode-local causal RGB 几何、任意插入的只读 ImageGoal、无外部 pose 的历史定位、冻结策略的最小方向接口以及配对 evidence。无需宣称 LingBot 是唯一能实现这些能力的模型。

最小处理：完成已冻结 HM3D authority ablation，明确 raw 同样含有几何记忆；不新增一长串不匹配的 SOTA 对比。

### 5.2 时间跨度、空间长程和定位评价需要分开

长期保存 RGB 历史，不等于能自主走完 30 m 绕行路线。现有同层 23-history 长程诊断：native 0/23、endpoint CEC 0/23、route tangent 4/23，+4/-0，p=.125。后者 14/23 在 route geometry 上停止，余下 9 条有 4 成功、5 stuck。最新 rear alignment 是已消费 9 条上的 3/9→4/9，+2/-1、p=1，仍有 4 条 geometry stop。

这些不是“相机高度没用”：高度只处理尺度，不消除相对旋转/形状累积误差；终点方向也不携带路线连接顺序。初始 certificate 与连续 route-state estimation 是不同任务。

最小处理：当前论文保持 episodic reuse；把新 Pi3X endpoint/route 分解写入内部证据，先定位后续更新错误。若继续空间长程，研究局部视觉重锚定与路线进度，而不是加半径或削弱初始 certificate。相关路线已经有开发尝试，不能称新发现或许诺成功。

### 5.3 双读出的实际复用成本还不够理想

代码默认对新 anchor 的历史 depth 做一次 dense replay，再 cache；不是每一步重新做完整 certificate。已有 eager 方案维护第二个等价 dense state，用写入成本和显存换 first-query 延迟，并非免费加速。既有部分 KV + 短 suffix 方案也曾破坏深度等价，不能再提为无损新优化。

最值得研究的减法候选是：同一在线几何读出在写入时物化必要的历史稀疏几何，之后直接用于 witness，减少新 goal 的长前缀回放。当前 route cache 已使用过主流写入深度，但它明确不进入 canonical 初始 certificate；不能声称这一替换已被验证。

这条改动不天然构成新算法，却能让“同一几何支持两个时间尺度”的架构更实在。若深度状态语义不同，就必须作为新变体检验 certificate/bearing 与 SR，不能冒称缓存等价或继承旧正式分数。

## 6. 下一步排序：先澄清、再减法，不先长训

### P0：关闭已知证据问题

1. Table II SPL 已按真实位移修正；Table III 旧 SPL 仍不能直接投稿，保留 SR 并由作者决定移除该列或报告已算出的界限。无需为次要效率指标重跑 210 个轨迹。
2. 本次将 Pi3X 的 route disagreement 与 endpoint localization error 分开；旧正式未通过决定不变。后续与 CEC 都应使用同一目标定义，并区分初次定位和后续更新。
3. HM3D authority 缺失八个 block 已有 exact-repair 提交收据。本轮没有监控它的当前调度状态；不能把早晨的 PENDING 当作现在的状态，也不能声称已经有全量结果。

### P1：最值得先做的小范围机制工作

以现有四条 Pi3X endpoint-tail query 为开发诊断，逐时刻分解相对目标关系与 current-state 更新。对同一批观测比较“每次重估目标关系”与“保留一次目标关系、持续更新当前几何”才能区分漂移来源；先离线，不重训，不冒充 fresh 测试。

若这个因果分解支持持久化目标关系，再考虑复用现有 learned relative-pose readout 做一次历史目标定位，后续由共享 causal state 更新方向。这是待验证方向，不是现成修复。Pi3X 与 LingBot 不同重建的尺度/坐标系必须先正确对齐，不能直接拼 pose 矩阵；也不能把初始小 bearing error 当作完整 pose 正确。

这是比“重新训练一个大 decoder”更有依据的学习路线入口。它保留已经验证的寻址与时序状态，尝试替换局部关系估计，不让一个小 head 同时承担长历史寻址、开放集识别、相对定位和长程路线规划。

### P2：再决定是否投入新方法

- 如果目标是当前论文：冻结 canonical CEC，补齐公平的归因和真实成本，不因图或措辞显得不够新而改全部方法。
- 如果目标是显著提高空间长程 SR：必须增加可执行路线顺序与可靠局部进度；这将是新的路线读出研究，不能只靠修 scale 或 U-turn。
- 如果目标是减少工程拼接：优先实际共享几何、减少重复推理；新的统一 learned relocalizer 只有在同口径下保住 utility 与方向长尾才值得升级。

## 7. 来源与审计边界

活动稿：`/home/asus/Research/Memnav_Paper/main.tex`，包含 `sec_lg/3_problem.tex`、`sec_lg/4_method.tex` 与 `sec/5_experiments.tex` 至 `sec/9_conclusion.tex`。不是旧 `paper/` 或 `main_lg.tex` 入口。

已有关键报告：

- `PROJECT_REAUDIT_20260906.md`、`PROJECT_REAUDIT_REPAIRS_20260906.md`
- `PAPER_SPL_CORRECTION_20260906.md`
- `FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md`
- `HM3D_LONGRANGE_ROUTE_TANGENT_FORMAL_RESULT_20260903.md`
- `LONG_RANGE_REVISIT_COMPLETE_AUDIT_20260904.md`
- `CEC_LATENCY_OPTIMIZATION_RESULT_20260818.md`

2026-09-06 查阅的公开一手来源：

- [ICRA 2027 reviewer guidelines](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-reviewers/)
- [AnyImageNav 正文](https://arxiv.org/html/2604.05351v3)，特别是任务观测定义与 depth/odometry limitations
- [ViNT 官方项目](https://general-navigation-models.github.io/vint/index.html)，区分局部 policy 与包含拓扑搜索的完整系统
- [Navigation without localisation](https://arxiv.org/abs/1711.05348)，bearing/heading-based teach-and-repeat 先例

没有使用未公开标题、摘要或结果做检索词；没有上传论文或原始实验到外部服务。本轮不是独立盲审，不给出录用概率，也不把未跑的改进表述为有效方法。
