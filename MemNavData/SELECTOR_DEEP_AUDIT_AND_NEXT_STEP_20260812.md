# Novel / Revisit Selector 深度审计与下一步

日期：2026-08-12（CST）  
状态：证据审计与文献复核完成；提出 train-only 下一门，**尚未授权闭环、blind 或论文性能声明**。

## 0. 结论先行

我们已经充分验证：问题不是“还少一个更大的 selector”。过去的实验大多在重复组合同一批
静态证据——DINO 相似度、候选分数分布、单图 RANSAC、单 anchor LingBot 几何——然后希望
它们同时回答三个不同问题：

1. 历史里**哪个 anchor**最像目标；
2. 历史里**是否存在足够目标表面**，即 Novel 还是 Revisit；
3. 即使位置正确，memory residual **此刻是否会帮助控制**。

已有结果把这三件事分开后，结论很清楚：

- **排序不是当前主瓶颈。** 已知 Revisit 的 fresh 160-episode 配对中，raw-DINO top-1 direct
  达到 `109/118`，geometry router 为 `93/118`，配对 `+20/-4`，exact McNemar
  `p=0.0015438795`。
- **unknown-goal existence 尚未解决。** 最好的 factorized selector 在相同严格风险下没有超过
  hard geometry；主要损失是有正例却 abstain，而不是激活后选错。
- **RANSAC 测的是局部 place consistency，不是 goal-surface existence。** 真 Revisit-C 中
  pass precision `90.9%`，Novel 起点却只有 `34.2%`；同时它会持续拒绝真实有用 anchor。
- **重复观察同一静态 goal-memory 对不是新时序证据。** hard pilot 中，一个 Novel 假阳性
  连续 `63/63` plans 稳定通过，另一个 Revisit 真阳性连续 `35/35` plans 被 geometry 拒绝。
- **正式 LingBot Phase-B 数据从未真正采集多视角共识。** train `1,098/1,098` rows、dev
  `933/933` rows 均为 `n_hypotheses=1, neighbor_offsets=0`，pose dispersion 全为缺失；这条
  通道只在早期 6 scenes、24 candidates smoke 中出现过，尚未做 unknown-goal 全量检验。

因此下一项最小而有信息增量的实验不是再训 gate，而是：

> **Memory Renderability Certificate（MRC）**：raw DINO 只提出 top-1 历史位置；从该位置
> 周围取已有的历史局部片段，用冻结 LingBot/Pi3 类多视角几何判断目标图能否在同一局部 3D
> 坐标系中被稳定注册和覆盖。能自认证才判 Revisit，否则退回 Novel/native。

它首先是一个待证伪的 train-only 资格实验，不是已经有效的方法。对于当前 metadata 已明确
给出 goal kind 的 benchmark，最简单、已有显著证据的系统仍是：`Novel -> native NavDP`，
`Revisit -> raw-DINO direct residual`。自动 Novel/Revisit 是更严格的部署扩展，不能混写成
当前 benchmark 的必要组件。

---

## 1. selector 到底在判断什么

### 1.1 数据中的操作定义

生成器不是用“是否出现过同一 JPEG”定义 Revisit：

- Novel-B：目标视图相对此前每个历史帧的最大 occlusion-aware co-visibility `<0.10`；
- Revisit：在历史轨迹 anchor 附近最多 `1.5 m`、朝向差最多 `45°` 采样，并要求可见表面重叠；
- selector teacher：candidate co-visibility `>=0.50` 为 positive，`<=0.20` 为 negative，
  中间为 ambiguous/ignore；
- Revisit-C 的目标不是 Leg-A goal JPEG 的重复发送，故哈希缓存或“见过这张图吗”不能解决。

所以真正的 latent 是：

```text
memory 中是否有足够的 goal-image 表面，可以产生任务上有用的历史位置/方向？
```

它比普通图像相似度更严格，也比“是否同一个房间”更严格。

### 1.2 四个曾被统称 selector 的模块

| 层级 | 正确问题 | 当前证据 |
|---|---|---|
| Proposal / rank | 若存在，哪个历史 anchor 最好？ | raw DINO 已很强；learned geometry 有相对排序信号，但没有稳定 top-1 增益 |
| Existence | 历史中是否存在足够目标表面？ | **未解决；当前唯一真正的 unknown-goal 缺口** |
| Pose / direction | anchor 到目标的相对方向是否可信？ | RANSAC 可作正支持；LingBot 单 anchor pose 有用但不等于 existence |
| Actionability | 使用这个 residual 是否优于 native？ | behind/front/critic/endpoint 等简单 gate 均未泛化；这是 existence 之后的独立问题 |

后续实验必须明确自己改的是哪一层；否则 candidate AUC、session accuracy 和 closed-loop SR
还会继续被错误互换。

---

## 2. 所有 selector 家族的去重总账

### 2.1 静态 DINO / 阈值家族

| 尝试 | 最硬结果 | 结论 |
|---|---|---|
| DINO 固定阈值 | 旧自动报告用 0.5 得到 `21.7%`，公平 train 阈值后为 `87.3%` | 旧“79.2 vs 21.7”优势无效；阈值必须公平迁移 |
| max-DINO / DINO-only set model | learned set model candidate top-1 `67.65% < 82.35%` | 增加小模型不等于增加信息 |
| GLP Stage 1 | `88.57%`，与 max-DINO 打平 | posterior 框架本身无害，但单 DINO 没有新增证据 |
| score distribution / F8 | F8 没稳定超过 F2/H；一 seed 的 strict FP 超预算 | 不再扫 top-4/8/16、temperature 或 margin |

额外只读探索也复现同一结论：在 436 个 train extreme sessions 上，DINO existence AUC 约
`0.885`；top-vs-median/mean 的训练自由分数约 `0.67`，frame-index spread 最好约 `0.78`。
这不是预注册结果，只用于排除“漏了一个显然免费的 score statistic”这一可能。

### 2.2 RANSAC / 几何硬门家族

| 证据 | 数字 | 正确解释 |
|---|---:|---|
| 旧 2-leg 闭环 | native `4/40` -> geometry `19/40`，`+15/-0`，`p=6.1e-5` | memory residual 在已知 Revisit 分布上很有价值 |
| R0 闭环复现 | native `3/40` -> geometry `20/40`，`+17/-0`，`p=1.53e-5` | 当前最硬的可部署 memory baseline |
| 全 train candidate | stable-support precision `76.2%`；positive recall `58.78%` | 全局 binary expert 不够安全且漏检 |
| 真 Revisit-C | pass precision `90.9%`，recall `66.3%` | 条件化后是高精度、有限召回的正证据 |
| Novel 起点 | pass precision `34.2%` | 会把同房间/背景匹配误当目标存在 |
| fresh known-Revisit | geometry `93/118`，raw DINO direct `109/118` | hard veto 丢掉了大量有用 proposal |

RANSAC 应保留为局部 pose/support 证据，但 `reject/insufficient` 不能等价为 Novel，pass 也不能
单独等价为 Revisit。

### 2.3 top-K / temporal-NMS / latch 家族

| 尝试 | 结果 | 结论 |
|---|---|---|
| cV4 单 episode temporal-NMS | 找到 raw rank-20 的正确簇并救活 | 有效的机制例子，不是总体效果 |
| 正式 K=1 vs K=8 | `18/40 vs 18/40`，`+1/-1`，`p=1.0` | 候选宽度/相邻帧挤占不是总体瓶颈 |
| geometry top-1/2/4/8/32 | 正例 top-1 只从 `116` 增至 `120`，strict FP 从 `5` 增至 `67` | 扩 K 的风险增长快于收益 |
| 两次确认 / temporal latch | hard natural stream 中稳定假阳性 `63/63`、稳定假阴性 `35/35` | 重复同一静态证据会锁住错误，不会产生信息 |

这里必须区分两种“temporal”：旧 Patch+temporal 的 temporal 是**历史库内相邻帧上下文**；
planning-stream latch 是**对同一 goal-history 检索的重复调用**。后者在 goal 和 eligible memory
都不变时不是独立观测。

### 2.4 learned router / patch / Phase-B / GLP 家族

| 尝试 | 强项 | 最终 decision-unit 结果 |
|---|---|---|
| patch+temporal pointwise | 离线 acc `96.3%`、AUC `0.991` | pooled shortcut；没有闭环优势 |
| task-aligned listwise | dev top-1 `30/35` vs DINO `24/35` | development 已耗尽；不能当 final |
| Phase-B | candidate AUC `0.9535` vs `0.9103`，独立复现 | top-1 `25/34 < 26/34`；session `72.7% < 87.3%` |
| OOF calibration | threshold migration loss约 `13.6 -> 6.4 pp` | `84.5% < 87.3%`，isotonic 无额外改善 |
| GLP Stage 2 | 能组合似然 | `70.9% < 87.3%`，未通过 |

这些结果不是“几何特征没信息”，而是：相对排序可以在同 session 消去尺度；绝对 existence
必须跨 scene 校准。继续增大 MLP、shared decoder 或重复调阈值不会改变可观测性。

### 2.5 Factorized unknown-goal F2 / F8

F2 已经修正了旧模型最重要的设计问题：existence 与 conditional ranking 分开、nested
scene-OOF、三 seeds、同一 geometry 风险预算、完全不输入 phase/goal role。

在 40 train scenes、436 extreme sessions 上：

| 系统 | Correct support | Correct anchor | Wrong anchor | Strict FP |
|---|---:|---:|---:|---:|
| hard geometry H | `365/436` | `93/155` | 14 | `9/281` |
| F2 seed 1 | `359/436` | 85 | 12 | 7 |
| F2 seed 2 | `365/436` | 89 | 12 | 5 |
| F2 seed 3 | `365/436` | 88 | 13 | 4 |

F2 existence AUC 为 `0.898–0.909`，高于 DINO `0.888`，conditional ranker 相对 DINO 有
`10/4–5` 的方向性优势；但主要错误是 `34–38` 个“top-2 有 positive 却 abstain”，不是
激活后选错。F8 加入 top-8 静态分布后 correct anchor 为 `90/89/90`，仍未超过 H 的 93，
且一 seed strict FP 为 11。预注册门均未通过。

这已经充分排除：

- 再分解一次 classifier；
- 再加 score/match/inlier summary；
- 再扫 logistic `C`、temperature 或阈值；
- 再扩大候选 K。

### 2.6 natural-stream 与 actionability 家族

- natural-stream hard pilot 并未证明时序无用；它证明**同一静态检索的 persistence 无用**。
  真正可能有用的时序必须包含新的 current-query 视角、odometry/pose-chain 或主动验证动作。
- `forward<0 -> native` 在 post-hoc pLe 上 `+1/-0`，但正式 T1 前四场景立即 `+0/-4`，被拒绝。
- endpoint、critic、behind bearing、早期 bearing slope 没有成为可靠 safety selector。
- active-glance V1/V2/V3 最好仍只有 `25/40`，低于 native `31/40`；原地转圈或每步 glance
  不能作为下一简单解法。

这些都是 controller/actionability 结果，不能拿来证明 existence head 已经解决或不可能解决。

---

## 3. 失败的共同根因

### 3.1 证据语义与目标语义不一致

DINO 回答“外观像不像”，RANSAC 回答“局部特征能否满足一个二维几何模型”，但标签回答
“目标图的大量可见表面是否已被历史观察”。同房间墙面、门框或重复纹理足以让前两者为真，
却不保证目标表面可用于导航。

### 3.2 大多数所谓时序没有增加观测

goal image 固定、历史 memory 在 goal switch 后固定或只追加当前自然轨迹，而 selector 每次
仍主要重算 `goal vs old anchor`。JeFG 与 YVUC 的持续错误说明，更多 plan 数不等于更多独立
样本。文献中有效的 query-history 方法会把**当前移动、odometry 或新的验证视角**纳入约束，
而不是对同一 pair 投票。

### 3.3 优化目标与部署指标错位

- pooled candidate AUC 会被 scene/episode 难度捷径抬高；
- candidate ranking 不包含 no-match；
- session oracle threshold 不代表 train-to-unseen threshold 可迁移；
- offline correct support 也不等于 residual 能带来闭环 SR。

因此下一轮仍须先过 `correct anchor / wrong anchor / strict FP`，再谈闭环。

### 3.4 形式上最关键的缺失：多视角共识其实没被正式测过

仓库审计结果：

```text
phase_b_train_repaired: 1098 rows; n_hypotheses=1 全部；neighbor_offsets=0 全部
phase_b_dev_repaired:    933 rows; n_hypotheses=1 全部；neighbor_offsets=0 全部
translation/rotation dispersion: 两份 artifact 均 0 个 finite value
```

`phase_b_feature_schema.py` 也明确把 pose-dispersion 三件套标成 deferred。早期 6-scene
完整 replay 的 12 positive + 12 hard negative 中：DINO AUC `0.549`、cloud overlap `0.694`、
pose consistency `0.736`、pose refinement `0.708`。N=24 只能证明这条通道有信号，不能选
阈值；但它证明“形式上没测”和“实现完全没有”是两回事——现有 collector 已支持
`neighbor_offset=-4,0,+4`。

---

## 4. 文献复核：哪些路线值得做，哪些已被我们覆盖

| 文献 | 原论文结论 | 对本项目的含义 |
|---|---|---|
| [Zaffar et al., CVPR 2024, Image-matching Uncertainty in VPR](https://openaccess.thecvf.com/content/CVPR2024/html/Zaffar_On_the_Estimation_of_Image-matching_Uncertainty_in_Visual_Place_Recognition_CVPR_2024_paper.html) | 简单 descriptor distance 很强；SUE 用 top-K reference poses 的空间散布估计不确定性，并与几何验证互补 | 应保留 raw DINO 强基线；真实 pose-spread 可作廉价对照，但 F2 的 top-2 pose agreement、F8 与 frame-index spread 已说明静态集合统计不应成为主赌注 |
| [Miller et al., 2025, Through the Lens of Doubt](https://arxiv.org/abs/2510.13464) | 从 similarity distribution 构造无训练 distinctiveness/ratio uncertainty | 我们的 F8 与附加只读统计已基本覆盖该家族；未转成更好的风险 operating point |
| [Sferrazza et al., CVPRW 2025, To Match or Not to Match](https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Sferrazza_To_Match_or_Not_to_Match_Revisiting_Image_Matching_for_CVPRW_2025_paper.html) | 现代 retrieval 已很强，local matching rerank 可能反而降级；更适合把 matching 当 verification/confidence | 直接支持“raw DINO 选 top-1，几何只认证、不重排、不 hard-veto”的角色分工 |
| [Claxton et al., 2024, Verifying Localization Estimates](https://arxiv.org/abs/2407.08162) | history-of-queries 通过最近已验证 match 加 odometry 外推，提高真实机器人定位完整性 | 有效时序依赖新 query motion/odometry；解释为何我们的静态 latch 无效。pose-chain consistency 是 MRC 失败后的备选，不是再投票 |
| [Deng et al., 2026, AnyImageNav](https://arxiv.org/abs/2604.05351) | semantic relevance 先粗门控，再调用 multi-view 3D foundation model；用内部 correspondence confidence 自认证注册 | 与下一步最直接：目标图应被当作几何 query；但其方法已非常接近，不能把“多视角自认证”本身宣称为我们的独创 |
| [Guo et al., ICCV 2025, IGL-Nav](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.html) | 构建增量 3D 表示，先 coarse localization、近目标再 differentiable refinement | 说明显式 3D 目标注册有效，但完整 3DGS 对当前 selector 资格实验过重 |
| [Lei et al., CVPR 2024, IEVE](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html) | 对歧义实例采取“走近再验证”而非一次判死 | 主动 probe 是合理的第二阶段；当前 actionability/active-glance 证据不足，先不让机器人为 selector 付出运动代价 |
| [Yanko & Shavit, 2026, KappaPlace](https://arxiv.org/abs/2605.19435) | vMF match-level uncertainty，可给 frozen backbone 加 post-training confidence module | 值得作为未来 learned baseline；但我们已多次遇到跨 scene calibration failure，不应先于新几何证据 |
| [Wang et al., ICLR 2026, Pi3 / Pi3X official repository](https://github.com/yyfz/Pi3) | unordered multi-view geometry，Pi3X 增强 confidence，并可注入 pose/intrinsics/depth | 本仓已包含 Pi3/Pi3X 源码；但本机未发现 checkpoint，首轮优先复用已可工作的 LingBot collector，避免先引入新依赖 |

文献与本项目证据共同指向的不是“再学一个 selector”，而是一个 coarse-to-certificate cascade：
检索负责 recall，几何负责证明目标图确实能由局部历史解释。

---

## 5. 候选下一步的排序

| 候选 | 新信息量 | 成本 | 决定 |
|---|---:|---:|---|
| 再调 DINO threshold / margin / temperature | 低 | 低 | 拒绝：已重复覆盖且 calibration 漂移 |
| 再加 top-K score/inlier statistics | 低 | 低 | 拒绝：F8 已正式失败 |
| 再换 MLP/Transformer selector | 低 | 中 | 拒绝：容量不改变可辨识性 |
| top-K 真实 pose spread（SUE） | 中低 | 很低 | 只作强廉价 baseline，不作为主方法 |
| current-query + odometry pose-chain integrity | 高 | 中 | 备选；确有新时序，但需要部署 VO/pose 契约 |
| 走近 / active verification | 高 | 高且有行为风险 | 后置；先证明被动历史片段自身可认证 |
| **局部历史多视角 renderability certificate** | **高** | **中低，无训练/无 Habitat rollout** | **下一主实验** |

---

## 6. 下一主实验：MRC-v0

### 6.1 一句话架构

```text
goal image + causal memory
        |
        | raw DINO（只做 proposal）
        v
top-1 anchor a*
        |
        | existing historical clip at a*: offsets {-4, 0, +4}
        v
frozen LingBot multi-hypothesis geometry
        |
        +-- goal-surface coverage across the clip
        +-- goal-pose translation / rotation consensus
        +-- refinement and internal confidence
        v
renderability certificate
   pass  -> Revisit: keep the same raw-DINO top-1 anchor -> existing residual
   fail  -> Novel: frozen native NavDP
```

三个设计约束：

1. **不 rerank。** fresh known-Revisit 已证明 raw DINO top-1 的闭环价值；certificate 只回答
   existence。
2. **不重复 planning-time 计算。** goal switch 时计算一次并缓存；没有新历史结构时不靠 plan
   次数累计假置信度。
3. **不把单图 reject 当负证据。** 需要局部片段对 goal surface 的覆盖与 pose 共识；这正是
   单图 RANSAC 缺少的量。

### 6.2 为什么它不是旧实验换名字

- 不是 top-K：只使用 raw-DINO top-1，邻居用于认证同一 hypothesis，不成为动作候选；
- 不是旧 patch+temporal：旧方法对历史帧 patch/索引做 rerank；MRC 检验一个 goal pose 能否
  在共同 3D 坐标中由多个相邻历史视角支持；
- 不是 natural-stream latch：三幅历史图提供不同视角/基线，而不是重复同一 pair；
- 不是 Phase-B v3/F2/F8：正式 artifact 的 `n_hypotheses` 全为 1，pose-consensus 根本没进入；
- 不是 RANSAC：目标是 dense goal-surface coverage + 3D pose agreement，而非单 pair 的稀疏
  essential-matrix pass。

### 6.3 最小数据与实现

只用 40 train scenes，禁止 development、旧 20-scene consumed outcome、blind：

- 480 sessions 中每 session 只收 raw-DINO top-1；
- `neighbor_offsets = {-4,0,+4}`；边界不足时 fail closed 或记录 2-view mask，不伪造共识；
- 复用 `diag_lingbot_goal_loop_closure.py`、现有 4.4 GB LingBot 权重和 causal cache；
- 固定记录 goal-to-candidate coverage、candidate-to-goal coverage、F1、normalized translation
  dispersion、rotation dispersion、refinement、model confidence 和 latency；
- 先跑 24-session timing/ABI smoke。历史 24-candidate H200 完整 replay 用时 `10m28s`；若近似
  线性，480 个 top-1 约 `3.5 h` H200 wall time，但正式预算必须由 smoke 实测，不能把此外推
  当承诺。它是 feature collection，不是 6–8 小时 Habitat eval。

Pi3X 只作后续对照：源码已在仓库，但 checkpoint/依赖尚未形成当前可复现收据；不应为了“更新”
先重写可工作的 LingBot 路径。

### 6.4 冻结评测门

评测单元仍是 436 extreme sessions：155 positive + 281 strict no-match。建议在收集前冻结：

1. **proposal 不变**：raw DINO top-1；
2. **三条 baseline**：DINO risk-matched、hard geometry H、上一轮 F2；
3. **certificate 特征很小**：DINO relevance + goal-surface coverage + translation/rotation
   consensus；不加 hidden state、phase、goal role；
4. **nested scene-OOF**：outer 5 folds、inner 4 folds、原三 seeds；inner OOF 决定 risk-matched
   operating point；
5. **主指标**：correct anchor、wrong anchor、strict FP、correct support；AUC/AP 只作诊断；
6. **Go 条件**：三 seeds 全部满足
   - strict FP `<=9/281`；
   - wrong anchor `<=14/155`；
   - correct anchor `>93/155`；
   - 相对 H 的 scene-cluster 差不出现稳定负向；
7. 任一 seed 失败：停止 MRC-v0，不加 offset、不调模型宽度、不进闭环。

这里 `93`、`14`、`9` 来自冻结 hard-geometry H，不是看完 MRC 后选择的数字。MRC 若只提高
AUC 而不提高 risk-matched correct anchor，视为失败。

### 6.5 通过后才做什么

只有离线门通过，才运行 unknown-goal shadow：

- Novel 与 Revisit 计划流只记录一次 cached decision；
- 检查 certificate 是否随错误 memory 稳定误激活，以及计算延迟；
- 再冻结一个小型 paired 3-leg/role-free 闭环，而不是先跑 20 scenes 长评测；
- 最终方法效果必须报告相对 native 与 known-role upper bound 的差距。

若离线门失败，正确决策不是再调 selector，而是把论文/benchmark 主线固定在已知 goal-kind：
Novel 用 native，Revisit 用 direct memory residual；unknown goal-kind 明确列为扩展限制。

---

## 7. 创新性与口径

MRC 比“两 expert 投票”更整洁，因为它把定义与测量对齐：

```text
Revisit 不是某个分类标签；Revisit = goal image is renderable from episodic memory.
```

但必须诚实：AnyImageNav 已经提出 semantic-to-geometric multi-view self-certification，IGL-Nav
也使用显式 3D 目标定位。因此不能把“多视角注册”本身包装成空白创新。若本项目最终成立，较
有辨识度的贡献应是三者组合：

1. **causal episodic historical clip**，而不是当前邻近视角；
2. **unknown goal-kind 的 memory existence certificate**，而不是 last-meter stopping；
3. **base-policy-preserving residual**：认证失败零干预，认证成功只供应历史方向/位置。

创新性最终仍取决于 untouched-scene closed-loop 是否超过 raw DINO/geometry/native，不能由
架构图决定。

---

## 8. 当前冻结决策

```text
do_not:
  - tune another static selector
  - widen top-K
  - train another shared decoder / gate
  - run long Habitat eval before the offline risk gate
  - use development or consumed outcomes to choose certificate thresholds

next:
  - pre-register MRC-v0
  - run 24-session timing/contract smoke
  - collect top1 x {-4,0,+4} on all 480 train sessions
  - run the same nested scene-OOF risk gate

current_evidence_backed_benchmark_candidate:
  known Novel   -> native NavDP
  known Revisit -> raw-DINO top1 direct memory residual
```

本文件提出的是目前**信息增量/实现成本最优**的下一次检验，不预测它一定成功。
