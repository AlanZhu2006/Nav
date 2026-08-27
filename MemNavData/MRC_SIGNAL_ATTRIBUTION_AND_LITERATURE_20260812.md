# MRC 信号为何难以转成 Unknown Novel/Revisit 决策：归因审计与文献复核

日期：2026-08-12（CST）  
状态：完成本地归因；**冻结 full-HPC MRC Stage F，不提交 480-session 长任务**。

## 0. 结论先行

MRC 的多视角几何信号不是假的，也不是 LingBot 完全没有能力。现有证据支持更窄的结论：

> 它能判断“在一个已知 Revisit、同一场景、候选已经给定的条件下，哪个 anchor 的几何解释
> 更稳定”；但我们要求它判断的是“部署时历史里是否存在任务上可用的目标支持”。

前者是**相对候选验证 / pose-quality**，后者是**开放集存在性 / decision-unit routing**。
当前 MRC-v0 用一个跨场景绝对阈值把二者当成同一问题，因此无法稳定兑现信号。

这不是单一 bug，而是五个结构因素叠加：

1. 旧正证据只来自已知 Revisit 且强制正负成对的条件分布，完全没有 strict no-match；
2. top-1-only certificate 无法区分“真 Novel”和“Revisit 但 proposal 选错”；
3. 几何量的绝对尺度主要由 scene/viewpoint/texture 决定，标签只解释很小部分方差；
4. `[-4,0,+4]` 的三个 replay 高度相关，名义三视角约等于 `1.1–1.2` 个独立观测；
5. 模型自洽、优化收敛和内部 confidence 衡量的是**模型自己的稳定性**，不是 task surface
   真值，也不是 residual 的闭环动作价值。

因此此时把 24-session smoke 线性放大到 480 sessions，只会花约 `9.4 GPU-hours` 去获得更精确
的同一种混合信号；它不会自动修复可辨识性。正确动作是先在本机做短、能区分根因的对照。

---

## 1. 它真正要预测什么

### 1.1 teacher 标签

当前 unknown-goal teacher 的 candidate label 是 task-aligned co-visibility：

- `teacher_covis >= 0.50`：positive；
- `teacher_covis <= 0.20`：strict negative；
- 中间为 ambiguous/ignore；
- session 是否 Revisit 由**整个 causal memory 中是否存在 positive**决定。

它不是“同一个房间”、不是“能估一个相机 pose”、也不是“外观相似”。正确 latent 是：

```text
E = causal episodic memory 中是否存在足够目标表面，能产生任务上有用的历史支持？
```

### 1.2 MRC-v0 实际观测

当前 extractor 对 raw-DINO top-1 周围三个历史帧分别重放相同的因果前缀，然后把同一 goal
image 人工 append 到每个 anchor，导出：

- 三次预测 goal pose 的 translation/rotation dispersion；
- candidate/goal 预测点云 overlap；
- camera-head refinement 幅度；
- depth confidence。

这些量直接观测的是：

```text
G = frozen geometry model 是否能对这个 goal-anchor pair 给出稳定、可配准的内部解释？
```

`G` 与 `E` 有相关性，但不等价：重复墙面、门框和房间布局可以让 `G` 高而 `E=0`；低纹理、
遮挡、视角差或预测深度误差又可让 `G` 低而 `E=1`。

---

## 2. 现有数据逐层说明了什么

所有数字可由
[`audit_mrc_signal_attribution.py`](./audit_mrc_signal_attribution.py) 复算。脚本明确不读取正式
24-session contract smoke 的 label，避免违反冻结协议。

### 2.1 旧 90-row artifact：有信号，但问题被条件化了

`multiscene100_20260806_job_15400645` 的 exact-three-view 部分为：

- 90 candidates，25 sessions，22 scenes；
- 46 positive / 44 negative；
- 全部 `kind=revisit_b`；
- sampler 只保留同时含 positive 和 negative 的 session。

因此它只估计：

```text
P(correct candidate | known Revisit, session has both classes)
```

而部署需要：

```text
P(memory support exists AND top1 is useful | unknown goal kind)
```

这两个总体不是同一个总体。旧 artifact 删除了全部 Novel/no-match 情况，还人为平衡 class，
所以它能证明 anchor verification signal，却不能证明 existence signal。

原始 AUC：

| 信号 | raw candidate AUC |
|---|---:|
| DINO | 0.596 |
| cloud overlap | 0.737 |
| translation pose consistency | 0.657 |
| rotation pose consistency | 0.571 |
| translation refinement | 0.684 |
| rotation refinement | 0.585 |

scene-LOO 小模型的 `D+H` AUC 为 `0.735`，且相对 DINO 的 scene-cluster bootstrap CI 为
`[+0.032,+0.272]`。所以“几何完全没信息”已被排除。

### 2.2 scene nuisance 大于 label effect

对同一 90 rows 做只读方差归因：

| 特征 | raw AUC | scene 内 z-score AUC | scene identity 解释的 R² | label 解释的 R² |
|---|---:|---:|---:|---:|
| DINO | 0.596 | 0.741 | 0.687 | 0.017 |
| cloud overlap | 0.737 | 0.917 | 0.398 | 0.188 |
| translation dispersion | 0.657 | 0.696 | 0.584 | 0.008 |
| rotation dispersion | 0.571 | 0.665 | 0.664 | 0.001 |
| translation refinement | 0.684 | 0.786 | 0.344 | 0.130 |
| rotation refinement | 0.585 | 0.605 | 0.176 | 0.013 |

R² 是描述性线性分解，不是因果效应；但比例差已经足够说明问题。最典型的是 rotation
dispersion：scene identity 解释约 `66.4%` 方差，candidate label 只解释 `0.1%`。cloud
overlap 在同 scene 相对比较时 AUC 从 `0.737` 升到 `0.917`，说明真正可利用的是**局部相对量**，
不是跨 scene 的绝对阈值。

这也复现了 Phase-B 的现象：candidate AUC 提升，但 train threshold `0.397` 到 dev `0.807`，
session decision 反而失败。不是简单“再校准一次”就一定能修复；scene shift 下 score 的含义
本身在变。

### 2.3 三视角不是三张独立选票

从每行 `hypotheses_json` 直接计算三个 offset 的相关性：

| 信号 | 三个 pair correlation | 平均相关 | exchangeable 近似有效观测数 |
|---|---|---:|---:|
| cloud overlap | 0.838 / 0.730 / 0.894 | 0.821 | 1.14 |
| normalized translation refinement | 0.836 / 0.733 / 0.752 | 0.773 | 1.18 |

原因来自实现本身：三个 inference 使用相同 goal、同一 model、嵌套且大幅重叠的 replay prefix，
只有末端少量帧不同。它们可降低随机噪声，却不能消除共享 bias；一个模型对 alias 的稳定错误会
在三次都稳定。

本机 11 条的真实轨迹审计显示 clip endpoint span 多数约 `0.30 m`，但最小只有 `0.145 m`；
最大视角变化从近 `0°` 到 `22.6°`。所以 frame offset 保证了时间差，却没有冻结几何条件数或
视角多样性。

### 2.4 top-1 certificate 存在结构上限

在 repaired Phase-B train 的 480 个部署 session：

- 155 positive sessions；
- raw-DINO top-1 只有 `115/155` 是正确 positive；
- 另外 40 条 top-1 为 wrong/ignore；
- 其中 13 条在 deployment top-2 仍有 positive，27 条 top-2 完全 miss；
- 281 strict no-match；44 ambiguous。

MRC-v0 被协议禁止 rerank，只能认证 top-1。因此即使 certificate 本身完美，正 session 的
proposal recall 上限也是 `115/155 = 74.2%`。对于剩下 40 条：

```text
Revisit + wrong top1  --MRC sees--> “这个 top1 不可注册”
Novel                 --MRC sees--> “这个 top1 不可注册”
```

二者对 top1-only verifier 是 observationally indistinguishable。它只能 abstain，无法判断
memory 中别处其实有 support。这解释了 F2 为什么主要错误是 `34–38` 条正 session 过度 abstain，
而不是大量 wrong activation。

### 2.5 本机 11-row pilot：它更像 pose-quality 信号

两 scene 的 exploratory deployment-top1 pilot 只有 9 个 signed rows，不能用于阈值结论；但
它能给出反例：

- translation dispersion candidate AUC `0.944`，说明 signal 很可能真实；
- 一个 true positive 的 cloud overlap 恰为 `0.0`，否定 overlap hard gate；
- translation refinement AUC `0.389`，否定“小更新就是正确”的直觉；
- 一个 ambiguous row 的 geometry-only transfer score 达 `0.851`；
- 固定模型跨 artifact 迁移时，geometry-only AUC `0.778`，加入 DINO 后降到 `0.556`。

更关键的是，内部 consistency 与 GT metric pose error 在这 11 条上有方向一致的 Spearman
相关：translation dispersion 对 position/direction/rotation error 分别约
`0.71 / 0.89 / 0.73`。这反而说明 LingBot signal 主要在做它擅长的事——估计 pose 是否稳定；
问题是我们拿它监督了另一个量 `teacher_covis >= 0.5`。

一个 ambiguous (`covis=0.350`) case 的 position error 仅约 `0.053 m`，而一个 strict no-match
case 虽 distance 错得很大，bearing direction error 约 `9.1°`。N 太小，不能据此宣称 Novel
也可定位；但它足以表明 binary surface-support label 与“这个预测是否有动作价值”并非一一对应。

---

## 3. 根因树

### 根因 A：目标语义错位（最高置信度）

```text
DINO                  -> 外观相似
SIFT/RANSAC           -> 单 pair 的局部 2-D 几何可解释
LingBot MRC           -> 冻结模型的 3-D/pose 自洽与可注册性
teacher               -> memory 中 task goal-surface 的 co-visibility
closed-loop desired   -> residual 是否比 native NavDP 更有动作价值
```

这五个变量相关但不同。之前 RANSAC 在真 Revisit 中 precision `90.9%`，在 Novel 起点只有
`34.2%`，已经展示同一种错位；MRC 只是用更强模型和 dense geometry 重做了相邻但不相同的量。

VPR 文献也明确强调“place”取决于 agent、environment 和 downstream task，并建议用视觉重叠
定义 matching，而非简单外观或位置；同位置、不同朝向也未必是 match
([Garg et al., 2021](https://arxiv.org/abs/2103.06443))。本项目 teacher 比普通 place
recognition 更严格，因为还要求对具体 goal surface 有用。

### 根因 B：相对排序被错误转换为绝对 open-set 判定（高置信度）

同 scene 中，正负候选共享 texture、深度尺度、运动模式和模型误差，因此差值能消去 nuisance；
换 scene 后 absolute overlap/dispersion 不再同尺度。Phase-B、F2、MRC 三条线都重复了“相对
信息好、绝对 activation 漂移”的模式。

[Zaffar et al., CVPR 2024](https://arxiv.org/abs/2404.00546) 发现简单 descriptor distance
已是很强的不确定性基线，reference-pose set uncertainty 与昂贵 geometric verification 互补，
而非由某个 learned/geometric score 单独解决。其跨数据集实验中，从 Pittsburgh 学到的组合
边界在 MSLS 也不是最优，和本项目的跨 scene threshold migration 同方向。

[Sferrazza et al., CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Sferrazza_To_Match_or_Not_to_Match_Revisiting_Image_Matching_for_CVPRW_2025_paper.html)
进一步发现现代 retrieval 已很强，local matching rerank 甚至可能降级，更适合作为 verification
confidence。这与本项目 “raw DINO proposal、geometry 只认证” 的角色分解一致，但不支持用
geometry absolute score 独自完成 open-set existence。

### 根因 C：采样问题把 existence 伪装成 verification（确定）

旧 90-row 数据的 sampler 明确丢弃没有正负两类的 session。这个设计对 feasibility 是合理的，
但它不能估计 strict no-match 的 false-positive tail。AUC 在这个总体上再高，也没有回答部署
问题。

VPR tutorial 特别区分“返回一个 match”“返回所有 match”和 no-match 评价单元，指出不同
problem type 不能直接比较
([Schubert et al., 2023](https://arxiv.org/abs/2303.03281))。我们的 candidate AUC 与
session existence 正是两种 evaluation unit。

### 根因 D：自洽不等于正确，confidence 也不是概率（高置信度）

三个 replay 使用同一模型与重叠输入，所以 shared hallucination / alias 可同时精确。refinement
小只说明 camera head 停在局部 fixed point，不说明 fixed point 是真 pose。

这不是 LingBot 独有问题。2026 concurrent 的
[Trust3R](https://arxiv.org/abs/2605.19539) 直接指出现有 feed-forward geometry model 的
confidence 多为 heuristic、缺少概率解释，常不能说明预测 geometry 在哪里可信。更早的
[Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
也显示 post-hoc calibration 在 dataset shift 下会失效；这解释了为什么 i.i.d. 上有效的
temperature/threshold 不能自动修复 unseen-scene shift。这里需要区分：
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) 证明 temperature scaling
在常规 held-out calibration 上常有效，但没有授权它跨 domain 保持校准；本项目恰好处于后者。

### 根因 E：成功的 multi-view geometry 系统有我们没有的前置条件（高置信度）

[AnyImageNav](https://arxiv.org/html/2604.05351v3) 并不对任意 memory top-1 直接做全局
Novel/Revisit 判定。它先用 semantic relevance 确认 agent 已在目标视觉邻域，才在近期自然
轨迹窗口上调用 geometry；confidence 在窗口内部相对归一化，long-memory alignment 还要求
至少三个不同位置。其 ablation 显示 2 帧因 viewpoint diversity 不足下降，6 帧又因局部 overlap
被稀释而下降；失败分析中最大项仍是 viewpoint diversity 不足或 foundation model pose 不可靠。

当前 MRC-v0 跳过了两个条件：

1. 没有“已进入 visual neighborhood”的 proximity precondition；
2. 没有按实际 baseline/condition number 选择视角，只按 frame index 取三帧。

所以“AnyImageNav 的几何认证有效”不能直接外推为“MRC 可以判断全局 memory existence”。

### 根因 F：proposal 与 certificate 串联后，错误被合并（确定）

certificate 只看 top-1，拒绝既可能表示 Novel，也可能表示 proposer miss。把二者串成一个二元
router 后，proposal error 全部表现为 certificate false negative。更大 classifier 不可能从没看到
的正确 anchor 中恢复信息。

### 根因 G：离线 certificate 不等于闭环 actionability（确定）

目标存在、anchor 正确、pose 稳定之后，residual 仍可能因方向、接管时机、控制器执行尺度而
伤害 native。项目里 behind gate、critic、endpoint、active-glance 已经证明简单 action proxy
会翻车。MRC 最多解决 existence/pose 的一部分，不能直接作为 SR proxy。

ImageGoal 近邻工作也普遍把 exploration、verification、exploitation 分开；例如
[IEVE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html)
采用“靠近再确认”，而不是把一次远距离匹配当最终行动依据。

---

## 4. 哪些解释已被排除

| 假设 | 证据 | 结论 |
|---|---|---|
| “几何完全没信号” | archived H/D+H 相对 DINO 有正 bootstrap CI；本机 pose dispersion AUC 高 | 排除 |
| “只是模型容量不够” | Phase-B、GLP、factorized F2/F8 多种模型重复同一 calibration/abstain 失败 | 基本排除 |
| “再扩 top-K 就行” | 正式 K1/K8 闭环 `18/40 vs 18/40, p=1`；风险随 K 快速上升 | 排除为主线 |
| “多跑几次会平均掉错误” | 三视角相关 `0.77–0.82`；natural stream 有 `63/63` 稳定假阳性 | 排除静态重复投票 |
| “RANSAC/LingBot 任一 expert 可单独判 Novel” | RANSAC Novel precision 低；MRC true positive overlap 可为 0，ambiguous 可高度自洽 | 排除 |
| “只差温度缩放” | train/dev threshold 大迁移；D+H 跨 artifact 反而低于 H | 不支持 |

---

## 5. 为什么现在不跑 full HPC

正式 smoke 已通过 ABI、三视角数量、因果配对和显存审计：24 sessions 用 A100 `28m13s`，约
`70.5 s/session`。线性估计 480 sessions 约 `9h24m`，且只有一个 top-1 row/session。

大样本能降低方差，但不能修复：

- no-match 在旧正证据中缺失；
- scene scale 不同；
- 三视角共享 bias；
- 40/155 positive sessions proposal 已 miss；
- label 与 pose/action utility 错位。

因此当前 Stage F 的 expected value 很低。这里不是因为“结果可能不好”而停止，而是因为它的
观测设计无法区分最关键的多个解释。先改判别实验，再决定是否值得正式采集。

---

## 6. 不用 HPC 的下一步：四个短归因实验

### T0：现有 artifact 方差与依赖审计——已完成，0 GPU

冻结收据就是本文第 2 节。它已经证明 scene nuisance、伪独立和 proposal ceiling，不再重复。

### T1：absolute-vs-relative certificate 对照——本机，约 20–40 分钟

固定 8–12 个已有 local sessions，不选择阈值：

1. 保留 deployment top-1 的 MRC score；
2. 每 session 加一个 label-blind null anchor，按同 scene、相近 DINO/texture、远 frame 选取；
3. 比较 absolute score 与 `top1 - null` contrast；
4. 只看方向：positive 是否更常有正 margin，strict no-match 是否接近零。

判别意义：若 contrast 明显稳定而 absolute 漂移，根因 B 成立，可把 MRC 降格为 scene-local
relative verifier；若 contrast 也失败，scene normalization 不能救它，停止 MRC selector。

### T2：frame-offset-vs-geometric-baseline 对照——本机，约 30–60 分钟

在同一小集上比较：

- 固定 `[-4,0,+4]`；
- 从半径内按 GT 仅用于实验设计选取最大 spatial baseline 的三帧；
- 按最大 yaw diversity 的三帧。

GT 只用于诊断“当前 clip 是否欠激励”，不进入最终方法。若 signal 随 baseline 大幅改善，后续
才值得设计可部署 VO/odometry-based keyframe selection；若不改善，不能再归因于 offset。

### T3：certificate-vs-pose-error-vs-action-utility 三标签审计——CPU + 已有闭环日志

把同一 row 分别用三种 target 评估：

1. surface support：teacher covis；
2. pose quality：GT position/direction/rotation error；
3. action utility：已有 residual/native paired gain/loss（只在合法已消费 train/analysis pool）。

当前 N=11 已显示 signal 对 pose error 的相关可能强于对 surface binary label。若扩大到已有本地
可用 rows 后仍成立，MRC 不应做 Novel/Revisit selector，而应做**已激活 Revisit 内部的 pose
quality / abstention expert**。

### T4：proposal decomposition——已完成，0 GPU

在 155 positive sessions 分三组报告：

```text
P1: top1 positive             115
P2: top1 wrong, deployment top2 有 positive     13
P3: deployment top2 无 positive                 27
```

只在 P1 评价 top-1 certificate；P2/P3 都是 proposal coverage 问题，P3 的 positive 只由
train-only teacher-forced row 暴露，部署 top-2 看不到。禁止把 P2/P3 算成 certificate false
negative。这会阻止后续再用一个总 accuracy 掩盖错误来源。

---

## 7. 架构决策

### 7.1 当前证据授权的系统

```text
benchmark known Novel   -> native NavDP
benchmark known Revisit -> raw-DINO top1 direct memory residual
unknown goal kind       -> native NavDP（研究扩展尚未解决）
```

known-Revisit direct 已有 fresh 160-episode 配对证据：`109/118` 对 geometry router `93/118`，
`+20/-4`，exact McNemar `p=0.00154`。这比继续为 unknown selector 牺牲 recall 更可靠。

### 7.2 MRC 若保留，正确角色是什么

不是：

```text
MRC absolute threshold -> Novel / Revisit
```

更可能是：

```text
known/high-belief Revisit
        -> raw DINO proposal
        -> MRC relative pose-quality certificate
        -> good pose: residual
        -> uncertain: native abstain
```

是否采用这一角色，必须先过 T1–T3；现在不预测它会成功。

### 7.3 如果仍想解决 unknown goal kind

文献和本项目共同支持的最小充分方向不是更多静态 feature，而是真正的新信息：

- target-domain/self-supervised calibration，例如用 pose-graph consistency 产生环境内 control
  ([Lajoie & Beltrame, 2022](https://arxiv.org/abs/2203.04446))；
- current-query motion + odometry 的 pose-chain integrity，而不是同一历史 pair 重投票
  ([Claxton et al., 2024](https://arxiv.org/abs/2407.08162))；
- 只在高价值歧义 hypothesis 上进行有限主动验证，而非每 episode 原地转圈。

这些都比再训一个 static MLP 有信息增量，但应在 known-Revisit 主线稳定后作为独立扩展。

---

## 8. 冻结决定

```yaml
full_hpc_mrc_stage_f: HOLD
reason:
  - existing positive evidence is candidate verification conditioned on Revisit
  - absolute features are scene dominated
  - three replays are strongly correlated
  - top1-only proposal ceiling is 115/155
  - certificate target is not yet aligned with pose/action utility

next_local:
  - T1 paired null-control contrast
  - T2 geometric-baseline ablation
  - T3 three-target attribution
  - T4 proposal decomposition (completed: 115 / 13 / 27)

stop_rule:
  - if T1 contrast and T2 baseline do not improve separation, retire MRC as selector
  - if MRC predicts pose quality but not support/action utility, keep only as Revisit pose-quality expert
  - do not submit full HPC until one local test demonstrates genuinely new decision information
```

这个决定不否定 geometry memory 的显著闭环结果，也不否定 LingBot 的 pose signal；它只否定
当前从“内部多视角稳定”直接跳到“unknown Novel/Revisit existence”的推理。
