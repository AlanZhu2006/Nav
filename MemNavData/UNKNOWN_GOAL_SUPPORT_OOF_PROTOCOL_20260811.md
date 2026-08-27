# Unknown-goal Memory Support：Factorized OOF 协议

日期：2026-08-11（CST）  
状态：协议冻结；只允许读取 40 个 train scenes，不读取 development、20-scene consumed
closed-loop pool 或 blind。

## 1. 为什么先做这个

部署策略不会收到 Novel/Revisit 标签。2-leg 生成器知道 `A=Novel, B=Revisit` 只用于构造和
统计；`--hybrid_route phase` 与 `known_revisit_direct` 是 privileged mechanism ablation，
不能成为最终方法。

因此双 expert 的第一个必要条件不是 action routing，而是：

> 给定 goal image 与 causal episodic memory，系统能否自主判断 no-match，并在存在匹配时
> 选择可用 anchor？

现有 geometry router 用 DINO top-8 + RANSAC hard gate。它闭环可靠，但 train-only 审计已
证明 RANSAC pass 是条件正证据，reject/insufficient 不是 task negative。最新 VPR 文献也把
candidate-score distribution、match-level uncertainty 与 geometric verification 视为互补
证据，而不是互相替代。

## 2. 冻结假设

将 unknown-goal memory support 显式分成：

1. **Existence head**：只回答 `memory-supported` vs `strict no-match`；
2. **Conditional anchor head**：只在 existence 被接受时给 deployment top-2 排序；
3. RANSAC 是候选条件 likelihood；pass 为正证据，reject/insufficient 保持可学习的连续证据，
   不直接产生 Novel/Revisit 标签。

本阶段不生成动作、不训练 bearing、不调用 Habitat oracle、不改变 NavDP。

## 3. 数据与隔离

输入固定为：

- repaired Phase-B train rows：40 scenes、480 sessions；
- geometry evidence train rows：相同 40 scenes、480 sessions；
- join key：`(session_id, candidate_path)`；
- existence 输入只使用 `candidate_selection_origin=deployment_topk` 的两个候选；
- teacher-forced positive/hard-negative 只允许出现在 conditional ranker 的训练折，绝不进入
  held-out deployment candidate set。

Session teacher：

- `session_has_positive=True`：positive；
- `session_is_strict_no_match=True`：strict no-match；
- 其余 ambiguous：不参与 existence 拟合、阈值或主指标。

所有模型与 operating point 都按 scene 做 nested out-of-fold：outer fold 只负责读数；outer
train 内部再做 scene-grouped inner OOF 选择阈值。任何 held-out scene 标签都不得影响本折
阈值。

## 4. 三个公平系统

### H：当前 hard-geometry reference

- DINO top-8；
- 选择第一个 RANSAC hard-pass candidate；
- 无 pass 即 abstain；
- 它定义每个 outer-train split 的 strict-no-match risk budget。

### D：公平校准的 DINO

- existence score：deployment top-2 的 max DINO cosine；
- anchor：DINO top-1；
- inner-train 阈值在不超过 H strict-no-match false-activation rate 的约束下最大化 positive
  activation。

### F：factorized support

- existence：小型 standardized logistic head，输入候选集合的 DINO 分布、RANSAC 连续证据、
  pose-hypothesis agreement 与 deployment 可见质量量；
- anchor：positive-session pairwise ranker；输入 frozen Phase-B task features + continuous
  geometry features；
- existence threshold 使用与 D 完全相同的 nested risk constraint。

禁止用 candidate AUC 选择分支。candidate AUC 仅作诊断。

## 5. 主要指标

每个 seed 独立报告：

- 155 positive sessions 中：activated、correct-anchor activated、wrong-anchor activated；
- 281 strict-no-match sessions 中：false activation；
- `correct support decision = positive 且 active/anchor correct，或 strict 且 abstain`；
- positive/top-2-covered sessions 上 F ranker 相对 DINO top-1 的 paired wins/losses 与 exact
  McNemar；
- existence ROC-AUC、AP、Brier；
- scene-cluster bootstrap 的 F-H correct-support difference。

## 6. 预注册决策

使用三个 scene-fold seeds。只有三次都满足以下条件，Stage-1 才通过：

1. F 的 strict-no-match false activations 不高于 H；
2. F 的 correct-anchor activations 高于 H；
3. F 的 wrong-anchor activations不高于 H；
4. F 的 conditional ranker paired wins 大于 losses。

分支：

- 全部通过：冻结 Memory Support Expert，下一阶段采集 train-only Native-vs-Memory
  counterfactual action advantage；
- existence 通过但 ranker 未通过：保留 DINO anchor，只训练 existence；
- risk/coverage 未通过：停止 action expert，不进行长训；优先补多视角/match uncertainty 数据。

无论哪种结果，本实验都不授权闭环、development、blind 或论文性能声明。

## 7. 明确不做

- 不使用已知 A/B/C phase；
- 不用 development 选择 feature、threshold 或 seed；
- 不在 20-scene consumed pool 上调规则；
- 不恢复 `navdp_front_support_v1`；
- 不把 RANSAC reject 当负标签；
- 不训练 action arbiter，直到本门通过。
