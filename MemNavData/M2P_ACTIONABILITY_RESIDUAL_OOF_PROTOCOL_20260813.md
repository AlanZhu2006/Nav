# DINO-preserving actionability residual：scene-OOF 协议

日期：2026-08-13（Asia/Shanghai）

状态：**运行前冻结；train-only 架构诊断，不是方法结果。**

## 问题

在 factual Revisit-C 的 materialized production candidates 内，DINO top-1 的 bearing
actionability 是 `74/80`，candidate oracle 是 `79/80`。是否能用部署可见的 GCT 几何特征，
只修正这 5 条可恢复错误，而不破坏 DINO 已经正确的选择？

这与旧 CDEC 的差异必须固定：

- 旧 CDEC 学 co-visibility/certificate proxy；本实验直接学 controller-relevant target：
  `relative_position_direction_error_deg_center <=30 deg`；
- 旧模型用 learned score 替换基线；本实验固定
  `score = dino_cosine + alpha * residual`，并把 `alpha=0` 放进 nested selection，故可严格
  退化为 DINO；
- 只学习 session 内 ranking，不学习 Novel/Revisit、NULL、activation 或跨场景阈值；
- scene-grouped nested OOF；不读取 development/blind。

## 数据与特征

- pinned production rows SHA256：
  `193c29da7e2904061691361d5285d2211ff61b997619156f8b74262fde18237b`；
- train40 factual `goal_c_t0`：40 scenes、80 sessions、232 materialized candidates；
- residual 特征固定为：`depth_scale_raw`、`cloud_overlap_f1_center`、
  `anchor_goal_distance_norm_center`、`goal_refine_translation_norm_median`、
  `goal_refine_rotation_deg_median`、`goal_depth_confidence_mean`、
  `candidate_depth_confidence_mean`；
- 禁止输入 GT error、target xy、candidate label、scene/episode/role 或 future frame。

模型是标准化特征上的无截距 pairwise logistic regression；每个 mixed session 构造所有
actionable-minus-non-actionable pair 及其逆 pair。outer 5-fold、inner 4-fold 均按 scene 分组。
inner grid 固定：`C={0.01,0.1,1,10}`，`alpha={0,0.001,0.003,0.01,0.03,0.1}`。
inner objective 依次为：top-1 actionable 数最多、相对 DINO losses 最少、alpha 最小、C 最小。

## 冻结判据

只有同时满足以下条件，才允许扩展为更大 train-only selector study：

1. OOF top-1 至少 `77/80`（净提升至少 3，兑现至少 60% 的 5-session oracle headroom）；
2. 相对 DINO losses 不超过 1；
3. 5 个 outer fold 中至少 4 个由 inner OOF 选择 `alpha>0`；
4. gains 分布在至少 3 个 scene。

通过也不授权闭环、held-out read 或八小时长训；只授权检查更多 train-only candidates/roles
是否重复该规律。未通过则停止 learned reranker，把 DINO address + geometric certificate 保留
为当前方法。
