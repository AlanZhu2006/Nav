# DINO-preserving actionability residual：scene-OOF 结果

日期：2026-08-13（Asia/Shanghai）

状态：**完成；未通过冻结继续门，不授权扩大训练。**

## 数据与问题

- train-only factual Revisit-C：40 scenes、80 sessions、232 materialized candidates；
- DINO top-1：`74/80` bearing error `<=30 deg`；
- candidate-set oracle：`79/80`，即仅 5 条 top-1 failure 在现有候选中可恢复，另 1 条候选集内
  无解；
- 80 sessions 中只有 24 个同时含 actionable/non-actionable candidate，pairwise 学习的有效
  决策样本远小于 232 rows；
- 输入与 nested scene-OOF 规则见
  `M2P_ACTIONABILITY_RESIDUAL_OOF_PROTOCOL_20260813.md`。

## 结果

| selector | CDF@15 | CDF@30 | CDF@45 | median error | p90 error |
|---|---:|---:|---:|---:|---:|
| frozen DINO top-1 | 64/80 | 74/80 | 78/80 | 4.05° | 22.97° |
| DINO + OOF bounded residual | 69/80 | **76/80** | 79/80 | 3.21° | 17.35° |
| materialized candidate oracle | 77/80 | 79/80 | 80/80 | 1.85° | 9.44° |

配对：

- residual vs DINO：`+2/-0`，风险差 `+2.5 pp`；
- exact McNemar：`p=0.5`；
- scene-cluster bootstrap 95% CI：`[0,+6.25] pp`；
- gains 来自 2 个 scenes；loss 为 0；
- 5/5 outer folds 的 inner OOF 都选择 `alpha>0`；
- selector 改变了 34/80 个 anchor；其中 23 个角度变小、11 个变大、46 个未改变。

两条 gain：

- `82sE5b5pLXE/episode_0001`：`42.96° -> 5.03°`；
- `XcA2TqTSSAj/episode_0001`：`163.71° -> 10.20°`。

## 冻结门

| 条件 | 结果 |
|---|---:|
| OOF 至少 77/80 | **失败：76/80** |
| losses <=1 | 通过：0 |
| 至少 4/5 folds 选择 alpha>0 | 通过：5/5 |
| gains 至少分布于 3 scenes | **失败：2 scenes** |

因此总门失败。这个结果支持“部署 GCT 特征含有 DINO 之外的弱增量排序信号”，但不支持
“继续扩大 residual selector 会形成方法级收益”。正向信号只有两条且不显著，不能用更长训练
把它包装成稳定提升。

## 决策

1. 不做八小时 residual/reranker 长训；
2. 不用 cloud overlap 直接替代 DINO：其 CDF@30 仅 `69/80`；
3. 当前最优方法结构仍是 DINO episodic address + geometry certificate + scale-free bearing +
   exact native fallback；
4. learned residual 仅保留为未来“有更多独立 mixed decisions 后”的备选，不进入当前论文
   方法主线；
5. 下一笔算力应花在新的 scene-disjoint、actual-online、role-free certificate 闭环确认，而不是
   在已基本饱和的 train candidate ranking 上继续调参。

## 复查文件

- report：`.diagnostics/m2p_actionability_residual_oof_train40_20260813/report.json`，SHA256
  `8d6de4a76c8d32f946a6a8b662bf5b07e5e6599e10a0b15fef3ef6dbdc77cab9`；
- session outcomes：
  `.diagnostics/m2p_actionability_residual_oof_train40_20260813/session_outcomes.csv`，SHA256
  `4c242038cb666ed7b9e97885156f07c76b08bf986fc547537087173a7f62cc71`；
- executable：`MemNavData/train_m2p_actionability_residual_oof.py`。
