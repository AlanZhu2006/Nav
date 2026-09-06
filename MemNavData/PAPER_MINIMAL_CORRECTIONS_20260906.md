# 本轮审计后的最小论文修改建议（2026-09-07 已按授权应用）

2026-09-07 更新：作者要求“先解决这几件事”后，以下三项已应用于活动论文
`/home/asus/Research/Memnav_Paper`。Table III 初次采用删列方案；作者随后要求保留 SPL，
现已改为 `SPL bounds` 上下界列，全部 SR 不变。最新决定见
[PAPER_SPL_RETAINED_20260907.md](PAPER_SPL_RETAINED_20260907.md)。
HM3D 四臂结果加入 Results 正文，不增设表格。标题、Abstract、Introduction 不变。
实施、终点追查和编译验证见 [PAPER_EVIDENCE_CLOSURE_20260907.md](PAPER_EVIDENCE_CLOSURE_20260907.md)。

以下保留 2026-09-06 的原始 Before/After 建议及风险说明；其中第 2 项删列建议已被
上述保留 SPL 的决定取代，不代表当前表格。

## 1. Analysis：区分 endpoint 误差与路线方向差

Location: `sec/7_analysis.tex`，Why the Memory Readout Remains Factorized 末段。

Before:

> A learned pose estimator retains 19/21 Revisit successes but releases 69/479
> accepted bearings with angular error above $90^\circ$.

After:

> A learned pose estimator retains 19/21 Revisit successes, but 20/479 accepted
> bearings deviate by more than $90^\circ$ from the goal's relative direction.

Reason: 69/479是与geodesic首段的夹角；相对真实目标方向的独立复算为20/479。
19/21成功数和原替换决定不变。

Risk of semantic change: 中等；这是统计含义的实质纠正，不作为自动语言润色处理。
没有必要再加一长段解释，可在补充材料/产物中保留完整两类角度统计。

## 2. Table III：保留SR，移除不可精确恢复的SPL列

Location: `tables/mono_factorial.tex`。

Before: 六列 `Depth input / History use / Novel SR / Revisit SR / Overall SR / SPL`，
SPL 为0.114、0.059、0.122、0.422、0.474。

After: 保留前五列及所有成功数，移除SPL列；不要用区间中点或截断最后一步的路径充当真值。

Reason: 210条旧记录缺末位姿，精确SPL不能恢复；当前单值沿用指令步长累加。
Table II已经按真实位姿修正，仍可报告路径效率。详见`PAPER_SPL_CORRECTION_20260906.md`。

Risk of semantic change: 中等；会减少一个指标，但不改变深度消融SR或结论。
替代选项是明确报告确定上下界，不称confidence interval；需要作者选择。

## 3. 新HM3D授权结果：紧凑补充，不另改主张

Location: `sec/6_results.tex` 的 Verification Regulates the Sparse Readout，
或Table IV增加独立HM3D小组。下面文字只涉及本轮已完成数据。

Suggested addition:

> A retrospective HM3D ablation shows the same intervention tradeoff but not
> higher overall success: CEC reaches 32/56 queries, compared with 35/56 for
> always-on retrieval and 30/56 for valid PnP alone. CEC leaves all 28 Novel
> executions unchanged; always-on retrieval gains seven and loses five against
> the base policy. CEC's overall differences from these two memory baselines
> are not significant ($p=0.581$ and $p=0.727$).

Reason: 避免只展示Final14方向有利的机制结果；保留原文“verification限制介入，而非
抬高Revisit ceiling”的主题。它没有反驳主表memory-on/off收益。

Risk of semantic change: 中等；新增已完成实验而非修辞变化。不要拼接旧Table I的
native6/28与本轮native7/28，不把retrospective ablation称为fresh confirmation。

## 4. 不建议做的稿件变动

- 不因上述两个非显著对照更换标题、摘要主题或新造MoE贡献。
- 不把4-query开发读出的低角误差搬到主表，或写成“已解决learned relocalizer”。
- 不为防守逐条增加免责声明；只修正具体数字语义、次要指标和缺失的完整结果。
- 不修改各表的SR分母、既定成功阈值与公开结果名称来让数字更有利。
