# REVISIT：RANSAC Expert 资格审计

日期：2026-08-11（CST）  
状态：协议冻结；只用 train scenes；等待 HPC 正式运行。

## 1. 为什么做这个，而不是再训一个 adapter

当前最强可部署 REVISIT baseline 是同机同进程复测的 geometry router：Novel A
`26/40`，在 A 成功条件下 Revisit B 从 native `3/26` 提升到 `20/26`，配对
`+17/-0`，exact McNemar `p=1.53e-5`。固定 2.5 m bearing 与 metric waypoint 同为
`20/26`，base PointGoal、mixed 与 X-NavDP 分别为 `20/26、20/26、21/26`。因此当前主要
空间不在 controller 或距离回归，而在 memory activation / anchor inference。

R0 的 6 条 residual 中：

- 4 条没有激活；事后 task co-visibility 显示四条都存在有效 memory；
- 其中 2 条被 DINO floor/prefilter 挡住，2 条有高 DINO 候选但 SIFT/RANSAC 全拒绝；
- 另 2 条已激活，才属于 controller residual。

现有 RANSAC 逻辑把“没有足够局部特征”和“有充分几何证据证明不相容”都编码成 0。
本实验只检验这个语义是否错误，不预设 learned fusion 必然有效。

## 2. 冻结问题

对同一个 causal candidate，令：

- `Y_task`：goal-surface co-visibility teacher 的 positive / negative / ignore；
- `E_app`：冻结 LingBot DINO cosine；
- `E_geo`：线上同构 SIFT ratio-test + essential-RANSAC 证据。

回答三个问题：

1. RANSAC stable support 是否具有足够高的 task precision？
2. RANSAC reject 中是否混入跨多个 scene 的 task positives，因而不能作为 hard veto？
3. 在 scene-grouped OOF 下，`E_geo` 是否提供超出 DINO 的 episode 内排序信息？

第 1+2 成立只授权 **support / unknown** 语义；第 3 也成立，才授权 RANSAC 成为真正的
ranking expert。二者不可混为一谈。

## 3. 数据与隔离

唯一输入是已哈希审计的 causal teacher：

- 全集：50 scenes、600 sessions、17,845 candidate rows；
- 本次只读 `split_role=train`：40 scenes、480 sessions、14,172 rows；
- labels：1,509 positive、11,279 negative、1,384 ignore；
- development 10 scenes 不读取、不报告、不选阈值；blind 不触碰；
- candidate universe 是 manifest 冻结的 causal DINO shortlist，不补未来帧。

原图通过只读 overlay 提供；每个 query/candidate 在计算前按 teacher 中的 SHA256 验证。

## 4. Geometry evidence

每个候选严格复现线上 verifier：

- SIFT `nfeatures=4000`；
- Lowe ratio `0.75`；
- essential matrix：RANSAC confidence `0.999`、threshold `1.5 px`；
- baseline pass：matches `>=20`、recoverPose inliers `>=12`、ratio `>=0.5`。

同一组 correspondence 额外运行 5 个由 `(session, candidate, repeat)` 哈希固定的 RANSAC
seed。它不是参数搜索，只用于测量随机估计稳定性。输出保留 canonical pass、pass rate、
inlier min/median/max/std、essential/pose recovery rate 与五态 missingness：

```text
insufficient_features
insufficient_matches
model_unavailable
unstable
stable_support / estimable_reject
```

任务按 session 原子写 shard，可安全 resume；最终 CSV 必须 exact-cover 全部选中 rows。

## 5. 只在 train 内做 scene-grouped OOF

透明线性 probe，不训练视觉 backbone：

- appearance：DINO cosine；
- geometry：matches/inliers/ratio、RANSAC availability/stability、两图 keypoint counts；
- fusion：appearance + geometry。

五折按 scene 分组。每个 test scene 的分数只来自其余 scenes 拟合的模型；每个 session 在
训练 loss 中总权重相同；ignore 不进 loss，但仍保留在部署式 candidate ranking 中。

主要单位不是 pooled candidate accuracy，而是：

- episode 内 positive-vs-negative concordance；
- positive-session top-1 与 MRR；
- strict no-match 的 session existence ranking；
- fusion vs DINO 的配对 wins/losses、exact McNemar、scene-cluster bootstrap CI；
- hard gate 的 positive recall 与 strict no-match false activation。

不从这些 OOF 分数冻结部署阈值，也不声称闭环 SR。

## 6. 预注册判据

### Gate A：RANSAC 可作为 support / unknown expert

同时满足：

- `stable_support` 对 task extreme labels 的 precision `>=0.90`；
- canonical hard rejection 漏掉 task positive，且这些漏检分布在至少 5 个 train scenes。

通过后的含义：强 RANSAC 是正证据；低纹理/不稳定/拒绝不能自动当成负证据。下一版 router
应让 task expert 主判存在性，RANSAC 提供 support likelihood，缺证据时取 neutral。

### Gate B：RANSAC 可作为第二 ranking expert

同时满足：

- geometry-only OOF 的 episode 内 session-macro AUC `>0.5`；
- fusion top-1 相对 DINO 净增益，且 exact McNemar `p<0.05`。

Gate B 很严格；不通过就禁止把 RANSAC 连续分数塞进 learned reranker。它仍可通过 Gate A
成为高精度 support expert。

### 任何 Gate 都不授权部署

通过后仍需冻结一个候选/session 后验和 abstain 规则，再做 consumed-pool 三臂同进程闭环：
native、现有 hard geometry baseline、new expert fusion。只有闭环 `+gain/-loss` 才决定是否
进入 fresh non-blind confirmation。

## 7. 可能结论与对应架构

| 结果 | 结论 | 下一步 |
|---|---|---|
| A、B 都过 | geometry 有独立排序价值 | task expert × geometry likelihood，显式 dustbin |
| 只过 A | 最可能，也最符合当前 residual | task expert 主判；RANSAC=positive support / otherwise unknown |
| A、B 都不过 | RANSAC 只是历史 heuristic | 保留最强 baseline；转向 pose-chain / temporal consistency 非外观证据 |

这个实验的价值不取决于得到正结果：它会阻止我们第六次把漂亮的 pooled 指标误当成可部署
episode 内能力。
