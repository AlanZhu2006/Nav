# HM3D 四臂授权消融：完整结果与独立复算

核验日期：2026-09-06（Asia/Shanghai）。这是先前 20/28 完成、8/28 因基础设施错误
中断的实验之完整续篇，不是另一批扩样。本文取代“仍在排队”的旧状态，但不覆盖旧日志。

2026-09-07 稿件同步：已按作者授权将完整四臂结果紧凑加入活动论文
`sec/6_results.tex` 的 sparse-readout 小节；未替换原 Table I 数字，未改标题、摘要或
Introduction。实施记录见 [PAPER_EVIDENCE_CLOSURE_20260907.md](PAPER_EVIDENCE_CLOSURE_20260907.md)。

## 1. 结论

**历史记忆的收益再次成立；严格 certificate 的总 SR 优势没有成立。**
CEC 限制了未经证实的历史介入，保持 Novel 的 native 行为；但在这批查询上，raw memory
的总成功数反而更高，差异不显著。不能把“少干预”直接解释为“更高成功率”。

## 2. 完整配对结果

28 histories、21 scenes，每个 history 一条 Novel、一条 Revisit；四臂共 224 条记录。
actual-mono A 历史、query-time monocular depth、同一 history 内四臂同机配对；role 不传给策略。

| 方法 | Novel / 28 | Revisit / 28 | 总计 / 56 | Novel 接管 / 28 | Revisit 接管 / 28 |
|---|---:|---:|---:|---:|---:|
| mono native | 7 | 8 | 15 | 0 | 0 |
| raw-DINO fixed bearing | 9 | 26 | 35 | 28 | 28 |
| 相同 top-8 proposal、只要求 PnP 可用 | 4 | 26 | 30 | 24 | 28 |
| strict CEC | 7 | 25 | 32 | 0 | 27 |

raw 并不是“只有 DINO”：它仍由几何读出生成 bearing，只是不经 certificate。
原 summary 的 `authorized_queries=0` 对 raw 仅表示没有 certificate 授权记录；
**不表示 raw 没有控制介入**。上述接管数是重新读取 `adapter_takeover_plans` 得到的。

| CEC 相对谁，总体 N=56 | 配对 + / − | 差值 | exact McNemar p | scene-cluster 95% CI |
|---|---:|---:|---:|---:|
| native | +19 / −2 | +30.36 pp | 0.00022125 | [+18.97,+42.19] pp |
| raw fixed | +5 / −8 | −5.36 pp | 0.58105 | [−18.00,+5.71] pp |
| PnP 可用即接管 | +5 / −3 | +3.57 pp | 0.72656 | [−7.41,+13.79] pp |

置信区间来自已封存 summary；本轮独立复算了 SR、W/L、exact p，未重新运行 bootstrap。

Revisit 中，CEC 对 native 为 +19/−2；对另外两个 memory 臂均为 +0/−1。
Novel 中，CEC 对 native 是 +0/−0，28 条终点、步数、成败与 native 完全一致，且原 verifier
确认 fully-rejected exact-native。raw 对 native 则是 **+7/−5**，净增 2 条。
这既不能证明 raw 获得了可靠 Novel 定位能力，也不能把其 7 条实际 gain 排除掉。

## 3. 它澄清了什么

1. memory-on 相对 memory-off 的大幅收益稳健，并非只有几条内部个案。
2. 固定候选的 PnP-witness 与 strict CEC，在全部 56 个初始 proposal 上相同；
   24 条 Novel 与 1 条 Revisit 的授权决策不同。因此后两臂确实在研究授权强度。
3. 严格授权将 Novel 的 24 次介入降到 0，但相对 PnP-witness 只得到 +5/−3；
   减少介入和增加 SR 是两个量，不能互换。
4. 相对 raw 的额外 SR 未确立，不能继续把 certificate 描述成总成功率最优模块。
5. 这是已知 native/CEC 结果之后设计的 **retrospective ablation**。
   样本不因结果选择，raw/witness 在提交前未知，但它仍不是 fresh confirmation。

与旧 Table I 的 HM3D `14/56→31/56`、完整 mono 系统的 `17/56→32/56` 分开报告。
前者是同一 Table-I population 的另一轮运行，后者是另一 query population；
不能跨运行拼接 native，也不能因为 CEC 总计恰好都为 32 就视为同一结果。

## 4. 唯一被 strict 拒绝的 Revisit

`023_uSKXQ5fFg6u_episode_0000`：CEC/native 失败，raw/witness 成功。
strict 最早拒绝原因为 `precheck_fundamental_reference_hull_coverage`。
只要求有限 PnP 的反事实读出得到：

- 54 个深度有效对应，9 个 PnP inliers；
- query inlier hull 0.007389，reference inlier hull 0.001958；
- RMSE 1.0215 px，有限 pose；
- witness 终距 0.9993 m，raw 0.9948 m，CEC 6.2033 m。

这说明至少一个能帮助导航的方向未通过目前的充分证据要求。
它不证明拒绝是代码错误，也不支持为了这一条降低 16-inlier/5%-coverage 门槛；
更不能用“导航最后成功”倒推其完整 6-DoF pose 正确。

## 5. 作业与验证

- `17010301_2`：COMPLETED，16m24s；
- `17010302_{3,6,7,10,13,18,21}`：全部 COMPLETED，约 14–29 分钟/配对块；
- `17010303`：COMPLETED，36s；summary 与 independent verifier 通过；
- 本轮读取时 `squeue -u yz11502` 为空，不再有这组任务运行/排队。

保留了原成功 20 个 history 的 completion 哈希；其余 8 个按修复协议整块重跑。
没有改样本、模型、候选、阈值、半径、预算，没有拼接失败块的部分臂。

远端根目录：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_authority_spectrum_repair_20260906
```

汇总与原 verifier 位于 `formal/POSTHOC/`。
summary SHA-256：`e62e8ac37170a52f674ff23ac9ded3e6bdbe91febac95f9ec41199e54bcbfb28`。
原 verifier 的 source digest：`aee47f8987a99c837a347b5b1710f4251fff48c1a10c03750f80653a5d13499d`。

本轮本机独立复算：

- `.diagnostics/overnight_project_audit_20260906/recount_authority.py`；
- 同目录 `authority_independent_recount.json`，`verified=true`；
- 同目录 `authority/` 保存原 112 个 CSV、28 个 completion、summary/verifier 与逐文件哈希。

224/224 记录的成功标签与 `<1 m` 末距一致；无 runtime failure，无 metric sensor depth
读入；A-prefix hash 检查通过，无 replay diffusion sampling；56 组配对 seed/geodesic 一致。
本轮未重新渲染这 224 条轨迹；exact-native 的完整动作链认证沿用原 verifier，另复核了
全部 Novel 的终点/步数/成败相等。

## 6. 对论文与后续工作的影响

应保留完整四臂结果，不以“与预期相反”将它搁置。最少可把它作为 Table IV 的 HM3D
扩展或正文一组平行验证，明确没有 strict-over-raw 的显著优势。
论文核心仍是 **因果单目 episodic memory 对冻结策略的可复用方向支持**；
certificate 是可选择干预的操作机制，不能独占全部 SR 增益的因果解释。

2026-09-06 审计阶段未自动修改论文。2026-09-07 获授权后采用正文平行报告方案，
Table IV 保留原 Final14 构造；标题、摘要、Introduction 保持不变。
