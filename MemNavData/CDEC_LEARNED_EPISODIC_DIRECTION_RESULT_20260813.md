# CDEC learned episodic direction adapter：最终结果与停止决定

日期：2026-08-13（Asia/Shanghai）

## 一句话结论

本轮已经完成一个可部署形态的 learned episodic direction proposal adapter，并完成
scene-OOF、训练/运行时逐值一致、同 GPU/同 LingBot 进程 PnP 审计和独立 raw-CSV 复算。
结论不是“再训练就会赢”，而是：

> DINO patch learned ranker 确实学到了与 geometry 互补的 anchor 排序信号，但现有信号和
> 监督不足以替代显式几何；geometry-first、learned-on-reject 可以保持安全，却只在
> 349 次 fallback 中增加 1 个 certified-actionable session。它通过最低安全门，没有通过
> 足以支持完整闭环或论文主方法升级的效用门。

因此不读取 development/blind，不提交预备好的 consumed 20-scene/160-episode 长闭环。
这不是因结果难看而取消：冻结参考行为下 learned 最多影响 5 个 episode，而预注册的零损失、
双侧 exact McNemar `p<0.05` 至少需要 6 个 gain；即使 5 条全部救活，`p=0.0625`。

当前最强、最诚实的主方法仍是 **proof-carrying episodic bearing residual**：历史只提出
候选，显式几何证书授权 scale-free bearing，拒绝则原样回退 frozen NavDP。CDEC 是严格验证
过的互补 proposal/负结果，不应包装成已经带来 SR 增益的 learned 方法。

## 1. 问题分解

Revisit 不是一个单一分类问题，而是三个不同任务：

```text
ImageGoal + actual-online causal memory
                 |
          proposal：选哪个历史 anchor
                 |
        certificate：这个相对位姿是否可信
                 |
      control：如何把方向交给 frozen NavDP
```

- CDEC 只学习第一个任务：在冻结 causal DINO top-8 内做相对排序；
- SuperPoint + LightGlue + Fundamental-MAGSAC 和 LingBot-depth PnP 完成实例级几何求解；
- 原子 certificate 使用 PnP 内点、双边空间覆盖和重投影误差决定是否授权；
- 通过后只释放 scale-free bearing，经固定 `2.5 m` residual 输入 frozen NavDP；
- 拒绝或异常不等于 Novel，只表示当前历史证据不足，严格回退 native ImageGoal。

这里的 `geometry` 不是纯手工特征：SuperPoint/LightGlue 本身是预训练神经 matcher。真正
没有被 CDEC 替代的是显式 correspondence、epipolar/PnP 求解及其逐实例证书。

## 2. actual-online 时序可观测性门

目标在 expert A 上定义为 Revisit，但部署 memory 来自实际 online NavDP A 轨迹。正式训练或
方法解释前，独立重渲染并审计了 certificate 160 集自己的 online-A memory：

- conditional-B 分母中的 `120/120` 条均有实际 online-A `max covis >=0.20`；
- `115/120` 还有 `max covis >=0.50` 的强支持；
- 低于 `0.20` 的 11 条全部是 A failure，从未进入 B 条件分母；
- RTX 4090 与独立 RTX 5090 复现全部 160 goal 图、34,437 个历史帧哈希、每条 covis 曲线和
  所有汇总统计；结果逐字段一致。

因此本轮训练与资格审计面对的确实是 actual-online causal memory 中可观测的 Revisit，而
不是把 expert-only 视图误当成在线记忆。这个门不解决任意 Novel/Revisit 二分类。

正式 HPC original-path 复核的首次 job `15657882` 在渲染前因 manifest sidecar 相对路径
错误退出；没有方法输出。唯一修复是切换到 sidecar 所在目录再校验，15 项测试通过；替代 job
`15677956` 在 `ga014` 上以 `00:06:42`、exit `0:0` 完成。输出 SHA256 为
`d904aed865b451e5463ea3009f19b96459fc063ec5c313cce1b7296b5ee00ade`；其
`protocol/summary/stratified_outcomes/160 rows` 与 RTX 4090 报告逐字段完全一致，再次复现
`160/160` goal 和 `34,437/34,437` trace hashes。

## 3. learned 训练与 OOF 结果

数据边界：train40、480 sessions、每 session 冻结 top-8；development/blind 从未读取。

### 3.1 联合 `8 anchors + NULL`：失败

scene-OOF set student 把“选哪个 anchor”和“是否允许激活”塞进同一个 softmax：

- set student top-1 随 seed 为 `113--116/155`，geometry 为 `126/155`；
- 同时产生 `13--21/282` strict-negative 误激活；
- raw differentiable patch matcher top-1 为 `116/155`；低风险点只恢复 `50/155` 正 anchor，
  仍有 `4/282` strict-negative 误激活。

失败归因不是 patch 完全没有信息。anchor 排序只要求 session 内相对顺序；NULL 激活要求跨场景
绝对尺度校准。后者的场景漂移破坏了前者。因此禁止继续通过调 softmax、temperature 或网络
宽度来掩盖任务定义错误。

### 3.2 因子化 pairwise ranker：有信息但不能替代

冻结为 `5 outer scene folds / 4 inner scene folds` 的嵌套 scene-OOF pairwise ranking：

| 指标 | learned | geometry | raw DINO |
|---|---:|---:|---:|
| positive-session top-1 | **128/155** | 126/155 | 115/155 |

- learned vs geometry：`+10/-8`，exact McNemar `p=0.814529`；
- oracle union：`136/155`，只说明互补，不是部署收益；
- factual Revisit C：learned `69/75`、geometry `67/75`、union `72/75`；
- 每一行预测均来自不含该 scene 的 outer-fold 模型；runtime artifact 不含 teacher/task label。

所以模型学到了相对候选偏好，但 `+2` 不显著，不能声称替换 geometry。

## 4. 训练/运行时精确一致

部署 artifact：

`.diagnostics/certificate_distilled_compass_20260813/`
`factorized_pairwise_oof_fixedbatch_v2/cdec_pairwise_runtime_unapproved_v3.json`

SHA256：

`eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31`

关键审计：

- 完整 `480 sessions / 3,840 candidates`，runtime 与训练公式 score 最大误差 `0.0`，top-1
  `480/480` 一致；
- production DINO parity：72 张图、`4,718,592/4,718,592` 个 fp16 token、relation、score
  全部逐值一致，top-1 `8/8`；
- 固定复现 `37x37 -> 8x8` pooling、L2 normalization、fp16 token、float32 relation 和
  batch-size 16；
- artifact 保持 `deployment_approved=false`，只能由显式研究开关加载。

这排除了“离线模型和线上模型其实不是同一个”的解释。

## 5. 480-session 同进程 PnP 审计

job `15672166` 在一个 GPU、一个 LingBot 进程中依次测 geometry 与 CDEC proposal，耗时
`6:03:19`；这是 one-view full replay 资格测量，不是一次部署调用耗时。

主报告 SHA256：

`3f00f95c569c0c68175700c82c315aa4660af7050d707e80265560f47f486d39`

| proposal | teacher-positive top-1 | GT actionable | certificate accepted | certified-actionable | certificate FP |
|---|---:|---:|---:|---:|---:|
| geometry | 126 | 153 | 131 | **122** | 9 |
| CDEC | 128 | 135 | 122 | **115** | 7 |

直接用 CDEC 替换 geometry：

- certified-actionable paired `+1/-8`；
- exact McNemar `p=0.0390625`；
- 即 CDEC-only 显著更差，禁止替换。

CDEC-first 再回退 geometry 也不成立：`+1/-2`，且 certificate FP 从 9 增至 11。正确的
安全顺序只能是：

```text
geometry proposal
  -> certificate pass：直接使用，CDEC 不计算、不得覆盖
  -> reject：CDEC 提一个不同 anchor
       -> 同一个 certificate pass：learned takeover
       -> reject：native NavDP
```

该 geometry-first cascade 的结果：

- geometry primary accept：131；
- learned second-certificate invocation：349；
- 额外 certified-actionable：`+1`；loss：`0`；
- certificate FP：仍为 9，没有增加；
- 相同 anchor：117；重复 certificate 决策 `117/117` 一致；
- paired `+1/-0, p=1.0`。

它通过了冻结的最低安全门，但 `1/349` 是非常稀疏的互补性，不是有效性证明。

## 6. 独立 raw-CSV 复算

独立 verifier 不导入 pandas，也不导入任一 production summarizer，直接读取完整 960 行，
重新实现证书阈值、actionability、两种 cascade、paired McNemar 和 repeatability。

- job：`15677326`；
- 状态：`COMPLETED`，15 秒，exit `0:0`；
- `verified=true`；
- 输出 SHA256：
  `28c29703d6bb636d3c53cc9ec913327fa364987495ac23833db5f2cf6dab1fe8`；
- 官方 report 的 policies、paired、proposal identity 与 method gate 全部逐字段复现。

因此 `+1/-8`、geometry-first `+1/-0` 和 117 次 repeatability 不是主 summarizer 的实现偶然。

## 7. 为什么不运行预备好的 160-episode 闭环

冻结 consumed protocol 原计划比较 geometry certificate 与 geometry-first CDEC cascade。旧的
哈希锁定参考报告中：

- shared A success / conditional-B eligible：120；
- geometry certificate takeover：115；
- geometry reject、因而 CDEC 有权改变处理的 episode：仅 5；
- CDEC 不得覆盖其余 115 个 geometry pass。

在零 loss 的最有利情况下：

| gain | 双侧 exact McNemar p |
|---:|---:|
| 5 | `0.0625` |
| 6 | `0.03125` |

所以参考行为复现时，最大 5 个可改变 pair 小于显著性门所需的 6 个。scene-cluster CI 和“至少
两个 gain scenes”只会进一步收紧，不会修复这个上界。若新机器突然产生 6 个以上 geometry
reject，首先说明 baseline 漂移，需要归因，不能自动算成 CDEC 证据。

机器可读功效审计：

`.diagnostics/cdec_closed_loop_attainability_20260813/attainability_audit.json`

- report SHA256：
  `a9a695b9dc5a99b8bb82e34518d9cb2b9b0cbaa9d44617e7d1690c871d487920`；
- producer SHA256：
  `45823117c6933a1d1f852a68797504e0eb5820c250bdfb57b7244de36e2d164a`；
- frozen protocol SHA256：
  `1917a8407ad7a04f2fbbb2f421f7a0d302b33ae6c048bc0c05f2e2260c2dd9e5`；
- 决策：`do_not_run_statistically_uninformative_consumed_replay`；
- 同时哈希绑定 runtime artifact，确认其继续保持
  `deployment_approved=false`；
- development/blind read：false；episode trace read：false。

因此停止 6–8 小时闭环是预注册统计约束推出的资源决定，不是观察 SR 后的 post-hoc 停止。

## 8. 部署时间成本

现有 certified residual 是每个新 ImageGoal 做一次冷定位，不是每个 planning step 重算：

- 正式 160 集：uncached p50 `5.01 s`、p95 `26.83 s`；
- 本机真实 smoke：首次 `2.09 s`；
- 同一 goal 后续请求复用缓存绝对目标 pose，只按当前位置更新 bearing，smoke 为 `0.15 ms`；
- 全部 planning request（含 cache hit）的正式 p50 为约 `0.286 ms`。

CDEC 只在首次 geometry reject 后运行一次 patch proposal；若选择同一 anchor，直接复用拒绝
决定，禁止重复 PnP。当前没有 CDEC 闭环 latency 分布，不能声称它已经加速。可优化的是缓存
memory patch token、批处理 top-8 matcher 和减少冷启动；不能用错误定位换速度。

## 9. 论文与工程结论

可以写：

1. actual-online causal memory 的时序可观测性门通过；
2. scene-OOF patch relation 能提高候选排序的互补覆盖；
3. 把 anchor ranking 与 open-set authorization 合并会发生跨场景校准失败；
4. 显式 proof-carrying certificate 对安全 Revisit residual 是必要的；
5. 当前 learned fallback 安全但效用稀疏，不能替换 geometry，也没有闭环 SR claim。

不能写：

- CDEC 提高了 Revisit SR；
- learned adapter 已经替代 LightGlue/PnP；
- certificate reject 等于 Novel；
- `1/349` 足以授权 held-out/blind；
- 6 小时 collector 是部署时延。

若未来要继续 learned 路线，必须获得新的、未消费场景，并把监督改为 correspondence/PnP
actionability、pose error 和可拒绝不确定性，而不是继续拟合 co-visibility 或调整 NULL 阈值。
在没有新独立数据前，本分支冻结，不再读 development/blind，不在 consumed pool 调参。

## 10. 关键代码与产物

- 训练：`train_cdec_pairwise_ranker_oof.py`；
- differentiable matcher：`cdec_differentiable_matcher.py`；
- runtime：`cdec_pairwise_runtime.py`；
- runtime export/parity：`export_cdec_pairwise_runtime.py`、
  `audit_cdec_runtime_feature_parity.py`；
- same-process collector：`diag_lingbot_goal_loop_closure.py`；
- 主汇总与独立复算：`summarize_cdec_dual_proposal_certificate.py`、
  `independent_verify_cdec_dual_proposal_certificate.py`；
- runtime integration：`NavDP/baselines/memnav/policy_agent.py`、
  `eval_2leg_habitat.py`；
- 功效审计：`audit_cdec_closed_loop_attainability.py`；
- 未提交的闭环 harness 被保留作可复现工程产物，但不构成结果：
  `prepare_cdec_consumed_closed_loop.py`、`summarize_cdec_consumed_closed_loop.py`、
  `independent_verify_cdec_consumed_closed_loop.py`。

完整方法动机和运行前冻结协议见
`CERTIFICATE_DISTILLED_EPISODIC_COMPASS_PROTOCOL_20260813.md`；原 consumed 闭环门见
`CDEC_CONSUMED_CLOSED_LOOP_PROTOCOL_20260813.md`。
