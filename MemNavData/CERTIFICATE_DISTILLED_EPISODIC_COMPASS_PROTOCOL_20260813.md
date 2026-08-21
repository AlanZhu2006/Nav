# Certificate-Distilled Episodic Compass（CDEC）冻结协议

日期：2026-08-13（CST）  
状态：**train40 可学习性与 scene-OOF 已完成首轮；联合 NULL 学习未通过，因子化
proposal 已进入同证书审计；development / blind 未读取。**

## 1. 研究问题

最新 closed-loop 结果已经证明一个强而慢的部署基线：在 shared-A 成功的 120 条
Revisit episode 上，`DINO top-8 -> LightGlue/Fundamental rank -> LingBot-depth PnP
certificate -> scale-free bearing -> frozen NavDP` 达到 `112/120 = 93.33%`，高于
raw-DINO direct 的 `106/120`、旧 geometry router 的 `91/120` 和 native 的 `27/120`。

本实验不再问“能否再拼一个 gate”，而问：

> 能否把慢几何 certificate 当作训练期 privileged teacher，蒸馏成一个快速、开放集、
> 带显式拒绝与方向不确定性的 episodic posterior；部署时只保留冻结视觉 backbone、因果
> memory map、小型学生和冻结 NavDP？

certificate 仍是不可替代的可靠 baseline。学生只有在 scene-disjoint OOF 中证明在相同
误激活风险下恢复更多正确 anchor，才有资格进入 closed loop。

## 2. 已冻结的事实与机会空间

train40 上已有完整、哈希绑定的 LightGlue/Fundamental 教师表：

- 40 scenes、480 sessions、每 session 固定 DINO top-8，共 3840 candidates；
- 155 positive sessions、282 strict no-match、43 ambiguous；
- top-8 对 positive session 的正确候选覆盖为 `143/155 = 92.26%`；
- 固定静态 **前证书 existence**（top-8 中任一候选满足 Fundamental 阈值）为
  `101/155` positive sessions covered、`2/282` strict-negative false activation；这不是
  “最终选中 anchor 正确”，也不是 PnP certificate；
- 按部署冻结的 LightGlue/Fundamental lexicographic rank 取 top-1 时，候选本身正确为
  `126/155`；再要求这个被选 top-1 通过静态前证书时，是 `86/155` 正确激活、3 个
  positive-session 错 anchor、`2/282` strict-negative 误激活；
- certificate 拒绝的 54 个 positive sessions 中，42 个仍在 top-8 含正确候选；其中
  DINO top-1 已能选对 31 个，LightGlue/Fundamental rank 能选对 34 个。

因此学生不能只模仿 certificate 的二值 accept/reject，否则会永久复制其 54 个假阴性。
真正的机会是：**用 certificate 教对应结构与安全拒绝，用 task-aligned train-only 真值
恢复 teacher 的保守召回损失。**

## 3. 冻结架构

方法名暂定 **Certificate-Distilled Episodic Compass (CDEC)**。

```text
ImageGoal patch tokens                 causal memory top-8 patch tokens
          \                                      /
           \---- shared low-rank projector -----/
                          |
            partial-OT / dustbin correspondence
                          |
              8 candidate relation tokens
                          |
          permutation-equivariant set reasoning
                          |
             softmax over {anchor_1...8, NULL}
                          |
       push candidate probabilities through memory bearings
                          |
        circular posterior q(theta), resultant, abstention
                          |
              fixed-radius residual -> frozen NavDP
```

### 3.1 部署输入白名单

- 当前 ImageGoal 与因果 memory candidate 的冻结 LingBot/DINO patch tokens；
- 已有 DINO cosine；
- candidate frame/index、有效 mask；
- LingBot 因果 memory map 给出的 candidate bearing（只用于把 anchor posterior 投影到
  圆周，不作为存在性标签）。

禁止输入：teacher co-visibility、candidate/session label、LightGlue matches、Fundamental
inliers/coverage、PnP inliers/pose/error、GT pose、goal role/variant、development/blind 统计。

### 3.2 输出不是独立分数加阈值

学生一次联合输出九类归一化后验：八个 anchor 与一个 `NULL`。正 session 允许多个正确
anchor，因此 task loss 是 set-valued NLL：

`L_task = -log(sum_{i in valid anchors} p_i)`；

strict no-match 或 shortlist 中无可执行正 anchor 时：

`L_task = -log(p_NULL)`。

这与旧 Phase-B 的“逐候选独立打分 + 跨场景绝对阈值”不同；后者已经观察到
train/dev 阈值 `0.397 -> 0.807` 的尺度漂移。CDEC 的 anchor 与 NULL 在同一个 session
内竞争概率质量。

候选的因果 bearing 为 `theta_i` 时，方向后验是 anchor 后验的圆周 push-forward，而不是
另一个 RGB-to-bearing 黑盒：

`q(theta) = sum_i p_i VM(theta; theta_i + delta_i, kappa_i)`。

第一阶段冻结 `delta_i = 0`，只验证对应/拒绝可学习性；只有完整 train-only bearing
teacher 覆盖通过审计后，才允许训练小角度 residual 与 concentration。

### 3.3 certificate 的训练期作用

certificate 是 privileged teacher，而不是部署输入或唯一真值：

1. **对应蒸馏**：学生 relation token 辅助预测 `log(1+matches/inliers)`、query/reference
   coverage 与 teacher 的 session 内排序；
2. **高精度锚定**：certificate 通过且 task label 一致的 anchor 提供高权重正监督；
3. **困难拒绝**：strict no-match 中高 DINO/高 teacher evidence 的候选作为 hard negatives；
4. **召回修正**：certificate 拒绝但 train-only task teacher 证明可执行的 42 个 session，
   仍按 set-valued task loss 学习，防止学生复制 teacher 假阴性。

主损失固定为：

`L = L_task + lambda_rank L_teacher_rank + lambda_evidence L_teacher_evidence`。

所有 lambda、宽度和训练 epoch 只能在 train scenes 的内层 scene-grouped folds 选择。

## 4. 与已经失败的方法的实质差异

- 不是旧 shared decoder：不预测 action，不改 NavDP；
- 不是旧 learned activation：没有独立候选 0.5 阈值，`NULL` 与 anchor 联合归一化；
- 不是 temporal latch：自然流 pilot 已证明 Novel 假 anchor 可稳定 63/63、Revisit 真 anchor
  可稳定 miss 35/35；简单时间一致性不再作为新信息；
- 不是把 RANSAC 换成一个 MLP：学生读取原始 patch 对应矩阵，certificate 只在训练期提供
  privileged geometric structure；
- 不是从图像凭空猜全局方向：方向来自真实 causal memory map，模型只推断哪段记忆可信及
  后验有多集中。

## 5. 预注册验证门

### G0：数据与因果性

- 480/480 sessions、3840/3840 candidates；
- scene 数为 40，future-frame consumption 为 0；
- 所有 image/token、teacher 表与 split 均记录 SHA256；
- development/blind 不可读取；ambiguous 只可做无标签蒸馏，不进入 task 指标。

### G1：无/低容量对应可观测性

先固定 DINO backbone，用非参数 mutual/partial-OT score 和极小线性 probe 做 scene-OOF。
若 raw patch correspondence 对 task anchor 或 certificate evidence 都不比 CLS cosine 提供
可重复的条件信息，则停止长训；不得靠扩大网络掩盖不可观测性。

### G2：学生 scene-grouped OOF

严格以 scene 为组做外层 OOF；超参数只在每个外层训练 fold 内选择。报告至少包括：

- positive-session correct-anchor 数；
- strict-no-match false activation 数；
- shortlist-miss 时错误强制激活数；
- exact safe action（正例选到任一有效 anchor，负例/不可恢复例选 NULL）；
- scene-macro、最差 scene、5 个训练 seed；
- 与 DINO、旧 geometry、static certificate 的相同风险 coverage 曲线。

进入 closed loop 的必要条件预先冻结为：

1. 在不超过 static ranked pre-certificate 的 strict-negative 风险量级（主点
   `<=2/282`，并报告 Wilson/scene bootstrap 不确定性）时，correct-anchor 必须严格超过
   `86/155`；同时单独报告相对 existence cover `101/155` 与无拒绝几何排序上限
   `126/155` 的差距，禁止把三者混为同一指标；或
2. 在旧 geometry 的固定风险点上，同时严格提高 correct-anchor 且不增加 false
   activation；
3. 改善必须跨多个 scene，且 5 seeds 方向一致。

单纯 pooled candidate AUC、训练准确率、teacher imitation loss 下降均不构成通过。

### G3：部署与闭环

G2 通过后才做：

1. 小规模 shadow：验证 runtime 只读白名单输入、延迟和圆周 posterior；
2. consumed pool 机制配对：native / raw direct / certified / CDEC；
3. 架构、权重、operating point 全冻结后，才进入 scene-disjoint fresh confirmation；
4. blind 最后一次性打开。

## 6. 今晚执行顺序

1. 绑定现有 480x8 certificate teacher 表及图像；
2. 提取冻结 DINO 8x8 patch token cache；
3. 先跑 G1 非参数与低容量 probe；
4. G1 通过才启动 CDEC 外层 scene-OOF 长训；
5. 生成完整 receipt、每 fold/seed 预测与诚实 Go/Stop 结论。

本协议不预言学生一定胜过 certificate。若 G1/G2 失败，结论将是：现有 certificate 的
增益依赖精细局部几何，冻结 DINO patch 无法可靠摊销；此时最优论文方法仍是 certified
bearing adapter，而不是再堆网络或继续调阈值。

## 7. 冻结后的实测结果与架构修正

### 7.1 联合 `8 anchors + NULL` 没有通过

严格 scene-OOF 的 set student 与 raw differentiable matcher 都没有超过冻结几何排序：

- set student 的无拒绝 top-1 随 seed 约为 `113--116/155`，低于 geometry
  `126/155`；联合 argmax 同时产生 `13--21/282` strict-negative 误激活；
- raw patch matcher 的无拒绝 top-1 为 `116/155`；在低风险 operating point 只恢复
  `50/155` 正 anchor，并有 `4/282` strict-negative 误激活；
- 因而 G2 **未通过**。这些结果不授权长训、closed loop 或 held-out 读取。

归因不是“patch 没有信息”。低容量 patch relation 的候选排序有信息；失败来自把两个
统计任务塞进同一个 softmax：`NULL` 要学习跨场景绝对证据尺度，而 anchor 排序只需学习
同一 session 内的相对偏好。前者的漂移反过来破坏了后者。

### 7.2 因子化后的 train-only OOF 结果

把模型限制为只做相对排序，并把激活权永久交给原子 PnP certificate。嵌套
`5 outer scene folds / 4 inner scene folds` 的 pairwise ranker 得到：

- learned top-1：`128/155`；geometry：`126/155`；DINO：`115/155`；
- learned vs geometry：`+10/-8`，exact McNemar `p=0.814529`；
- 两者正确集合的 oracle union 为 `136/155`，但这只是互补性诊断，不是可部署结果；
- 在 factual Revisit C 上，geometry 为 `67/75`、learned 为 `69/75`、union 为
  `72/75`；learned top-3 与 geometry 的候选覆盖达到 `73/75`；
- selection artifact 不含 task/teacher label，且每一行绑定其 held-out outer fold。

所以 learned ranker **不能替换** geometry；`+2` 很小且不显著。唯一被数据授权的下一步
是：让两者各提议一个候选，并对两者运行完全相同的一视图 LingBot-depth PnP certificate，
测量互补候选能否变成额外的安全通过，而不是直接把 oracle union 当收益。

### 7.3 当前最简架构

```text
causal memory top-8 + ImageGoal
          |
  geometry proposal first
  learned pairwise proposal only on reject
          |
  same atomic PnP certificate
   reject / certified anchor(s)
          |
  certified bearing posterior
          |
  fixed residual -> frozen NavDP
```

这里没有 learned revisit/novel 二分类器，也没有可迁移的绝对阈值：学习只负责在首个
geometry proposal 被拒绝后提出一个互补 anchor；它不得覆盖一个已经通过 PnP certificate
的 geometry anchor。几何只负责“这个具体相对位姿是否足够可信”，控制仍由冻结 NavDP
负责。若第二 proposal 不增加安全覆盖，就回退到单 geometry proposal，停止模型扩容。

### 7.4 同证书审计运行记录（运行前冻结内容）

- train-only、scene-OOF、无 development/blind、非 closed-loop；
- 单个 collector 在**同一 GPU、同一 LingBot 进程**内依次评估 geometry 与 learned
  proposal，避免跨机器 CUDA 差异破坏配对；
- collector：`15672166`；早期依赖 summary `15672172`、`15672692` 后由调度恢复提交
  `15677212` 产生最终 `report_repeatability_v2.json`；
- immutable bundle：
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/cdec_dual_proposal_certificate_becd3be95c7dbbfe`；
- source receipt SHA256：
  `b7969c2b7c0df6cc27368eee9bdec10d5d013cf6c976a7123a8877e5229013c1`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/cdec_dual_proposal_certificate_20260813/cdec_dual_sameprocess_nonarrayfix_20260813`。

首个提交 `15669915` 因只读 bundle 中 Python bytecode 缓存写入失败，在任何 PnP
measurement 前退出。修复为每个 task 使用本地临时 `PYTHONPYCACHEPREFIX`；该失败不含
方法结果。

第二个提交 `15670080` 在 PnP 前的精确 bundle pytest 发现漏打包
`phase_b_feature_schema.py`；第三个提交 `15670851` 的 bundle 闭包已通过，但 CDEC
preflight 正确发现收据把“静态 top-8 派生表 SHA”误与“上游原始 teacher SHA”比较。
随后待运行的双 task 提交 `15671629_[0-1]` 在零 GPU 秒时主动取消，因为跨 GPU
比较违反本项目的同机配对纪律。首次同进程提交 `15672004` 又在 22 秒的启动日志行
引用了已不存在的 `SLURM_ARRAY_TASK_ID`，在 preflight 和第一次 PnP measurement 前
退出。当前修订删除该数组变量，显式哈希绑定静态训练表，并逐行验证
`selection -> static top-8 -> raw teacher` 三段 identity/DINO 一致；本地 35 项与精确
staging bundle 30 项均通过。上述失败均没有产生可用的 PnP 或方法结果。

## 8. 论文架构与创新边界（运行前冻结）

聚焦检索显示，相邻工作分别覆盖了这个问题的不同切面：SLING 用神经特征与 PnP 做
[ImageGoal last-mile geometry](https://arxiv.org/abs/2211.11746)，IGL-Nav 用增量 3DGS 做
[ImageGoal 3D localization](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.html)，
VPR integrity work研究[定位结果的拒绝/验证](https://arxiv.org/abs/2407.08162)，而
Revisit Anything 研究[跨视角局部内容检索](https://arxiv.org/abs/2409.18049)；MemoNav 与
GOAT-Bench 则覆盖多目标/长期记忆。因而本项目不能把“检索”“PnP”“abstain”或“多目标
memory”任何单项声称为新颖。

可成立的方法核心必须是它们之间一个更窄、可证伪的接口：

> **proof-carrying episodic residual**：学习模块只能从因果历史中提出 anchor；每次记忆
> 干预必须携带一个由冻结单视图几何独立核验的相对位姿证书；证书只释放 scale-free
> bearing residual，拒绝则原样返回冻结 ImageGoal policy。

这不是 Novel/Revisit 分类，也不是把整个 controller 换成 learned policy。其三个需要共同
成立的贡献点是：

1. **检索与授权因子化**：scene-OOF 学习只提高 proposal coverage，独立 certificate
   保持干预精度；联合 `8+NULL` 的失败反而构成该分解的实验证据。
2. **最小控制接口**：历史不输出地图路径或 metric waypoint，只输出带证书的方向残差；
   frozen NavDP 保留局部避障与到达控制。
3. **跨目标可组合性**：actual-online 3-leg 中共享并逐字节审计 A 与 B prefix，只改变 C
   是否可读长期 A memory，从而测量第二次 Revisit 的因果价值，而不靠 joint SR 猜测。

若当前双 proposal PnP 审计没有额外 certified-actionable rescue，则 CDEC 不成为论文方法：
论文保持单 proposal 的 certified residual，并把 learned result 报为“proposal 学习有离线
信息但没有增加可认证覆盖”。若审计通过，它也只授权一个 **geometry-first、learned-on-
reject** 的 consumed-pool 闭环；在 fresh/held-out 前不得声称 learned 方法优于 certificate。

## 9. 可部署代码边界（2026-08-13 本地完成，尚未批准）

已把 all-train pairwise fit 从训练报告中导出为独立 runtime artifact：

- artifact：
  `.diagnostics/certificate_distilled_compass_20260813/factorized_pairwise_oof_fixedbatch_v2/cdec_pairwise_runtime_unapproved_v3.json`；
- SHA256：`eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31`；
- `deployment_approved=false`，approval state 明确绑定当前同进程双 proposal certificate gate；
- `CDECPairwiseRanker` 严格复现训练时 `37x37 -> 8x8` adaptive pooling、L2
  normalization、fp16 token cache 与 float32 relation quantization；context-free DINO
  固定补齐到训练时 batch size `16`；
- 在完整 `480 sessions / 3,840 candidates` 上，runtime 与训练公式的最大 score
  误差为 `0.0`，top-1 一致 `480/480`；
- production `MemNavAgent.lb.dino` parity 使用 SHA256 label-blind 选出的 8 sessions、72
  张输入，`4,718,592/4,718,592` 个 fp16 token、所有 relation、所有 score 均逐值一致，
  top-1 `8/8` 一致；receipt SHA256 为
  `cce0d5ac6f1159399fff9657602ebbcfa3eb4731f09511bc3203913dc4265306`；
- 当前相关回归为 `51 passed`；覆盖未批准 artifact 默认拒绝、分数精确复现、方向 posterior
  不得授权执行、accepted geometry 不得被覆盖、同 anchor 不得重复 PnP，以及 agent 侧
  causal JPEG shortlist 边界。

这一 parity audit 不是形式检查：原 v1 cache 的最后一批只有 3 张图，而在线 top-8 加 goal
是 9 张；BF16 GEMM 随 batch shape 改变舍入。未补齐时样本 token 的平均误差虽只有
`8.5e-5`，fp16 逐值一致率仅 `4.7%`。补齐到 16 后 token 全部一致；随后又发现 cache 的
float32 relation 边界会造成最大 `3.7e-5` score 漂移，v3 显式复现该 quantization 后才达到
零误差。失败的 v1/v2 artifact 与 parity receipt 被保留作审计历史，不得用于部署。

fixed-batch cache SHA256 为
`f5561b23bf5a42e3f4e203f3ac2dd222acddc9d3cebba93e30caa3cb57d34a21`。相对 v1，只有
排序后最后 3 张图变化，涉及 `6/3,840` candidate rows、`4/480` sessions、`1/40`
scene。完整 scene-OOF 已重跑；所有 480 个 top-1 proposal 身份零变化，核心数字仍为
`128/155`、learned vs geometry `+10/-8`、union `136/155`。因此当前 HPC PnP collector
测到的 proposal 身份仍精确适用于 v3，无需因数值契约修正重跑长时 PnP。

服务器只增加默认关闭的两个参数：`--cdec_pairwise_artifact` 与显式研究覆盖
`--cdec_pairwise_allow_unapproved`。启用 artifact 时仍必须同时启用 certified
relocalization。闭环顺序被代码结构固定为：

1. geometry proposal 先运行原证书；
2. 若接受，CDEC 不计算且不能覆盖；
3. 若拒绝，CDEC 才对同一个冻结 causal top-8 排序；
4. 若选中相同 anchor，复用第一次拒绝，禁止重复 PnP 挖随机性；
5. 若选中不同 anchor，运行同一个 PnP certificate；仍拒绝则原样回退 native。

softmax 输出在代码和 JSON 中都标为 `uncalibrated_pairwise_utility`，只可解释为 shortlist
内部的相对质量/方向质量，不得当作“这是 Revisit”的概率。候选 anchor bearing 的圆均值、
resultant 与 entropy 只是 posterior 诊断；certificate 精化出的目标 pose bearing 才可能进入
固定 2.5 m residual controller。

运行中的 2026-08-12 20:15 EDT 只读 checkpoint 曾为 `53/480 sessions`
（`106/960` paired seed rows）；当时未读取局部通过率。collector 随后完成 480 sessions，
最终结果见第 11 节。

## 10. 结果前冻结的独立复算

为避免主汇总实现本身决定结论，在 collector 完成前已经冻结第二套 raw-CSV verifier：

- 实现：`independent_verify_cdec_dual_proposal_certificate.py`；
- 不导入 pandas，也不导入两个 production summarizer；
- 直接读取完整 `960` 行，独立重算结构证书、`0.75 m` actionability、两种 cascade、
  paired McNemar、same-anchor repeatability 和 method gate；
- 另外要求 CSV scene universe 精确等于冻结 train40，官方 input SHA 与实际文件一致，且
  scope 明确为 same-GPU/same-process、train-only、非 closed-loop；
- 本地相关测试 `20 passed`；
- immutable verifier bundle：
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/independent_cdec_dual_verifier_90351f683937e4c6`；
- bundle receipt SHA256：
  `36fa2825a727fa3109375b350b481473ad72d7cb126376a7e154871f1c38423c`；
- 原依赖 verifier `15676473` 未运行；调度依赖恢复后的 replacement `15677326` 最终输出
  `independent_verification_v1.json`，结果见第 11 节。

该 verifier 只验证完整结果，不能读取 partial collector。只有官方报告与独立复算逐字段一致，
后续 gate 才有效。

## 11. 最终结果（2026-08-13）

collector `15672166` 已完成 480 sessions / 960 paired rows；官方
`report_repeatability_v2.json` SHA256 为
`3f00f95c569c0c68175700c82c315aa4660af7050d707e80265560f47f486d39`。
独立 verifier replacement job `15677326` 完成，输出 SHA256
`28c29703d6bb636d3c53cc9ec913327fa364987495ac23833db5f2cf6dab1fe8`，
`verified=true`，逐字段复现全部 policy、paired、identity 和 gate。

结果为：geometry certified-actionable `122/480`；CDEC-only `115/480`，配对
`+1/-8, p=0.0390625`，因此 CDEC 不得替换 geometry。geometry-first、CDEC-on-reject
为 `123/480`，相对 geometry `+1/-0`，false positive 仍为 9；但这是 349 次 fallback 中
仅 1 次安全 rescue，只通过最低安全门，不构成有效性证明。

预冻结的 consumed 160 闭环不提交。哈希绑定的参考闭环只有 5 个 geometry-reject episode，
而零损失双侧 exact McNemar `p<0.05` 至少需要 6 个 gain；最佳可能值仍为 `p=0.0625`。
机器可读审计 SHA256 为
`a9a695b9dc5a99b8bb82e34518d9cb2b9b0cbaa9d44617e7d1690c871d487920`。

最终完整解释、边界和代码索引见
`CDEC_LEARNED_EPISODIC_DIRECTION_RESULT_20260813.md`。
