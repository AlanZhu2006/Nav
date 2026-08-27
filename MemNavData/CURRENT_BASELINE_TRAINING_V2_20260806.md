# 基于当前可靠基线的新训练设计（V2）

日期：2026-08-06
代码范围：`/home/asus/Research/Nav-graph-blind`
状态：设计冻结前草案；尚未提交长训，也没有使用 final blind 场景调参。

## 1. 为什么不再延长 gatecurr600

`gatecurr600` 是从 `flowgate2600` 暖启动的历史端到端 continuation。它训练的
scalar gate、retrieval projection、shared diffusion decoder 与当前可靠系统的实际
推理路径已经不一致：当前系统由几何 router 决定是否启用 memory，并由冻结官方
NavDP 完成 ImageGoal/point-goal 局部控制。

因此下一次训练不再：

- 从 `gatecurr600` 或 `flowgate2600` 整体暖启动；
- 训练 complementary gate；
- 更新 NavDP ImageGoal encoder、diffusion decoder 或 critic；
- 用共享 decoder 同时学习 Novel 与 Revisit；
- 把旧 aux action theta 当作相机相对 yaw。

新模型只学习长期记忆中当前尚未解决的三个问题：

1. 目标图在 memory 中对应哪个 node；
2. memory 中是否根本不存在目标（显式 no-match）；
3. LingBot 给出的相对 SE(2) pose 是否可靠、应如何小幅修正。

## 2. 当前可靠基线 R0

冻结的 20-scene / 40-episode 2-leg 开发基线为：

```text
ImageGoal
  -> frozen DINO complete-history candidates
  -> temporal-NMS
  -> SIFT / essential-RANSAC verification
  -> LingBot per-episode metric relative pose
  -> frozen NavDP image + point-goal controller
```

结果：

| 指标 | Native NavDP | R0 geometry memory |
|---|---:|---:|
| Novel A SR | 31/40 | 31/40 |
| Revisit B SR given A | 4/31 | 19/31 |
| Joint SR | 4/40 | 19/40 |
| Novel false activation | N/A | 0/40 |

R0 是新训练的唯一闭环 reference。旧 checkpoint 的 W&B loss 不作为 baseline。

Reverse graph 在旧运行中得到 `25/40`，相对 direct 的 `19/40` 有六个 paired gain、
零 loss，但该运行没有严格匹配每次 diffusion request 的随机噪声。因此它是结构信号，
不是已经冻结的最终成绩。新 localizer 首先使用 direct point-goal 做因果 A/B；严格
graph A/B 通过后再组合。

## 3. 失败与训练目标的对应关系

31 条成功到达 Novel A、因而有资格执行 Revisit B 的路线中：

- 19 条成功；
- 7 条失败是 memory route 没有激活；
- 5 条失败是激活后未到达目标；
- 至少 3/5 active failure 使用了错误或很弱的 anchor；
- 其余约 2 条是长 direct point-goal/local-control 问题。

因此下一训练的首要目标是 localization recall、no-match calibration 和 pose tail，
而不是继续降低 action epsilon-MSE。

## 4. 新模型：LingGraph Localizer

### 4.1 推理结构

```text
frozen LingBot/DINO memory
        |
        |-- cheap temporal-diverse top-32 candidates
        |       -> patch/cost-volume candidate encoder
        |       -> top-4 shortlist
        |
        `-- LingBot-native top-4 loop factors
                - cloud overlap
                - goal-pose consistency/refinement
                - metric scale/depth confidence
                - raw relative SE(2)
                         |
                         v
                factorized set localizer
                - P(candidate_i valid)
                - P(anchor_i | usable shortlist)
                - P(global no-match)
                - delta SE(2)_i
                - covariance_i
                - loop utility_i
                         |
                 posterior stable for two plans
                         |
              reverse graph short subgoal
                         |
                 frozen official NavDP
```

RANSAC 在第一版只作为 uncertain fallback 和对照臂，不再作为模型输入的必需条件。
模型通过 blind/closed-loop 验收后，才能报告 learned-only 结果。

### 4.2 可训练与冻结部分

冻结：

- LingBot-Map long 权重；
- DINO/global/patch feature backbone；
- 官方 NavDP encoder、diffusion decoder、critic；
- 第一版 reverse graph 构图和固定 1.25 m subgoal 规则。

可训练：

- patch/correspondence candidate encoder；
- 两层、128-d、4-head 的 candidate set transformer；
- K 个 anchor logits 和一个显式 dustbin/no-match logit；
- LingBot raw pose 的小残差与异方差 covariance head；
- loop utility/calibration head。

模型不依赖旧 gatecurr decoder。运行时应进一步把 pose-only LingBot service 从完整
MemNav checkpoint 中拆出，使 checkpoint 血缘只包含：

```text
LingBot frozen weights + new localizer weights + official frozen NavDP weights
```

### 4.3 Pose 的定义

Pose head 学习的是相机/记忆图的几何相对位姿：

```text
(dx, dy, sin(delta_yaw), cos(delta_yaw))
```

并预测 translation covariance 与 yaw concentration。这里的 `delta_yaw` 是两个相机
坐标系之间的 wrapped relative yaw，不是导航 expert trajectory 的累计 action theta。
最终 pose 为：

```text
T_final = Exp(delta_T) * T_LingBot
```

这让网络只修 LingBot 的长尾残差，而不是从图像重新学习完整米制 pose。

## 5. 数据与 split

### 5.1 Cheap candidate 数据

复用已经修正 directional co-visibility 的 teacher CSV 和现有 patch/temporal cache：

- temporal-diverse top-32；
- 训练时把 gap `4/8/16` 作为确定性 candidate-set augmentation，部署默认固定为
  16 frame，与当前 R0 对齐；
- label 使用连续 co-visibility，不使用旧 gate label；
- 同场景重复纹理、DINO 高分但低共视候选作为 hard negative。

### 5.2 LingBot-native 数据

当前运行中的 deployment collection 已选定：

- 40 个 train scene、80 个 episode；
- 1,244 个 top-4 candidate seed；完成后才会形成可训练 CSV/JSON；
- strict positive、strict no-match 和 ambiguity 分离；
- cloud overlap、metric pose、direction/yaw error、depth confidence；
- predicted/target relative point-goal 和完整 goal pose。

该任务只使用 `allowed_role=train`，且当前 candidate gap 为 4，而 R0 部署 gap 为 16。
当前 rows 仍可用于学习 hard local correspondence，但不能假装与部署 candidate set
完全一致。任务完成后先根据原始 top-32 表计算 gap-4/gap-16 candidate key 的交并集，
只为 gap-16 缺失 key 补采，不重算已有 rows。随后用冻结代码在 10 个 development
scene 上采集一次 evaluation-role gap-16 数据；不能用 train rows 校准阈值。

当前 center-only collection 不提供邻帧 dispersion。第一版可以利用同一 session 的
多个 candidate 在公共 LingBot map frame 中预测同一个 goal pose，构造 set-level
consistency；只有当 covariance 学习仍不充分时，才对 hard cases 补采 `-4/0/+4`，不
重新计算所有样本。

### 5.3 场景角色

- 40 scene：训练；
- 10 scene：offline development/threshold calibration；
- 已检查过的 20-scene navigation benchmark：只做闭环 development A/B；
- `strict_graph_blind_20260806.json` 中 16 scene / 32 episode：最终一次性闭环 blind；
- 4 个 overlay-only `final_reserved` scene：只在 episode 完整性允许时做额外 router
  blind，不用于训练或阈值选择。

所有采样与 loss 都按 session/scene 加权，不能让长 episode 或候选数多的 scene 支配
训练。

## 6. 损失

总损失为：

```text
L = L_verify
  + L_rank
  + L_novel
  + L_pose_nll
  + 0.25 * L_utility
  + 0.10 * L_cycle
  + 0.10 * L_cal
```

### 6.1 Factorized localization loss

这里不再把三种不同问题硬塞进同一个 K+1 softmax：

- `L_verify` 对每个 strict positive/negative candidate 做 class-balanced BCE，学习绝对
  几何有效性；
- `L_rank` 只在 shortlist 内存在 positive 时计算 listwise loss，positive target 按
  co-visibility 归一化；
- `L_novel` 只使用全历史存在 positive 的 session 和 strict no-match session；全局
  ambiguous session 不参与，shortlist miss 也不被改标成 Novel。

最终可用匹配概率分解为：

```text
P(usable match) = P(global match) * max_i P(candidate_i valid)
```

这同时保留“memory 里有没有目标”“top-K 里有没有可靠候选”“候选中选哪个”三个可诊断
语义，避免 accuracy 很高但实际 anchor 仍错误的旧问题。

### 6.2 Top-1 consistency（当前作为指标，非额外 loss）

```text
L_top1 = relu(margin + max(score_negative) - max(score_positive))
```

它补上旧 set-InfoNCE loss 降低、live argmax 却不改善的问题。当前 Phase-B 先由
class-balanced `L_verify + L_rank` 解决这一点，并直接报告 conditional Recall@1/MRR；只有
正式 40-scene 数据仍出现“loss 降而 argmax 不变”时，才在训练消融中加入这个 hinge，不能
在 development 上看到结果后临时打开。

### 6.3 Pose uncertainty loss

`L_pose_nll` 只作用于 strict positive candidate，使用 wrapped SE(2) residual 的
heteroscedastic NLL；大误差使用 Huber residual，避免少量 metric/pose outlier 支配
训练。yaw 使用 sin/cos 或 wrapped residual，禁止直接对 `179°/-179°` 做普通 MSE。

### 6.4 Utility 与 cycle

训练 utility target 不使用 development navigation success。它由 train-only 几何质量
构成连续标签，例如 co-visibility、relative position error 和 yaw error的平滑组合。

`L_cycle` 把同一 session 各 candidate 的 predicted goal pose 变换到公共 LingBot map
frame，约束它们对同一 goal 的估计一致。它利用 LingBot graph，而不是把每个 pair 当作
独立分类样本。

### 6.5 Calibration

`L_cal` 使用 Brier/variance calibration。推理时 route confidence 为：

```text
P(match) * P(pose error < controller tolerance)
```

而不是固定 `gate > 0.5`。

## 7. 训练顺序

### Phase A：candidate proposal

- cheap top-32 数据；
- 只训练 rank + dustbin；
- scene-balanced batch；
- 先保证 candidate recall 和 no-match，再接 pose。

### Phase B：LingBot loop factor

- top-4 exact LingBot rows；
- 加入 pose residual、covariance、utility；
- hard positive/negative 与 strict no-match 1:1 session sampling；
- ambiguity 只进入 ranking/cycle。

### Phase C：joint calibration

- 联合微调全部新 head；
- 按 short/medium/long recall gap 分层采样；
- 3 个训练 seed 做 ensemble 或选择稳定单模型；
- temperature 和 activation risk 只在 10-scene development 上校准一次。

初始优化配置：AdamW，batch 32 sessions，LR `3e-4`，weight decay `1e-4`，gradient
clip 5，最多 200 epoch；每 5 epoch 做 scene-level validation，patience 25。超参数只在
train 内部 scene folds 选择，不能反复查看 navigation blind。

## 8. 必须做的消融

在同一 candidate set 上依次比较：

1. raw DINO；
2. patch/temporal set localizer；
3. LingBot loop factors only；
4. patch + LingBot K+1；
5. `4 + pose residual/covariance`；
6. learned-only 与 learned + RANSAC fallback；
7. direct point-goal 与 reverse graph。

这样能分别归因 candidate recall、loop verification、metric pose 和 graph planning，避免
再出现“很多模块一起改，但不知道 SR 为什么提高”。

## 9. Go / No-Go 门槛

### 9.1 Offline development

相对当前最好 development 结果：candidate top-1 `30/35`、joint localization
`101/118`、match AUC `0.927`、Brier `0.095`，新模型至少满足：

- candidate top-1 严格超过 `30/35`，或在新扩展集上有 scene-bootstrap 稳定正增益；
- joint localization 高于 `85.6%`；
- Brier 低于 `0.095`；
- 最差 scene 不出现大于 5 pp 的 accuracy 退化；
- pose median 与 p90 相对 raw LingBot 均改善，p90 至少下降 15%；
- predicted uncertainty 与真实 pose error 的 Spearman correlation 大于 0.4；
- 不确定样本回退 RANSAC 后不能破坏 geometry reference 的已成功样本。

若只降低 train loss、但没有超过这些 held-out 指标，禁止提交闭环 full。

### 9.2 20-scene closed loop development

必须使用 shared Goal-A trace 和逐 request deterministic diffusion seed：

- Novel A 必须保持与 R0 完全一致的 `31/40`；
- Novel false activation 必须保持 `0/40`；
- direct-controller A/B 先比较 R0 `19/31`；
- learned + fallback 至少达到 `22/31`，且最多回退一个 R0 success；
- 只有 localizer A/B 通过后，才与 strict-validated reverse graph 组合。

### 9.3 Final blind

冻结模型、阈值、candidate K、temporal gap、graph spacing 和 fallback 后，只运行一次
16-scene / 32-episode strict blind manifest。结果必须同时报告 Novel SR、conditional
Revisit SR、joint SR、SPL、activation、false activation、memory utilization、pose
uncertainty coverage、fallback rate 和延迟。

## 10. 运行资源与任务安排

这个新 head 很小，真正耗 GPU 的是 LingBot exact feature collection，不是优化器。
因此不应为了“长训”空占 H200 八小时：

2026-08-06 的实际采集结果修正了原计划：

- development job `15421650` 完成 310 rows / 78 sessions；修正 teacher 后按
  `scene + session_id + candidate_frame` 重新关联，310/310 的 co-visibility 和 label
  均未变化，因此昂贵特征可保留，只需更新 provenance；
- train job `15430677` 在 872/1244 rows、218 sessions、28 scenes 后 OOM。旧实现把
  每个 episode 的 CUDA cache 永久保存在 `cache_by_episode`，最终占用约 79 GB；
- 旧实现只在整个进程结束时写 CSV，因此失败目录为空，已计算的 70.1% 不构成可训练
  artifact，禁止从 stdout 冒充恢复结果。

修复后的 collector 必须：

1. 使用默认容量为 1 的 episode LRU，并在 eviction 时清除 aggregation、camera 和
   scale KV 后执行 CUDA cache 回收；
2. 使用 SQLite transaction 原子提交一个完整 session 的 rows 与 completion marker，
   因而中断最多损失当前 session；
3. 将精确 candidate/config/weight/teacher/split/source-commit signature 写入 checkpoint，
   resume 时任何不一致都 fail closed；
4. 每个 session 更新 progress JSON，最终 CSV 和 report 也使用原子替换；
5. 先运行跨两个 episode 的短 GPU smoke，再恢复 40-scene train collection；完整 train
   artifact 通过 scene/session/label/SHA 审计后才能开始正式训练。

如果 raw patch/cost-volume 需要重新抽取，可以把“特征抽取 + 训练 + offline report”放进
同一个最长 8 小时 pipeline，但必须阶段性落盘，并保证 GPU-heavy feature extraction
期间的利用率满足集群要求。

## 11. W&B 必须记录

- `train/dev set_loss`、top-1 hinge、dustbin loss；
- candidate Recall@1/MRR，按 scene 和 recall-gap 分组；
- no-match ROC-AUC/AP/Brier/ECE 与 risk-coverage；
- pose translation/yaw median、p90、NLL；
- covariance-error correlation；
- positive/no-match/ambiguous session 数；
- strict no-match false activation；
- 每个 artifact 的代码、split、teacher、feature、LingBot/NavDP weight SHA；
- trainable/frozen parameter 数和实际数据角色。

随机 batch 的单点 loss 只用于健康检查；是否保留 checkpoint 由固定 scene-disjoint
evaluation 决定。

## 12. 2026-08-07 的 fail-closed 采集与 Phase-B 训练链

当前已建立的前两级依赖为：

```text
15438554  two-episode bounded-cache smoke
    └─ afterok
15438709  40-scene / 311-session / 1244-seed exact collection
```

第二级使用 `--kill-on-invalid-dep=yes`；smoke 失败或取消时不会误跑八小时采集。
collector 固定在 commit `32145f3a3c1ae9f98d9fddbfa8e942e332c7356e`，训练代码使用
独立 worktree，不能为了更新 trainer 改动这个排队 checkout 的 HEAD。

正式训练前新增两层保护：

1. `audit_lingbot_native_localizer_artifact.py` 同时读取 SQLite、CSV、progress 和 collector
   report，要求它们对 40 scenes、311 sessions、1244 seeds/rows 完全一致；每个 selected
   candidate 再与 corrected teacher 一对一关联，并核对 split、teacher、source commit、
   label、session flags 和全部训练输入的有限性；
2. `train_lingbot_native_localizer.py --preflight-only` 在真实 artifact 上执行一次所有 head
   的 backward，要求 encoder、rank、no-match、pose mean 和 pose covariance 都有有限非零
   梯度，且不读取 development 指标。

只有这两层都通过，才运行 CPU Phase-B 正式训练：

- 显式输入白名单只包含 DINO cosine、LingBot overlap/refinement/depth confidence、metric
  scale/source 和 LingBot predicted `(forward,lateral)`；
- `teacher_covis`、`label`、target pose 以及所有 GT error 列都禁止作为模型输入；
- localization 被明确分成三项：candidate absolute verification、只在
  selected-positive sessions 上计算的 listwise rank、以及只用全局 strict-positive /
  strict-no-match session 监督的 Novel head；“memory 有 positive 但 top-4 漏掉”的
  shortlist miss 只作为安全拒绝，绝不能错误改标成 Novel，ambiguity 也不进入 Novel BCE；
- 推理时先由 `P(global match) * max P(candidate valid)` 决定是否存在可用 anchor，再在
  candidate 内归一化排序。这保留了 RANSAC 的“绝对验证”作用，同时不把它固化为最终
  必需模块；
- pose head 只在 corrected-teacher positive 上学习 translation residual mean 与 diagonal
  covariance，并对 batch 内最差 20% translation error 加 CVaR tail-risk 项；当前 artifact
  没有完整 target yaw，因此不能把 translation-only 结果冒充 yaw head；
- pose residual 从零初始化，并只在 train-internal validation 上从 `[0, 1]` 选择 gain；
  若任何 learned correction 不能降低 translation p90，`gain=0` 会原样保留 raw LingBot
  pose。未采用的 residual mean 会并入预测风险，避免“pose 不修了、uncertainty 也失真”；
- stopping epoch 和 threshold 只由 40 个 train scenes 内部的确定性 scene split 决定；
  split 同时保证 core/tune 都含 selected positive 和 strict no-match，否则 fail closed；
  十个 development scenes 在冻结后三 seed ensemble 上评估一次；
- 输出仍标记 `deployment_approved=false`，只有达到第 9 节门槛才允许闭环 learned+fallback
  评测。

提交 `slurm_train_lingbot_native_localizer.sbatch` 前必须逐项通过依赖检查：trainer commit
和 task files clean、corrected teacher SHA、split SHA、immutable development CSV SHA、
collector source commit、artifact audit identity，以及 online W&B 的 API key。任何一项不符
都必须退出，不能降级为继续训练。
