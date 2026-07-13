# MemNav v2 修改与结果记录

> 更新时间：2026-07-14
> 分支：`feat/align-lg-dataloader`
> 本文对应代码基线：`ed7cf63`
> HPC 动态状态快照：2026-07-14
> 最新训练决策与作业清单见 [`MEMNAV_TRAINING_PLAN_2026-07-14.md`](MEMNAV_TRAINING_PLAN_2026-07-14.md)

## 1. 结论摘要

这轮工作的核心不是替换 planner，也不是重新生成一条 NavDP trajectory，而是把已有的 MP3D 多段 revisit 数据正确接入 MemNav，修复“弱 revisit 被错误标成 novel/NULL”的训练标签语义，并补齐可复现的 checkpoint 离线诊断。

目前可以确认：

- scenes 0-53 的 `pt1` 原始数据包 1944/1944 个 episode 结构完整；25 个长 episode 的 4096-frame cache 正在由 job `13500804` 补算。
- revisit/novel 标签语义、cache 静默漏数、Li Guo DataLoader 采样和 R2R scene-disjoint split 已完成，并通过 25 个针对性单元测试。
- 旧 E3 在 2746-sample 训练集诊断上显著优于 `lg154 checkpoint-956`，但该结果混合了全部 54 个 scene，只证明拟合，不证明泛化。
- E3 的 aux pose MSE 几乎不变，且加权后占离线 total loss 约 88.7%；已经提交 aux 0.5 对 0.05 的严格配对训练。
- 新训练仅使用 pt1 中的 38 个 R2R train scene；7 个 val-unseen scene 用于模型选择，9 个 test scene 暂不使用。
- 当前仍没有 Habitat 闭环 SR、SPL、collision rate，不能据离线 loss 判断真实导航已经提高。

## 2. 当前系统到底在做什么

### 2.1 数据轨迹

MP3D revisit 数据由 Habitat 中的多段 episode 生成：

- two-leg：`start -> A -> revisit B`
- three-leg：`start -> A -> novel B -> revisit C`

每一段先由 Habitat geodesic 给出可行路径，再经过 ElasticBands、pure-pursuit 和基于 navmesh 的 dense safety gate 处理。因此数据里的轨迹不是 A* 在线 planner 的输出，也不是从 NavDP 仓库直接拿来的 trajectory。

图上出现不自然的弧线或贴墙路径，可能来自 geodesic、平滑、跟踪器和采样离散化的组合；只看离线轨迹图不能直接断定是当前学习 planner 的问题。是否真实撞墙必须通过 Habitat 闭环 rollout 的 collision 指标验证。

### 2.2 模型结构

当前 MemNav 包含：

1. 冻结的 LingBot-Map/GCT 视觉与流式记忆前端。
2. 可训练的 DINO CLS retrieval head，在历史帧与一个 NULL 槽之间建模。
3. `revisit_gate = 1 - P(NULL)`，用于软组合 revisit 和 novel 分支。
4. 基于 NavDP 代码和动作标签形式的 DDPM 局部 action decoder。

我们复用的是 NavDP 风格的 action label、diffusion decoder 和训练框架，不是 NavDP 自带的轨迹，也没有把 A* 作为当前 policy。`lg_e1` 是旧 MemNav checkpoint，不是官方 NavDP baseline。

### 2.3 scale frame、anchor KV 和语义 anchor

这里有三个容易混淆的概念：

- 前 8 帧在逻辑上是 LingBot KV cache 的 **scale frames**，需要使用完整 K/V；当前 pt1 cache 使用 `--skip_scale`，因此不把这部分 K/V 写入磁盘，而是在训练读取样本时由原始前 8 帧重算。
- 后续历史帧每帧只保存 6 个 special-token K/V，文件中叫 **anchor_k/anchor_v**；这里的 anchor 只是压缩缓存类型。
- 它们不是 HLoc 意义上的稀疏重定位 keyframe，也不是“轨迹前 8 帧全部叫 anchor frame”。

修复后的 MP3D cache 参数为 `num_scale=8`、`window=32`、`max_frame_num=4096`。旧的短轨迹 cache 可以继续使用，因为 3D-RoPE 是解析生成的，扩展 table 不会改变已有索引。可执行 retrieval 候选从

```text
anchor_margin = num_scale + window - 1 = 8 + 32 - 1 = 39
```

开始，因为更早的位置无法构造完整的 32 帧重计算窗口。

### 2.4 当前 DataLoader 采样

默认采样已与 Li Guo 的 `3af2c8d data_loader` 对齐：每次访问都在目标 leg 内随机选择 `k`，保留 `goal_slack=4`，并加入使用 14/83 帧阈值的动态 Goal A 样本。唯一有意的标签差异是继续以 `goal.kind` 为语义真值，不把 weak revisit 改成 novel。

为复现旧 E3，仍可设置 `MEMNAV_SAMPLING_MODE=fixed_switch` 和 `MEMNAV_ADD_GOAL_A=0`；默认训练使用 `random_leg` 和 Goal A。

## 3. 数据检查结果

### 3.1 数据位置与范围

| 内容 | 路径或状态 |
|---|---|
| pt1，scenes 0-53 | `/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf` |
| pt2，scenes 54-89 | `/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt2.sqf` |
| pt1 容器内 trajectory root | `/mp3d_revisit_v0/vln_n1/traj_data` |
| LingBot feature cache | `/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data` |

`.sqf` 是只读 SquashFS/Apptainer overlay，不是 SQL 数据库。当前训练只使用 pt1；pt2 尚无同等完整的 LingBot cache，因此还没有进入训练。

### 3.2 审计结果

- pt1 共 1944/1944 个 episode 通过结构检查。
- feature tree 审计到 1919 对 cache；25 个超过 2048 帧的 episode 与 25 个空 cache 目录完全一一对应，最长轨迹为 3954 帧。
- 语义修复后的显式 covis goal 共 2795 个（1587 revisit、1208 novel）；现有 cache 覆盖其中 2746 个：
  - revisit：1563
  - novel：1183
  - 因没有任何 `covis >= 0.5` 的强正例而跳过的 weak revisit：357 个 goal record
- 对齐 Li Guo 采样后还会加入 1411 个动态 Goal A，因此完整数据为 `2795 + 1411 = 4206` 个样本；现有 cache 只能覆盖 4141 个，25 个长 episode 共缺 65 个样本（49 covis goal + 16 Goal A）。
- 若完全使用 Li Guo 原始 `null_pos = not pos.any()` 语义，则会得到 4563 个样本，其中多出的 357 个正是被误改成 novel 的 weak revisit；这些不会重新加入。
- feature tree 约 791 GiB，所以训练直接在 HPC 读取，不适合整体复制到本机。

因此，“原始数据包损坏”目前没有证据；问题位于 feature 预计算和 loader 覆盖检查。代码已阻止再次静默漏数，但在补算 25 个长 episode 之前，feature tree 仍然不完整。

## 4. 本轮代码修改

### 4.1 接入真实 MP3D 多段数据（`8106284`）

- 将 dataset 从旧的合成 `(k, k_goal)` 采样改为读取 `meta/gen_meta.json` 中真实的多段 goal。
- 一个训练样本现在是 `(episode, goal j)`，action label 使用该 goal 对应的下一段正向轨迹。
- goal 使用单独渲染的 `goal_j.jpg`，而不是把某个 trajectory frame 当 goal；goal CLS 在前向时由冻结 DINO 计算。
- 支持“只读 frame overlay + 独立可写 feature root”，避免要求 cache 和图像放在同一目录。
- 强制训练参数与 cache 预计算参数一致：`window=32`、`num_scale=8`、`max_frame_num=2048`。
- 扩展 LingBot camera head 的 3D-RoPE table 到 2048，修复长 three-leg episode 在 1024 帧以后维度不匹配的问题。
- 增加 pt1 feature 预计算和 H100/A100/H200 训练 sbatch 脚本，并在启动前检查 overlay、cache、权重、Python 和 CUDA 环境。

主要文件：

- [`InternNav/internnav/dataset/memnav_dataset_lerobot.py`](InternNav/internnav/dataset/memnav_dataset_lerobot.py)
- [`InternNav/internnav/model/basemodel/memnav/lingbot_stream.py`](InternNav/internnav/model/basemodel/memnav/lingbot_stream.py)
- [`InternNav/internnav/model/basemodel/memnav/memnav_policy.py`](InternNav/internnav/model/basemodel/memnav/memnav_policy.py)
- [`InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch`](InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch)

### 4.2 修复 revisit/novel 标签语义（`9eccd5e`）

旧代码的关键问题是：

```python
null_pos = not pos_mask.any()
```

这会把 metadata 明确标记为 revisit、但没有任何 frame 达到 `covis >= 0.5` 的样本静默改成 novel/NULL。模型收到的监督与 episode 真实语义相反，aux pose 监督也可能随之错位。

新逻辑以 `goal.kind` 为语义真值：

- `revisit` 且存在强正例：保留样本，真实 frame 是正例，NULL 是负例。
- `revisit` 但没有强正例：标为 `weak_revisit` 并跳过，不再伪装成 novel。
- `novel`：只有 NULL 是正例；如果反而存在强真实匹配，则拒绝该冲突样本。
- trainer 断言 `null_pos == ~is_revisit`，防止以后再次发生隐式改标。
- aux pose 只在 metadata revisit 样本上计算。

retrieval mask 定义为：

- positive：结构候选中 `covis >= 0.5`
- negative：结构候选中 `covis <= 0.1`
- ignore/gray：`0.1 < covis < 0.5`
- 前 39 帧和 padding：结构上不可执行，完全 mask

gray frame 目前不进入 retrieval loss，但仍可能在推理时成为 top-real 候选。这是已知问题，不是已经解决的部分。

主要实现与测试：

- [`InternNav/internnav/dataset/memnav_labels.py`](InternNav/internnav/dataset/memnav_labels.py)
- [`InternNav/internnav/trainer/memnav_trainer.py`](InternNav/internnav/trainer/memnav_trainer.py)
- [`InternNav/tests/unit_test/test_memnav_labels.py`](InternNav/tests/unit_test/test_memnav_labels.py)

### 4.3 增加可控 smoke/full training（`3599028`）

- 增加 `MAX_STEPS`，可先做 10-step smoke test，再提交长任务。
- 将 batch size、epoch、worker、日志后端和训练名改为环境变量可覆盖。
- 保留逐 epoch checkpoint，便于 E1/E2/E3 对比，而不需要重复训练。

### 4.4 增加统一离线评测（`f0b9879`、`309bdd5`）

新增 checkpoint evaluator、Slurm 评测脚本和 metric 单元测试，输出：

- action/noise loss
- retrieval multi-positive loss
- top-real frame 是否落在 positive、explicit negative 或 ignored gray 区域
- joint argmax 是否选择 NULL
- gate 在 0.5 阈值下的 revisit/novel accuracy
- revisit/novel 平均 gate 和 separation
- revisit aux pose MSE

一个重要修正是区分：

```text
实际 frame 执行：argmax(real frame scores)
revisit/novel gate：1 - P(NULL)
旧 joint metric：argmax(real frame scores + NULL)
```

旧的 `revisit_match_accuracy` 把 NULL 与 real frame 一起 argmax，不能代表 policy 实际选择的 frame。现在使用 `revisit_top_real_match_accuracy`，并把失败分成 explicit negative、ignored gray 和 NULL，避免把不同严重程度的错误混在一起。

主要实现与测试：

- [`InternNav/internnav/model/basemodel/memnav/metrics.py`](InternNav/internnav/model/basemodel/memnav/metrics.py)
- [`InternNav/scripts/eval/eval_memnav_offline.py`](InternNav/scripts/eval/eval_memnav_offline.py)
- [`InternNav/scripts/train_memnav/eval_memnav_mp3d.sbatch`](InternNav/scripts/train_memnav/eval_memnav_mp3d.sbatch)
- [`InternNav/tests/unit_test/test_memnav_metrics.py`](InternNav/tests/unit_test/test_memnav_metrics.py)

### 4.5 阻止 feature cache 静默漏数

- 预计算默认 `max_frame_num` 从 1024/2048 统一提高到 4096，并在加载 GPU 模型前检查所选 shard 的最长轨迹。
- 单个 trajectory 即使在运行中失败，shard 完成其余任务后也会返回非零状态，Slurm 不再把不完整预计算标成成功。
- 失败前不再提前创建输出目录，减少空 cache 目录；输出文件继续使用临时文件加原子替换。
- DataLoader 默认要求每个 source-ready episode 同时存在 aggregator 和 camera cache；缺失时汇总示例并终止训练/评测。
- `MEMNAV_STRICT_FEATURE_COVERAGE=0` 只保留给明确需要部分数据的诊断任务。

### 4.6 对齐 Li Guo DataLoader 采样

- 默认恢复 `3af2c8d` 的 leg 内随机 `k`、Goal A、14/83 glimpse 阈值和 `goal_slack=4`。
- covis curve 仍只监督进入 goal leg 之前的历史；own-leg 帧保持 ignore，但可作为结构上有效的 retrieval 候选。
- 保留 metadata `goal.kind` 语义、早期 K/V 候选屏蔽、双 cache 严格检查和 32/8/4096 参数一致性。
- 增加 `fixed_switch`/关闭 Goal A 的复现模式，并把采样配置写入离线评测 JSON。
- 增加轻量 `audit_memnav_sampling.py`，无需加载图像或模型即可审计两套规则的样本数。

## 5. 已有实验结果

### 5.1 单元测试

标签、采样、评测、cache 覆盖与 scene split 共 25 个针对性测试通过：

```bash
pytest -q \
  InternNav/tests/unit_test/test_memnav_cache_coverage.py \
  InternNav/tests/unit_test/test_memnav_labels.py \
  InternNav/tests/unit_test/test_memnav_sampling.py \
  InternNav/tests/unit_test/test_memnav_metrics.py \
  InternNav/tests/unit_test/test_memnav_scene_splits.py
```

这些测试验证标签、random-leg/Goal A 采样、metric 和 cache 覆盖防护，不验证 Habitat 中的闭环导航行为。

### 5.2 本机统一 16 样本 checkpoint 对比

评测条件：同一份本地独立多段数据、同一评测代码、seed 0，共 16 个 sample（10 revisit、6 novel）。两次评测峰值显存约 19.3 GiB，可在本机 RTX 4090 上运行。

| 指标 | 旧 `lg_e1` | 新语义修复版 `ours_e2` | 观察 |
|---|---:|---:|---|
| action loss | 0.2375 | **0.1090** | 降低 54.1% |
| retrieval loss | 1.1389 | **1.0624** | 降低 6.7% |
| aux pose MSE | 9.9864 | **9.5088** | 降低 4.8% |
| revisit top-real positive | **70%** | **70%** | 没有提升 |
| revisit top-real explicit negative | **0%** | 20% | 新版更差 |
| revisit top-real ignored gray | 30% | **10%** | 新版更少 |
| novel NULL accuracy | 100% | 100% | 持平 |
| gate overall accuracy @ 0.5 | **81.25%** | 75% | 新版更低 |
| gate revisit recall @ 0.5 | **80%** | 60% | 新版更保守 |
| gate novel recall @ 0.5 | 83.33% | **100%** | 新版更好 |
| mean gate：revisit | 0.7565 | 0.6642 | 新版更低 |
| mean gate：novel | 0.2162 | **0.0620** | 新版更干净 |
| mean gate separation | 0.5403 | **0.6022** | 新版更好 |

正确解读是：新版 action 学习和 novel 判别更好，但 gate 变得更保守，漏掉更多 revisit；真正执行的 top-real positive accuracy 仍是 70%，且新版两个错误落入 explicit negative。它不是全面提升。

这个对比也不是严格受控实验：旧 checkpoint 约使用 1912 个样本、batch 2、训练 1 epoch/956 step；新版 E2 使用修复后的 2746 个样本、batch 4、训练到 step 1372。数据覆盖、标签和优化步数都不同，因此不能把差异全部因果归因于标签修复。16 个样本也很小：一个 revisit 就是 10 个百分点，一个 novel 是 16.67 个百分点。

本地原始结果：

```text
/home/asus/Research/results/memnav_local_compare/lg_e1_16samples_execution_metrics.json
/home/asus/Research/results/memnav_local_compare/ours_e2_16samples_execution_metrics.json
```

### 5.3 新语义版 E1 的 pt1 全量训练集诊断

`memnav_v2_semantics_e1/checkpoint-686` 已在全部 2746 个训练 sample 上跑完旧一版离线 metric：

| 指标 | 结果 |
|---|---:|
| action loss | 0.12974 |
| retrieval loss | 0.92003 |
| joint retrieval accuracy | 47.82% |
| novel NULL accuracy | 99.66% |
| mean revisit gate | 0.79240 |
| mean novel gate | 0.31974 |
| gate separation | 0.47266 |
| aux pose MSE | 7.68320 |

其中旧 metric 给出的 `revisit_match_accuracy=8.57%` 是 real+NULL joint argmax，不是实际 top-real frame accuracy，不能据此说“重定位只有 8.57%”。修正后的 execution metric 已经完成。

原始结果：

```text
/scratch/yz11502/Research/Nav/InternNav/logs/eval_memnav/memnav_e1_eval_full-13450741.json
```

### 5.4 E3 与 lg154 checkpoint 的统一全量诊断

相同 2746-sample、相同 seed 和 execution metric 下：

| 指标 | `lg154 checkpoint-956` | 我们的 E3 `checkpoint-2058` |
|---|---:|---:|
| action loss | 0.241358 | **0.102639** |
| retrieval loss | 1.197876 | **0.385376** |
| retrieval positive accuracy | 45.34% | **62.64%** |
| revisit top-real positive | 48.24% | **81.70%** |
| revisit top-real explicit negative | 22.84% | **3.20%** |
| novel NULL accuracy | 95.35% | **99.75%** |
| gate accuracy @ 0.5 | 74.04% | **91.48%** |
| gate separation | 0.3579 | **0.7471** |
| aux pose MSE | 7.7210 | 7.6650 |

这证明 E3 对旧训练口径拟合得更好，但两者都看过这些 scene，因此仍然是 train diagnostic。aux pose 几乎不改善，是下一轮 loss-weight 对照的直接依据。

## 6. 当前 HPC 任务状态

2026-07-14 状态与依赖链如下：

| Job | 作用 | 状态 | 进度 |
|---|---|---|---|
| `13500804_[0-7]` | 25 个长 episode cache 补算 | Running/Pending | 8 shards，最多并行 2 GPU |
| `13504863` | aux 0.5，5-step preflight | Dependency | cache 全部成功后启动 |
| `13504864` | aux 0.05，5-step preflight | Dependency | cache 全部成功后启动 |
| `13504865` | train38，aux 0.5，3 epoch | Dependency | preflight 成功后启动 |
| `13504866` | train38，aux 0.05，3 epoch | Dependency | preflight 成功后启动 |
| `13504997 13504998 13504999 13505001 13505002` | `aux=0.5` 的 val-unseen 多 epoch/seed 评测 | Dependency | full training 成功后启动 |
| `13505003 13505004 13505005 13505006 13505007` | `aux=0.05` 的 val-unseen 多 epoch/seed 评测 | Dependency | full training 成功后启动 |

新训练每 epoch 759 step，3 epoch 共 2277 step，预计约 14.5 小时。所有长任务使用 `afterok` 依赖；cache 或环境预检失败时不会继续消耗长任务 GPU。

## 7. 当前不能得出的结论

以下结论目前都没有足够证据：

- “标签修复后所有指标都提高了”——本地 gate recall 和 explicit-negative error 反而退步。
- “trajectory 撞墙就是 planner 训练错了”——还没有闭环 collision 数据来定位问题。
- “模型已经能可靠 U-turn/relocalization”——离线 retrieval 只能证明表示与标签匹配情况。
- “单目 SLAM/里程计已经解决全局定位”——当前 relocalization 依赖冻结 LingBot 视觉记忆和 learned retrieval，不是纯靠传统 SLAM，也没有独立验证长期漂移。
- “已经超过 NavDP baseline”——目前没有在同一 held-out split 上运行官方 NavDP 与 MemNav 的 SR/SPL 对比。

## 8. 仍然存在的技术问题

1. **训练目标与实际决策不完全一致。** 当前 multi-positive loss 优化所有正例的总概率质量，但执行时只取一个 top-real frame。
2. **NULL 同时承担 gate 与 retrieval 竞争。** `1-P(NULL)` 既控制分支又受所有 real score 总量影响，容易出现 novel 更准、revisit recall 更低的保守解。
3. **gray band 没有监督。** `0.1 < covis < 0.5` 不进入 loss，但可以在执行时被选中。
4. **aux pose 目标尺度不理想。** 当前直接对 `(x, y, theta)` 做 MSE，角度 wrap 和不同量纲可能造成尖峰。
5. **优化器配置有歧义。** 顶层配置包含 weight decay/warmup 等字段，但实际 trainer 使用 `Adam(lr=1e-4)` 和 `LinearLR(1.0 -> 0.5, 10000 steps)`；还不是配置中容易让人误以为的 cosine schedule。
6. **完整 90-scene split 尚未覆盖。** pt1 已有严格 38/7/9 train/val-unseen/test 划分，但 pt2 的剩余 36 个 scene 尚未完成 cache。
7. **没有完整在线 runner。** 目前缺少把 streaming memory、DDPM action、controller 和 Habitat episode 串起来的可复现闭环评测。
8. **长序列 keyframe 策略缺失。** 82.7% 的 pt1 episode 超过 LingBot 官方说明的 320-view 训练长度；4096 RoPE 只解决容量，不保证长程精度。

## 9. 下一步

按优先级应执行：

1. 完成 `13500804` 并验证 1944 对 cache 的 key、shape、frame length 和有限值。
2. 完成 aux 0.5/0.05 配对训练，只按 val-unseen component metric 选模型，不比较不同权重的 total loss。
3. 按 memory length 分桶，验证 dense per-frame K/V 在 `>320`、`>1024` 序列上的退化，再设计 keyframe/windowed 对照。
4. 将 binary revisit gate 与 real-frame ranking 分开，并处理 gray frame 与候选数量校准。
5. 将 aux pose 改为 scale-aware 或 scale-free 表示，并正确处理角度周期。
6. 补齐 Habitat 闭环 runner，报告 SR、SPL、collision rate、goal distance、U-turn/revisit 子集结果。
7. 完成 pt2 cache 后扩展到官方完整 61/11/18 building split。
8. 在同一 held-out 闭环 episodes 上运行官方 NavDP，才能回答 MemNav 是否真正优于 baseline。

## 10. 相关提交

| Commit | 内容 |
|---|---|
| `28bac92` | 增加 packed MP3D revisit 数据流水线 |
| `8106284` | 接入真实 MP3D 多段训练、cache 参数和 HPC 脚本 |
| `9eccd5e` | 修复 revisit/novel 标签语义 |
| `3599028` | 增加 bounded smoke/full training |
| `f0b9879` | 增加 checkpoint 离线诊断 |
| `309bdd5` | 区分 top-real 与 NULL，并分类 retrieval failure |
| `d18e9e1` | 强制完整 cache 覆盖与 4096-frame 预计算 |
| `9200db2` | 对齐 Li Guo random-leg/Goal A DataLoader |
| `ed7cf63` | 增加官方 R2R scene-disjoint split、loss env 与可复现 seed |
