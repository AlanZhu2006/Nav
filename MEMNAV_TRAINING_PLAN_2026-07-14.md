# MemNav 下一阶段训练与评测计划

> 制定时间：2026-07-14
> 实验代码：`ed15f61`（模型/数据改动 `ed7cf63`，W&B 与启动环境检查修复 `ed15f61`）
> 数据：MP3D revisit pt1，scenes 0-53
> 目标：从训练集拟合诊断转向 scene-disjoint 泛化实验

## 1. 决策摘要

今天完成的 3-epoch checkpoint 证明了当前模型可以拟合旧的 2746-sample 训练口径，但没有证明对新建筑泛化，因为旧训练和评测都混用了 pt1 的 54 个 scene。下一轮不继续盲目增加 epoch，而是：

1. 补齐 25 个超过 2048 帧的 episode cache，训练前强制达到完整覆盖。
2. 按官方 R2R building split 使用 pt1 中的 38 train / 7 val-unseen / 9 test scene。
3. 从相同随机初始化训练两条 3-epoch 配对实验，只改变 `w_aux_pose`：0.5 对 0.05。
4. 每个 epoch 在 val-unseen 上评测，最终 epoch 再用 3 个随机 `k` seed 重复评测。
5. test 9 scenes 暂不使用；模型选择完成后才允许做一次最终 test。

这次对照的核心问题不是“是否多训几轮”，而是高权重、尺度不适定的 aux pose 监督是否在伤害共享 revisit 表征。

## 2. 今天的实验说明了什么

统一口径的旧 2746-sample 训练集诊断如下：

| 指标 | `lg154 checkpoint-956` | 我们的 E3 `checkpoint-2058` | 变化 |
|---|---:|---:|---:|
| action loss | 0.241358 | **0.102639** | -57.47% |
| retrieval loss | 1.197876 | **0.385376** | -67.83% |
| retrieval positive accuracy | 45.34% | **62.64%** | +17.30 pp |
| revisit top-real positive | 48.24% | **81.70%** | +33.46 pp |
| novel NULL accuracy | 95.35% | **99.75%** | +4.40 pp |
| gate accuracy @ 0.5 | 74.04% | **91.48%** | +17.44 pp |
| gate separation | 0.3579 | **0.7471** | +0.3892 |
| aux pose MSE | 7.7210 | 7.6650 | -0.73% |

E1 到 E3 之间，action loss 从 0.129738 降到 0.102639，retrieval loss 从 0.920029 降到 0.385376；action 在后期趋于平台，而 retrieval 仍明显学习。aux pose 基本没有学习。

旧 E3 的离线总 loss 为：

```text
0.102639 + 1.0 * 0.385376 + 0.5 * 7.665015 = 4.320523
```

其中 aux 项贡献 3.832508，占 88.70%。改为 0.05 后，在相同 component loss 下 aux 贡献会降到 0.383251，与 action + retrieval 的 0.488015 接近。总 loss 数字在不同权重之间不可直接比较，必须比较各 component 和下游 metric。

### 能得出的结论

- 当前 action、retrieval 和 gate 在训练过的 scene 上不是完全失效的。
- 第 3 epoch 对 retrieval 仍有价值，因此新实验保留 3 epoch。
- aux pose 是最明显的优化异常，也是当前最值得隔离的变量。

### 不能得出的结论

- 不能说 E3 对未见建筑泛化，因为旧实验存在 scene 泄漏。
- 不能从离线 loss 推导 SR、SPL 或 collision rate。
- 不能说已经超过 NavDP。当前只复用了其 diffusion/action 代码路径，没有同 split 的 NavDP 闭环基线。

## 3. 为什么 aux pose 权重是今晚的变量

当前 `RevisitMerge` 只接收 LingBot camera head 的两个绝对 pose9，去掉 FoV 后用平移和单位四元数预测米制 `(x, y, theta)`。问题是 LingBot-Map 本身对单目尺度采用 anchor-derived normalization：训练时用 anchor 点云平均距离归一化深度和相机平移；论文的 ATE 也在 Sim(3) 对齐后报告。

仓库已有 revisit sweep 进一步验证了这个问题：8 条轨迹的全局 metric scale 约为 1.40 到 3.72，局部 scale 约为 0.94 到 4.22，并非一个可由 pose token 单独恢复的固定比例。因此米制 aux translation 对当前输入是部分不适定的。

这不等于 aux pose 永远没有用。后续正确方向是：

- 用 robot odometry、depth、已知相机高度或速度校准每条轨迹的 Sim(3) scale；或
- 改成 scale-free 方向、`sin/cos(theta)`、归一化或 log-distance 目标；或
- 用 Huber loss，并采用 GradNorm/uncertainty weighting 等基于梯度或任务不确定性的权重方法。

今晚只降低静态权重，不同时修改 target representation，保证实验只有一个变量。多任务 loss 的尺度敏感性和自适应平衡可参考 [Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) 与 [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html)。

## 4. 严格数据划分

官方 R2R 明确区分 train、val-seen、val-unseen 和 test，val-unseen/test 的 building 不在训练 building 中。依据官方 scene ID 对 pt1 的 54 个 scene 求交后：

| Split | scenes | episodes | 语义正确样本 | covis revisit | covis novel | Goal A |
|---|---:|---:|---:|---:|---:|---:|
| `r2r_train` | 38 | 1407 | 3037 | 1146 | 875 | 1016 |
| `r2r_val_unseen` | 7 | 280 | 620 | 229 | 175 | 216 |
| `r2r_test` | 9 | 257 | 549 | 212 | 158 | 179 |
| all pt1 | 54 | 1944 | 4206 | 1587 | 1208 | 1411 |

代码在 `ed7cf63` 中加入：

- `MEMNAV_SCENE_SPLIT={all,r2r_train,r2r_val_unseen,r2r_test}`；
- Dataset、audit、训练和评测使用同一个 split 定义；
- split typo 直接失败；
- 训练 seed 在模型初始化、TrainingArguments 和 sampler 中统一；
- `MEMNAV_W_AUX_POSE`、`MEMNAV_W_RETRIEVAL` 写入作业环境与日志。

官方 split 说明见 [Matterport3DSimulator R2R README](https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/README.md)。

## 5. 长序列风险

pt1 frame-length 审计：

| 范围 | 全 pt1 | train split |
|---|---:|---:|
| `> 320` frames | 1607 / 1944 (82.7%) | 1162 / 1407 (82.6%) |
| `> 1024` frames | 263 | 181 |
| `> 2048` frames | 25 | 11 |
| `> 3000` frames | 4 | 3 |
| max | 3954 | 3954 |

把 RoPE table 扩到 4096 只解决数组容量和崩溃，不自动保证长序列精度。LingBot 官方说明 video RoPE 训练长度为 320 views，超过后建议减少 KV keyframe 数；超过约 3000 frames 建议 windowed mode。我们当前对每帧保留 6 个压缩 trajectory-memory token，没有 keyframe interval 或 state reset。

因此今晚的新训练是必要的干净 baseline，但不是最终长序列方案。后续必须单独做 keyframe stride / flow keyframe / windowed Sim(3) 对齐实验，不能把它混进 aux 权重对照。参考 [LingBot-Map 论文](https://arxiv.org/abs/2604.14141) 与[官方实现的长序列说明](https://github.com/robbyant/lingbot-map#streaming-with-keyframe-interval)。

## 6. 当前任务与故障记录

### 6.1 Cache 补算

| Job | 配置 | 作用 |
|---|---|---|
| `13500804_[0-7]` | H100/H200, 8 shards, `max_frame_num=4096` | **8/8 completed, errors=0**；补算 25 个长 episode，已有 1919 个通过 symlink 复用 |

任务幂等、原子写入，任何 shard 有错误都会返回非零。完成后 `lingbot_cache.npz` 与 `lingbot_cam_cache.npz` 均为 1944/1944，其中各 25 个是新写入的 regular file。

### 6.2 配对训练

| 变量 | control | treatment |
|---|---:|---:|
| scene split | `r2r_train` | `r2r_train` |
| samples | 3037 | 3037 |
| batch | 4 | 4 |
| epoch / steps | 3 / 2277 | 3 / 2277 |
| seed | 0 | 0 |
| `w_retrieval` | 1.0 | 1.0 |
| `w_aux_pose` | **0.5** | **0.05** |
| sampling | random-leg + Goal A | random-leg + Goal A |

第一次提交的 `13504863/13504864` 在容器内执行 `which python` 时因 alias 不兼容而分别以 exit code 2 失败；`afterok` 阻止了长训练。未运行的 `13504865/13504866` 和旧评测 jobs 随后被取消，运行时间均为 0。

修复后的依赖链：

```text
13560476 aux0.5 online-W&B preflight (5 steps, completed)
  +-> 13560538 aux0.5 full E3 (running)
13560516 aux0.05 online-W&B preflight (5 steps, completed)
  +-> 13561091 aux0.05 full E3 (running)
```

两个 full job 均运行于 H100，申请 20 小时。启动检查确认 Python 3.10.20、PyTorch 2.8.0+cu128、Transformers 4.51.0、W&B 0.28.0、LingBot 权重 `0 missing / 0 unexpected`、3037 样本和完整 cache。当前 Slurm 平均 GPU utilization 为 81%/82%，显存约 25 GiB。旧 E3 实测 22.90 秒/step；2277 step 估计约 14.5 小时。

在线训练曲线：

- [aux 0.5 W&B run](https://wandb.ai/yz11502-new-york-university/memnav/runs/gn0gue4u)
- [aux 0.05 W&B run](https://wandb.ai/yz11502-new-york-university/memnav/runs/g4hn0mp8)

### 6.3 自动 val-unseen 评测

每条训练评测 `checkpoint-759`、`checkpoint-1518`、`checkpoint-2277` 的 seed 0；最终 checkpoint 再评测 seed 1/2。

```text
aux0.5:  13561416 13561419 13561429 13562122 13562266
aux0.05: 13562641 13562646 13562663 13562684 13562688
```

每份评测为全部 620 个 val-unseen sample，batch 4，预计约 55 分钟。随机 seed 改变的是 random-leg 当前帧 `k`，最终报告使用三个 seed 的均值和离散程度。

## 7. 明早如何判定结果

不同 aux 权重的 total loss 不可比较。按以下顺序选择：

1. **Primary:** val-unseen action loss、revisit top-real positive accuracy、novel NULL accuracy。
2. **Gate:** revisit/novel balanced accuracy 和 gate separation，不只看受类别比例影响的 overall accuracy。
3. **Failure type:** explicit-negative fraction 必须下降或至少不恶化；ignored-gray 单独报告。
4. **Stability:** final 三个 `k` seed 的均值与范围；单 seed 小幅领先不算胜出。
5. **Aux:** 仅作为诊断，不因 aux MSE 更低就选择 action/retrieval 更差的模型。

若 aux0.05 在 primary 指标上持平或更好，则采用 0.05 进入结构实验。若 control 更好，说明 aux 梯度仍提供有效 regularization，下一步应做 0.1/0.2 或自适应权重，而不是直接删除。

如果两条曲线在 E2 后都开始恶化，不再训练 50 epoch；选择 val 最佳 epoch。只有 val 在 E3 仍一致改善时，才考虑实现真正的 optimizer/scheduler resume 后扩到 E5。

## 8. 结果之后的优先级

1. **长序列 K/V 实验。** 按 memory length 分桶评测 gate/retrieval/pose；比较 dense history 与 keyframe-strided history，检查 `>320` 和 `>1024` 的退化。
2. **修复 aux representation。** 用方向 + `sin/cos(yaw)` + 归一化距离，或引入可用的 metric scale calibration。
3. **拆分 gate 与 retrieval ranking。** 当前 `1-P(NULL)` 会随 real candidate 数量变化，应增加独立 matchability head，并按 history length 校准。
4. **Oracle ablation。** 分别使用 GT gate、GT positive retrieval frame，定位失败来自 retrieval、relocalization pose 还是 action decoder。
5. **Habitat 闭环。** 报告 SR、SPL、collision rate、distance-to-goal，以及 U-turn/revisit 子集。Habitat 的 SPL 定义可参考[官方 challenge](https://github.com/facebookresearch/habitat-challenge)。
6. **扩展 pt2。** pt2 包含剩余 36 个 MP3D scene；完成 cache 后可覆盖官方完整 61/11/18 building split。
7. **公平 NavDP baseline。** NavDP 官方方法是 RGB-D 条件的 diffusion trajectory generation + critic selection；当前 MemNav 去掉了 critic，不能把旧 `lg_e1` 当官方 NavDP。需在同一 held-out 闭环任务中比较。参考 [NavDP 论文](https://arxiv.org/abs/2505.08712)。

## 9. 已知实现债务

- `MemNavTrainer` 实际使用 plain Adam + 10000-step LinearLR；配置中的 AdamW weight decay、warmup、cosine 没有生效。
- `ckpt_to_load` 只载入模型权重，`trainer.train()` 没有恢复 optimizer/scheduler，不能称为连续 resume。
- `--skip_scale` cache 在训练时重算前 8 帧 full K/V；sample shuffle 使 4-entry GPU LRU 命中率可能偏低。
- 当前 offline eval 每个 goal 只采一个随机 `k`，所以最终 checkpoint 必须保留多 seed 评测。
- 仍缺在线 policy/controller runner；离线 loss 不是导航成功率。
