# MemNav 概率式记忆定位：本机验证、结论与最终方案（2026-08-06）

## 1. 结论先行

当前最大瓶颈不是 LingBot 在所有转弯处都丢失 pose，也不是在线轨迹中根本没有可用记忆，而是三个不同问题此前被压进了同一个 scalar gate：

1. 候选生成：正确历史区域是否进入候选池；
2. 候选定位：候选池中应该选择哪一个 memory node；
3. 拒绝与置信度：当前是否根本没有可靠匹配，以及相对位姿是否可信。

本轮证据支持以下决定：

- **No-Go**：继续长训旧 DINO-CLS gate、只调 gate threshold、或让一个共享标量同时承担排序和 Novel/Revisit 判定；
- **Go**：先做共享 dense correspondence backbone、但分离 `rank head + no-match head + relative-pose/uncertainty head` 的离线原型；
- **控制策略**：始终保留冻结 NavDP 的原始 image-goal 路径，memory 只提供经过置信度约束的定位/中间 point-goal/residual；
- **时序策略**：用跨时刻 posterior/filter 约束 memory node 的连续性，而不是把少量 temporal summary 直接拼进浅层单 head；
- **几何验证**：RANSAC+SIFT 暂时只作为不确定样本的 fallback 和对照，不作为最终创新主体。

因此现在不应直接提交另一次旧 gate 的 8 小时训练。先把定位目标做对，比继续扩大 policy training 更有效。

## 2. 实验 A：完整候选池到底够不够

输入是 9 个场景、364 个 session、101,873 个历史候选。所有候选均使用 task-aligned directional goal-surface co-visibility 重新标注，而不是把 SIFT/RANSAC 当成 GT：

- co-visibility `>= 0.5`：positive；
- co-visibility `<= 0.1`：negative；
- 中间区域：ambiguous；
- 156/364 个 session 的完整 memory 中存在 positive；其余 208 个没有严格 positive。

完整标注只耗时 40.65 秒。候选 recall 是“在确实有 positive 的 156 个 session 中，候选选择是否保留至少一个 positive”，不是导航 SR：

| 候选方式 | Conditional candidate recall |
|---|---:|
| DINO raw top-1 | 109/156 = 69.87% |
| DINO raw top-4 | 74.36% |
| DINO raw top-8 | 80.77% |
| DINO raw top-16 | 86.54% |
| DINO raw top-32 | 89.10% |
| DINO raw top-64 | 95.51% |
| DINO raw top-128 | 98.72% |
| Temporal-NMS top-32, gap=4 | 151/156 = 96.79% |
| Temporal-NMS top-32, gap=8 | **153/156 = 98.08%** |
| Temporal-NMS top-32, gap=16 | 96.79% |
| Temporal-NMS top-32, gap=32 | 92.95% |

这说明：DINO 的问题不是完全看不懂场景，而是相邻高相似帧会占满 top-K，导致时间上较远但真正可用的帧被挤出去。按时间抑制重复候选后，top-32 已经覆盖绝大部分 oracle positive。gap 太大又会删除有用帧，因此当前 `gap=8` 只是该数据上的候选生成默认值，不能写死为通用真理。

## 3. 实验 B：真实 closed-loop memory 里有没有目标

离线生成轨迹不等于 policy 实际走过的轨迹。新脚本 `audit_online_memory_oracle.py` 读取实际保存的 `legA_memory_trace`，在 Habitat 中逐 pose 重渲染 RGB/depth，再计算 Goal B 表面的 directional co-visibility。

关键校验：旧 trace 只保存 `(x,z,yaw)`，遗漏楼层高度 `y`。脚本通过同楼层 navmesh `snap_point` 恢复 `y`，之后重渲染 RGB 与真实 buffer 的每像素 MAE 为约 `1.2–2.5/255`，说明 pose/坐标映射是对齐的。未来 trace 应直接保存完整 3D pose，避免这个隐含假设。

在已有五个开发场景的 9 条匹配 Goal-B rollout 上：

| 指标 | 结果 |
|---|---:|
| 实际在线 memory frames | 1,134 |
| memory 中存在 co-vis `>=0.5` 的 episode | **9/9** |
| raw DINO top-1 是 strict positive | 6/9 |
| router 激活 | 9/9 |
| 激活 anchor 是 strict positive | 6/9 |
| oracle 最佳 node 到 goal 的直线距离中位数 | 0.805 m |
| 最佳 node yaw error 中位数 | 14.81° |
| reverse-memory + local connector route stretch 中位数 | 1.57 |

这 9 条现有 safe-guidance rollout 最终都到达 B，所以 `co-vis >= 0.5` 不能被误用成“anchor 是否有导航价值”的绝对真理。部分低 co-vis anchor 仍可能通过少量共享背景、metric residual 和 image-goal controller 成功。正确做法是把连续 co-visibility 用于排序监督，再结合 relative-pose NLL、下游 utility 和 calibrated uncertainty，而不是用一个硬阈值代替所有判断。

此外，纯 `(distance >= 0.5 m) OR (yaw change >= 20°)` 的 adaptive keyframe 仅保留约 10 帧/session，但在 1/9 条轨迹中删掉了所有 strict positive。长期 memory 因此必须同时考虑运动覆盖和视觉覆盖，不能只按固定 interval 或纯几何阈值压缩。

## 4. 实验 C：学习式排序是否真的能超过 DINO

在 raw DINO top-32 的 frozen LingBot/DINO symmetric patch cache 上，训练场景固定为：

- `17DRP5sb8fy`
- `1LXtFkjw3qL`
- `1pXnuDYAj8r`
- `Uxmj2M2itWa`

开发留出场景固定为：

- `e9zR4mvMWw7`
- `rqfALeAoiTq`
- `s8pcmisQ38h`
- `yqstnuAEVhm`
- `zsNo4HB9uLZ`

所有 L2 选择只使用四个训练场景的 scene-grouped OOF；五个开发场景只在选定 L2 后评估一次。本轮共有 85 个开发 session，其中 27 个 raw top-32 内存在 strict positive。

| 排序方法 | Correct top-1 | Mean first-positive rank | MRR |
|---|---:|---:|---:|
| DINO cosine | 20/27 = 74.07% | 3.407 | 0.775 |
| Listwise patch+temporal | **22/27 = 81.48%** | **1.519** | **0.874** |

这是本轮最重要的正结果：使用 task-aligned continuous co-visibility 的 listwise objective，确实能学会“在同一个 session 的候选之间排序”，而不只是做 pair classification。提升是 `+7.41 percentage points`，但只有 27 个 positive session，而且这五个开发场景过去已被多次查看，所以它是结构性 Go 信号，不是最终 blind 结论。

## 5. 实验 D：一个 K+1 head 能不能同时排序和拒绝

新建了一个凸的线性 K+1 softmax 诊断模型：K 个 memory candidates 加一个显式 dustbin/no-match 状态。它直接学习：

- 有匹配时，把概率分配给 co-visible candidates；
- 无匹配时，把概率分配给 dustbin；
- 不再通过 action loss 间接期待 gate 自己悟出正确行为。

相同 scene-grouped OOF protocol 下的开发结果：

| K+1 输入 | Candidate top-1 | Match ROC-AUC | AP | Brier↓ | Precision@0.5 | Recall@0.5 | Joint accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosine | 74.07% | 0.734 | 0.710 | 0.183 | 0.824 | 0.519 | 0.812 |
| Patch | 74.07% | **0.920** | **0.889** | **0.104** | **0.905** | **0.704** | **0.847** |
| Patch+temporal | 62.96% | 0.911 | 0.868 | 0.108 | 0.818 | 0.667 | 0.788 |

解释：

- patch correspondence 明显改善“这个候选集合是否有可靠匹配”的判断和校准；
- 但同一个线性 K+1 head 没有改善 candidate top-1；
- 浅层 temporal summary 进一步伤害排序；
- 与实验 C 的 listwise 改善结合看，排序和拒绝存在真实的多任务冲突。

所以最终网络应该共享 dense visual features，但使用不同 head 和 loss。时序连续性应在每步视觉 likelihood 之后作为 posterior transition/filter 使用，而不是直接要求一个浅层 head 同时理解视觉、排序、时序和拒绝。

## 6. 为什么 RANSAC+SIFT 之前有效，但不应成为最终主体

RANSAC+SIFT 的优势是局部特征需要满足同一个几何变换，能排除“整体语义很像但不是同一地点”的 DINO false positive。它很好地证明了 memory 路径有用，也适合作为高精度 fallback。

但在完整 task-aligned 标注上，SIFT extreme teacher 相对 co-visibility 的：

- precision：0.8765；
- recall：0.6265。

所以 RANSAC 也不是 GT：视角变化、低纹理和动态遮挡会让真正有用的 memory 被拒绝。最终系统应让 dense learned localizer 覆盖常规请求，只把低置信样本交给 RANSAC；待 untouched-scene 证据足够后，再逐步降低 fallback 比例。

## 7. 最终架构

### 7.1 Hierarchical memory，而不是固定 320 帧均匀抽样

保留两个层级：

1. recent ring buffer：保留短期连续帧，支持局部避障和刚发生的回环；
2. long-term keyframes：依据位移、yaw、视觉新颖度、局部质量和覆盖度增量写入；相邻重复帧合并为一个 node，但保留其时间范围和 pose covariance。

320 是预算上限，不是把 900 帧简单等比压成 320 帧的唯一规则。关键是保留视觉/拓扑事件和转弯覆盖，而不是追求均匀时间间隔。

### 7.2 Candidate generator

- frozen DINO/LingBot global embedding 做快速全库近邻；
- temporal-NMS/分时段峰值保证候选时间多样性；
- 初版以 top-32、gap=8 作为待验证默认，而非不可变超参数；
- 长轨迹可采用 coarse segment retrieval，再在段内 dense refine，复杂度不需要线性扫描所有 frame。

### 7.3 三个独立 head

共享 query-goal 与 memory candidate 的 dense patch correspondence feature：

1. `rank head`：在同一候选集内输出 listwise score，连续 co-visibility 为监督；
2. `no-match head`：输出 dustbin probability，并进行 scene-disjoint calibration；
3. `pose head`：输出 axis-fixed 地面平面 `(dx,dz,sin Δyaw,cos Δyaw)` 及 covariance，而不是回归未 wrap 的角度或假设 raw LingBot 单位天然等于米。

建议目标：

```text
L = λ_rank * L_listwise
  + λ_nomatch * L_dustbin
  + λ_pose * L_pose_NLL
  + λ_cycle * L_pose_cycle
  + λ_cal * L_calibration
```

不要在本轮证据不足时先拍死具体 λ；先分别验证每个 head 的可学习上限和梯度，再做多任务权重扫描。

### 7.4 时序 posterior，而不是 frame-wise gate

维护 `P(memory_node_t | observations_1:t)`：

- 当前 dense matcher 提供 observation likelihood；
- memory graph 邻接关系和机器人运动提供 transition prior；
- dustbin 是显式状态；
- posterior 连续多个时刻稳定后才启用 memory route；
- posterior 分散、pose covariance 高或几何检查冲突时自动回退 image-goal。

这比“每步独立 gate>0.5 就切 branch”更不容易在 point-goal 和 image-goal 之间来回摇摆，也比直接拼 temporal statistics 更符合问题结构。

### 7.5 控制与 loop closure

- 冻结 NavDP image-goal backbone/controller，Novel 行为始终可用；
- memory localizer 给出可信 node 和相对 pose 后，在 memory graph 上选下一个局部 subgoal；
- subgoal 作为 point-goal/residual 辅助 NavDP，不关闭目标图；
- 每到一个 node 重新定位并更新图，不要求一次 pose 预测覆盖整条 3-leg；
- 最终接近 ImageGoal 后恢复视觉精对齐；U-turn/terminal alignment 是控制动作，不替代 loop closure 本身。

这才真正发挥 LingBot 的优势：不是只把它当一次性的 metric point-goal 生成器，而是把 dense correspondence、streaming pose、记忆图和不确定度用于反复重定位。

## 8. 下一轮实验门槛

### 本机/离线原型（现在应该做）

1. 用 temporal-diverse top-32 重新抽取 dense features；当前缓存只覆盖 raw top-32；
2. 分离 rank/no-match heads，并加入 pose mean+covariance head；
3. 报告每场景 Recall@1、MRR、no-match AUROC/AP/Brier、ECE、pose translation/yaw error；
4. 做 leave-one-scene-out，关注最差场景，不只看平均值；
5. 冻结配置后才使用从未调参的 final-reserved scenes。

### HPC closed-loop（离线达到门槛后再做）

统一同一批 2-leg 和 3-leg episode，对比：

1. 官方/冻结 NavDP；
2. 当前 RANSAC geometry router；
3. learned rank+no-match+pose，RANSAC fallback；
4. oracle memory node/pose 上限。

必须同时报告 Novel SR、Revisit SR、joint SR、SPL、激活率、false activation、memory utilization、每步延迟和按场景方差。一次只报两个场景的平均 SR 不足以判断泛化。

### Go/No-Go 数值门槛

- rank head 在 untouched scenes 上相对 DINO top-1 有稳定提升，且最差场景不明显退化；
- no-match 在设定 precision 下具有可接受 recall，并有可靠 calibration；
- pose covariance 与实际误差正相关，能识别长尾；
- learned-only 失败时 fallback 能恢复，而不是把成功样本破坏；
- 最后才允许一次 8 小时联合训练。

## 9. 可复现命令与依赖边界

完整候选池 co-visibility relabel 和 Recall@K 使用 `memnav` 环境；真实 online-memory 重渲染使用 `habitat` 环境。不要混用，因为 `memnav` 环境没有 Habitat/quaternion，而 Habitat 环境不负责训练 DINO head。

概率式定位 CPU 诊断命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python \
  MemNavData/diag_probabilistic_memory_localizer.py \
  --teacher-csv /tmp/memnav_complete_covis_20260806_v1.csv \
  --feature-cache .diagnostics/patch_temporal_router_20260805/medium_sparse_top32_v1/patch_temporal/patch_temporal_features.npz \
  --out-report /tmp/probabilistic_memory_localizer_20260806_v1.json \
  --heldout-scene e9zR4mvMWw7 \
  --heldout-scene rqfALeAoiTq \
  --heldout-scene s8pcmisQ38h \
  --heldout-scene yqstnuAEVhm \
  --heldout-scene zsNo4HB9uLZ
```

本次输出耗时 55.999 秒。输入对齐通过全部 11,648 个 selected rows 的 DINO cosine 复核，最大误差 `2.98e-08`。

每次提交 Slurm 前必须至少检查：

1. 当前工作目录、Git commit 和 dirty files；
2. Python/conda 环境路径与关键 import；
3. checkpoint、dataset、scene asset、feature cache 是否存在且可读；
4. CUDA 设备、显存和模型权重 SHA；
5. 输出目录是否唯一且可写，禁止静默覆盖；
6. smoke 使用与 full 完全相同的代码路径，只缩短场景/episode/step 数；
7. smoke 完成后才释放 dependency full job；
8. W&B run name、config、commit、seed 和 split manifest 写入输出。

## 10. 已知限制

- online oracle 只有五个旧开发场景中的 9 条匹配 Goal-B rollout，不能代表 broad 20-scene 或 3-leg；
- 当前五个“held-out development”场景过去已多次被查看，不等于最终 blind test；
- K+1 只是线性 objective diagnostic，不是可部署模型；
- 当前 dense cache 是 raw top-32，不是实验 A 中更好的 temporal-NMS top-32；
- geometry graph route 是可行性 upper bound，不是导航 SR；
- strict co-visibility threshold 是监督工具，不是导航 utility 的完整定义。

这些限制意味着本轮可以批准“下一步离线 dense multi-head prototype”，但还不能宣称 router 已经解决，也不能据此宣称 3-leg SR 已提升。
