# Cyclic Goal Compass（CGC）冻结协议

日期：2026-08-09  
状态：**train-scene observability gate 已冻结；smoke 通过前不授权正式训练。**

本文定义今晚最多 8 小时的 HPC 任务。它不是超参数搜索，不读取 development、
final-reserved 或 blind 场景，也不把 Habitat 教师结果表述成可部署闭环结果。

## 1. 要回答的唯一问题

冻结 NavDP 在 Novel-A 上已有显著 oracle-bearing 上限，但单张单目图像通常不包含
“目标不可见时全局路径向左还是向右”的充分信息。CGC 不强迫一个网络从单帧猜
全局方向，而把方向估计改写成一个主动、对称的局部比较问题：

> 在同一物理位置依次观察完整的 8 个 yaw，冻结 NavDP 的 ImageGoal encoder
> 能否在未见训练场景 OOF 上读出一个随目标变化的局部进展场，并且只在有把握时
> 优于保持原朝向？

通过只说明这个方向信号**可观测、可学习**；不说明 active scan 已有闭环收益，
也不说明 scan 成本、触发器和执行安全已经解决。

## 2. 架构：一个圆周接口，两种方向来源

```text
                         reliable revisit memory
                                  │
                                  ▼
                           von-Mises ring prior       （后续）
                                  │
goal image ─┐                     ▼
            ├─ frozen NavDP ImageGoal encoder ── C8 ring head ── abstain/heading
8-yaw scan ─┘             （每个 yaw 共享同一 encoder）
                                  │
                                  ▼
                         existing token executor      （闭环 gate）
```

Novel 场景的方向来自主动环视；通过几何验证的 revisit memory 以后只作为同一个
圆周场上的解析先验，不另建 memory policy。这样“记忆”和“探索”不是两套架构，
只是 bearing evidence 的两个来源。

### 2.1 C8 等变，而不是八分类器

- 单目相机在原地依次采集 8 个相隔 45° 的视图；不假设全景相机。
- 每个 `(goal image, yaw view)` 经过冻结的 NavDP ImageGoal encoder，得到 384 维
  feature。
- head 没有绝对方向 embedding，也不接收 expert/native index。
- 循环平移输入环必须精确循环平移输出环；测试容差 `max_abs_error <= 2e-5`。
- 每个物理状态用与标签无关的 SHA256 gauge 随机化环起点，堵住“expert 总向前”
  的 shortcut。

这使网络学习的是“哪一幅目标条件视图更有进展”，而不是某个数据集固定方向。

### 2.2 教师是 transition progress，不是 bearing class

对每个请求 yaw，冻结 navmesh 定义一次 1 m `try_step` transition：

```text
A_k = d_geo(state, goal) - d_geo(T_k(state), goal)
```

- 撞墙或不动自然得到约 0 m 进展；
- 沿墙滑动按实际终点计分；
- 8 个方向全部定义，训练和 OOF 推理都**不使用 Habitat candidate mask**；
- extent 与 heading fidelity 只作为诊断字段，不能帮模型筛方向；
- soft listwise target 的温度固定为 0.25 m。

Habitat 因而只产生训练监督，不向部署选择器泄露可达性。这一约束比把 oracle
bearing 量化成分类标签更接近“采取这个方向干预后会怎样”。

### 2.3 因果 goal-swap

每个物理状态同时有 factual 与同场景 counterfactual Goal-B：

- 两行共享完全相同的 pose、8 张 scan 图像和几何；
- 只替换 goal image，并用相应 goal 重新计算进展场；
- OOF 时把另一目标喂给当前 row，若 NLL 没有显著恶化，则模型没有真正使用
  goal，实验失败。

它直接排除“识别场景难度”“看自由空间”“永远沿 expert 朝向”等伪方向信号。

## 3. 数据与独立单位

- 来源：冻结 `nlsr_v2_multistage_expert_candidate_manifest_v1`；
- 范围：只取冻结 train split 的 40 scenes；
- 状态：Novel-B `goal_b_t0` 与 `goal_b_midpoint_t1`；
- 规模：160 个物理 scan groups、320 个 goal-conditioned rows、1280 张新渲染
  单目视图；
- 独立统计单位：scene，不把 320 rows 当作 320 个独立场景；
- 10-scene development、final-reserved 和 blind 均不被代码接受。

Novel-B expert states 与真正的 Novel-A frozen-policy failure states 存在分布差异，
所以本任务只做 observability gate。若通过，下一步必须是新预注册、场景不相交的
policy-state paired gate，不能回到已经耗尽的 development 上挑阈值。

## 4. 容量阶梯，不是调参

所有训练设置在看到结果前固定：5-fold scene OOF、seeds `11/29/47`、300 epochs、
AdamW、learning rate `3e-4`、weight decay `1e-4`、batch size 32。

只允许按 Occam 顺序评估两种 head：

1. `linear`：每个 yaw 共享 LayerNorm + linear evidence；
2. `ring`：只有 linear 未通过时，才评估共享 projection + 两层 circular conv。

linear 一旦通过，ring 不再训练。二者不是待搜索的超参数集合，而是“单视图匹配
是否足够”与“是否确实需要邻域圆周上下文”的预注册可观测性分解。任何一个都不
通过时停止，不增加 Transformer、不扩大 hidden size、不改 loss 温度。

## 5. 冻结通过门

主分析只用 `goal_b_t0`，并同时满足：

1. factual/counterfactual primary pairs 中至少 25% 的 teacher best bin 不同；
2. OOF 置信度最高的 50% 非 native 干预，相对保持 expert/native heading proxy 的
   scene-cluster bootstrap 95% CI 下界大于 0 m；
3. 上述 50% takeover budget 中只允许 `best_non_native > native` 的 row 真正接管，
   且实际接管率至少为 25%；其余 row 必须 abstain；
4. 正确 goal 相对 same-state swapped goal 的 NLL increase，其 scene-cluster
   bootstrap 95% CI 下界大于 0。

报告同时给出 25/50/75/100% risk-coverage、gain/loss、top-1、regret 与每折三个
seed，但不得从这些 secondary 数字重新选择 coverage 或模型。

冻结决策只有两个：

```text
go_<linear|ring>_to_preregistered_disjoint_policy_state_gate
stop_active_goal_compass_observability_not_established
```

## 6. 今晚 8 小时任务的实际工作

一个 H100 作业内顺序完成：

1. 校验 source bundle、manifest、split、GLB、navmesh、overlay、NavDP checkpoint
   与 backbone 的 SHA256；
2. Habitat 逐 scene 渲染 active scans 并构造八方向 counterfactual progress；
3. 冻结 NavDP ImageGoal encoder 抽取 correct-goal 与 same-state swapped-goal feature；
4. 做 5-fold × 3-seed scene OOF；
5. 按上述冻结门自动 `go/stop`；只有通过时才训练 all-40 三 seed ensemble。

8 小时是墙钟上限，不是必须耗尽的训练配额。脚本不做 learning-rate、hidden-size、
温度、coverage 或 epoch sweep；低容量模型通过时会主动停止更大模型。

实现入口：

```text
MemNavData/build_cgc_multiyaw_dataset.py
MemNavData/circular_goal_compass.py
MemNavData/train_cgc_scene_oof.py
MemNavData/slurm_cgc_observability.sbatch
MemNavData/test_circular_goal_compass.py
```

## 7. 通过后仍未解决的事情

- active scan 的旋转距离、时间成本和碰撞安全；
- 何时触发 scan：novelty、低圆周 margin、stall，三者需要独立冻结；
- expert Novel-B 到 frozen-policy Novel-A failure distribution 的迁移；
- 预测 heading 经现有 iterative token executor 后的真实 paired SR；
- memory von-Mises prior 是否在可靠 geometry gate 下无损融合。

因此最早的可部署闭环版本必须是：native 默认保留，只有 trigger 与 compass 均通过
OOF 校准时才接管一次；转不完、可达性失败或 margin 不足都 abstain 回 native。
