# Pi3X 时序方向异常：封存轨迹审计与因果上下文小重放

日期：2026-09-06（Asia/Shanghai）。这是 [CEC 成果复审](CEC_CONTRIBUTION_AND_IMPROVEMENT_AUDIT_20260906.md) 的后续，不是新一轮正式 SR 评测。

后续已完成：[连续当前状态与稀疏目标关系读出实验](PI3X_PERSISTENT_GOAL_READOUT_RESULT_20260906.md)。同两条记录序列、全部 70 个重规划时刻；原 accepted 的 64 个时刻中，连续 LingBot 状态 + 固定初始 Pi3X 目标关系将中位方位误差从 16.28° 降到 2.19°，>90° 从 12 次变为 0 次。主要改善在保留动态目标关系、改用连续状态与共同历史对齐时已出现。仍为 N=2 的离线诊断，无新 SR。

## 1. 本轮结论

此前不能把 Pi3X 的 69 次 geodesic 方向差全部解释成定位错误；改用目标直线方位后，仍有 20 次 >90°。本轮进一步发现：

1. 正式 natural-direction 中，Pi3X 获得授权的 23 条 query，首次方位误差全部 ≤5.565°。20 次大角度异常全部发生在后续更新，而非首次授权。
2. 四条异常 query 的 anchor 始终未切换。更新时当前 RGB 与均匀桥接帧一起变化，不能仅凭时序关联认定哪一个导致异常。
3. 对其中两条做本机固定当前图、固定目标图、固定 anchor 的上下文干预：一个案例仅换回上一时刻的因果桥接帧，误差从 178.342° 变为 0.383°；另一案例仅从 72.750° 变为 45.904°，并未恢复到原训练的 ≤30° 范围。
4. 在共同 anchor 的分析坐标系下，变化主要落在 **current-to-anchor 关系**，而不是 anchor-to-goal 关系。这支持检查稀疏联合重建对当前状态的稳定性，不能直接归咎于 DINO、LingBot 漂移或“学习本身做不好”。

这不是通用修复，也不是冻结旧 bridge 后 SR 提升的证据。当前 CEC 不变，Pi3X 原正式替换决定仍为未通过。

## 2. 封存轨迹的独立复算

读取：21 histories × 2 roles × 2 arms，共 84 个 plans JSON；另读取 manifest、episode contracts 和 completion receipts。没有执行导航或训练。

| 指标 | Pi3X | CEC |
|---|---:|---:|
| query 总数 | 42 | 42 |
| 有 accepted plan 的 query | 23 | 23 |
| 首次 accepted 方位误差中位数 | 3.005° | 0.872° |
| 首次 accepted 方位误差最大值 | 5.565° | 3.199° |
| 全部 accepted plans | 479 | 449 |
| 全部 accepted 方位误差中位数 | 4.066° | 2.609° |
| 全部 accepted 方位误差 P95 | 88.249° | 12.825° |
| 目标直线方位误差 >90° | 20 | 0 |

没有缺失 pose/bearing 对应关系；复算距离与计划收据距离的最大差为 `4.44e-16 m`。两臂后续轨迹不同，这不是同一状态下的逐计划比较；计划也不是独立 episode。小的初始 bearing error 不代表完整相对位姿或目标语义已正确。

### 四条异常 query

| scene / episode / role | 固定 anchor | accepted / 总 plans | >90° | 原 Pi3X / CEC outcome |
|---|---:|---:|---:|---|
| 8WUmhLawc2A / 0000 / Revisit | 140 | 49/55 | 11 | 失败 / 成功 |
| PuKPg4mmafe / 0001 / Revisit | 79 | 37/38 | 3 | 成功 / 成功 |
| PuKPg4mmafe / 0004 / Novel | 8 | 42/51 | 4 | 成功 / 成功 |
| V2XKFyX4ASd / 0004 / Revisit | 39 | 15/15 | 2 | 成功 / 成功 |

因此“大角度计划异常”不与 episode failure 一一对应。三个最终成功 query 也经历过异常，不能将 20 次错误当作 20 条可挽回失败。

V2XKFyX4ASd 的 step 96→104，实际移动 0.2993 m，真实世界目标方位只变了 3.119°，预测世界方位却变了 167.525°；endpoint error 从 8.049° 到 172.454°。这不是机器人真实转头本身造成的坐标表达变化，计算已消去当前真实 yaw。

8W 后段存在完全相同 RGB 哈希的多次查询，但这些重复图对应的预测摆幅只有约 0.2–4.2°。它们说明重复观测期间仍有上下文相关变化，却不能解释前段的大跳变，故没有据此宣称“相同图片导致 180° 翻转”。

## 3. 原实现究竟怎样更新

本机文件与实际 attempt7 source bundle 的下列文件 SHA 相同：

- `MemNavData/pi3x_online_relocalizer.py`
- `MemNavData/pi3x_spatial_proof_runtime.py`
- `MemNavData/train_pi3x_viewtoken_reliability_oof.py`
- `MemNavData/diag_pi3x_multiview_consistency.py`
- `third_party/Pi3/pi3/models/pi3x.py` 与本机 Pi3 的对应模型文件。

Pi3X 的运行流程是：首次选定历史 anchor；之后每次重规划，固定该 anchor，从它到当前帧之前重新等间隔抽 16 个 bridge 帧，并加入 anchor±8 支持帧；将当前图、这些历史图和目标图联合前向。方向由新估计的 current pose 与 goal pose 求得，proof 也会重新计算，并非首次 accept 后永远跳过检查。

CEC 的状态生命周期不同：首次认证后保存目标位姿，当前位姿来自连续 LingBot 状态。Pi3X 这条 learned arm 的空间读出不使用 LingBot pose/depth，因此本次 Pi3X 异常不能作为 LingBot 漂移的证据。

另外，旧 learned proof 已经有 `navigation_action_label`：正 session 要求 endpoint bearing error ≤30°，并另设 support 辅助目标。训练方向标签本来就是目标直线方位，不是 geodesic 首段。因此不能把下一步包装成“第一次加入方向正确性监督”；错误口径主要发生在正式闭环统计解释链，不是已证明训练标签也用错。

## 4. 本机固定观测重放

### 4.1 实验设计

只选两条已消费 query，诊断异常前后：

- 8WUmhLawc2A / episode_0000：step 232 和 240；原 Pi3X 唯一未成功而 CEC 成功的该组异常案例。
- V2XKFyX4ASd / episode_0004：step 96 和 104；原记录中明确的大方向跳变，但 episode 最终成功。

每条三个条件，重复两次，第二次倒序执行，共 12 次前向：

| 条件 | 当前图 | 桥接帧 |
|---|---|---|
| old original | 异常前 | 该时刻原 b16 |
| new original | 异常时 | 该时刻原 b16 |
| new previous bridge | 与 new original 完全相同 | 上一时刻 b16 |

目标图、anchor、模型、输入尺寸及 dtype 均固定。第三条件不包含未来帧，两组桥接帧都严格早于当前帧。没有让旧 current 读取未来 bridge，也没有改变模型权重或搜索角度阈值。

拉取了 70 个图像文件，合计 3,345,074 bytes；历史图逐项对照原 rollout 哈希，目标图对照 frozen manifest，全部一致。仅处理已封存观测，GT 在网络前向完成后用于评分；没有把 GT pose/depth 传入模型。

### 4.2 实际结果

| 案例 | 原 H100 异常时误差 | 本机 old original | 本机 new original | 本机 new previous bridge |
|---|---:|---:|---:|---:|
| 8WUmhLawc2A | 93.897° | 25.754° | 72.750° | 45.904° |
| V2XKFyX4ASd | 172.454° | 8.139° | 178.342° | 0.383° |

两次同进程重复的所有相机 pose 矩阵最大绝对差为 0。它只表明这次局部对照可重复，重复前向不增加统计样本量。

必须披露的复现边界：本机 RTX 4090、Torch 2.5.1+cu124、bfloat16；启动日志提示使用 PyTorch RoPE2D 实现而非编译 CUDA 版本。原 H100 结果未逐位复现，尤其 8W 异常方向相差 21.147°。因此局部因果结论建立在本机同进程比较上，不将它冒称原 H100 轨迹的精确反事实。V2 的翻转现象得到复现；8W 原 >90° 阈值事件没有在本机原条件精确复现。

没有运行 learned proof heads、NavDP 或动作执行；因此这些角度结果不说明新上下文一定会获授权，更不说明导航成功。

### 4.3 哪个关系在变

为比较两个重建，分析中以同一历史 anchor 为参考朝向，并用共同 anchor−8 / anchor+8 相机基线长度归一化平移。该尺度只是分析单位，不是米，也没有直接拼 Pi3X 与 LingBot pose。

比较固定 current 的 `new original` 与 `new previous bridge`：

| 案例 | current-to-anchor 平移差（基线单位） | anchor-to-goal 平移差 | current 相对旋转差 | goal 相对旋转差 |
|---|---:|---:|---:|---:|
| 8W | 1.7043 | 0.0179 | 30.536° | 0.370° |
| V2 | 2.8734 | 0.0175 | 56.680° | 0.264° |

补充的纯分析交换也指向相同部分：V2 保留 new current 估计、只换上一上下文的 goal 关系，误差仍为 178.038°；保留 new goal 关系、换上一上下文的 current 估计，误差为 0.637°。

这不是部署方式或完美 gauge 对齐证明；anchor 自身和支持基线仍可能有误差。不过它与固定输入上下文干预共同说明：**在这个具体重放中，主要不稳定项是当前状态与历史的连接，而非目标相对历史 anchor 的关系。** 不能扩展成所有 Pi3X 失败的统一原因。

## 5. 对下一步方法设计的影响

### 不值得立即做

- 不重新训练 DINO selector：这四条没有发生 anchor 切换，首次方向也不差。
- 不把 proof threshold 调到更严：本轮没有证明这样能保留 recall；旧 head 已有方向正确性监督。
- 不把旧 bridge 永久冻结：机器人继续走以后会失去与旧片段的视觉连接，本轮仅测一个更新间隔。
- 不直接冻结第一次 bearing：它必须随机器人位姿更新。
- 不把 Pi3X、LingBot、matcher 再全部并列接入主架构，增加组件并不解决状态职责混合。

### 值得继续的最小研究问题

**稀疏 learned readout 只负责“历史 anchor 与目标的关系”，连续几何状态负责“我现在在哪里”。**

这保留当前 CEC 已验证的时序分工：

```text
连续因果几何状态 ─────────────── 当前状态
历史地址 + 目标图 ── 稀疏关系读出 ── 目标在历史中的关系
                                  ↓
                          相对方向 → 冻结 controller
```

若研究 learned 替换，应先在固定观测上检验一次目标关系读出与共享状态的组合，不让一个稀疏 b16 重建每次同时承担当前定位、目标定位和开放集授权。Pi3X 可以作为该诊断的关系估计器，但不是决定把第三个大模型加进论文主方法。

这个建议仍有实质待办：Pi3X 与 LingBot 必须用共同历史证据对齐尺度/坐标；初始小 bearing error 不保证完整目标关系正确；减少重复重建是否真正降低延迟并保持 SR 未被测试。因此本轮没有实施生产替换或提交闭环长评测。

对当前论文最重要的是讲清共享时序状态与稀疏目标查询的职责，而非再声称“加一个 gate 就解决学习”。本轮提供了为什么不能随意用稀疏 batch relocalizer 替代连续状态的一项具体开发机制证据；它不构成新的正式主表。

## 6. 文件与运行状态

所有新诊断位于：

`.diagnostics/pi3x_temporal_attribution_20260906/`

- `audit_frozen_plans.py` / `frozen_plan_audit.json`：全 natural-direction 的时序复算、四条 timeline、源文件 SHA。
- `prepare_replay_inputs_remote.py` / `replay_inputs.json`：两条诊断的精确图像列表与 frozen SHA。
- `inputs/`：70 张封存图像的本地副本。
- `replay_causal_context.py` / `causal_context_replay.json` / `replay_console.log`：12 次前向及完整输出 pose。
- `analyze_replay.py` / `context_component_attribution.json`：共同 anchor 分析与 reporting-only 关系交换。

复现本机小重放：

```bash
/home/asus/miniconda3/envs/pi3x/bin/python -u \
  .diagnostics/pi3x_temporal_attribution_20260906/replay_causal_context.py \
  --manifest .diagnostics/pi3x_temporal_attribution_20260906/replay_inputs.json \
  --input-root .diagnostics/pi3x_temporal_attribution_20260906/inputs
```

本轮未修改生产策略、论文、原正式 summary 或冻结决定，未提交/取消 HPC GPU 作业，未操作机器人，未 commit/push。本机前向已结束，显存回到本轮开始时 6,556 MiB 的已有占用。

使用既有 `alantorch` / `yz11502` 共享连接。SFTP 子传输停滞后按项目手册走 localhost-only 小文件通道；已停止本轮 HTTP 子进程、撤销转发、退出审计 PTY，共享 master 保持正常。不是认证过期，也没有更换账户/socket。
