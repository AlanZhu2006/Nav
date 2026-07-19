# MemNav 长序列稀疏记忆修复：本地验证报告

日期：2026-07-17

本文记录新一轮训练前已经完成的本地验证。所有代码修改、缓存、checkpoint 和诊断输出均位于个人工作树
`/home/asus/Research/Nav-axis-fix`；母目录 `/home/asus/Research/Nav` 只读，未被修改。

## 1. 这次实际修复了什么

旧预计算把每一帧都写进 LingBot aggregator 和 camera KV cache。对超过约 320 帧的长序列，这不符合
LingBot 当前官方 streaming demo 的 keyframe 策略，造成长程姿态明显恶化。

新实现采用官方规则：

```text
keyframe_interval = ceil(num_frames / 320)
```

并明确区分两条时间线：

- `dino_cls` 和 `cam_pose_enc` 仍然逐帧保存，长度始终是原始帧数；
- aggregator/camera KV 只保存 scale 帧和选中的稀疏 keyframe；
- aggregator 的内部时间只按实际 append 的 keyframe 数增长；
- camera head 注入稀疏 KV 时仍使用原始帧号作为 RoPE 时间，不能用稀疏数组行号代替；
- goal relocalization 使用“全局稀疏记忆 + 最近 64 帧 dense warm replay”的双时间尺度结构。

这不是简单把输入每隔几帧抽样。每一帧仍产生 pose 和 retrieval descriptor，只压缩会进入长程记忆的 KV。

## 2. 12 个完整 episode 的 dense 与官方 auto-keyframe 对照

评测集包含 6 个 2-leg 和 6 个 3-leg episode。两组使用相同 LingBot 权重、W32、8 个 scale frame、pad、
bf16 和 SDPA；唯一变量是 post-scale KV 是否执行官方 auto-keyframe。

| 分组 | dense ATE RMSE 中位数 | auto-keyframe ATE RMSE 中位数 | dense 均值 | auto 均值 |
|---|---:|---:|---:|---:|
| 2-leg | 0.165 m | 0.120 m | 0.260 m | 0.167 m |
| 3-leg | 2.025 m | 0.481 m | 2.295 m | 0.552 m |

3-leg 的中位 ATE 下降约 76%。所有 `interval > 1` 的 episode 均改善，`interval = 1` 的短 episode 数值不变。

六个 3-leg episode 的逐条结果：

```text
2.502 -> 0.262 m
1.548 -> 0.528 m
1.039 -> 0.433 m
2.619 -> 0.403 m
1.493 -> 0.704 m
4.568 -> 0.981 m
```

3-leg 的最佳 yaw-axis error 中位数从 `20.36 deg` 降到 `1.84 deg`；RPE gap=256 的方向误差从
`26.83 deg` 降到 `2.76 deg`。

这说明旧实验中的主要长程异常不是“LingBot 天生无法记住转弯”，而是我们的预计算没有遵循官方长序列
keyframe 语义。修复后仍存在少量真实 drift，但量级已经不同。

## 3. production cache 等价性与存储

在 12 个 episode 上重新生成 versioned sparse cache，并与官方 auto-keyframe 诊断逐元素比较：

- `cam_pose_enc`：全部 episode `max_abs = 0`；
- 稀疏 DINO CLS 与旧 dense cache：全部 episode `max_abs = 0`；
- schema、签名、原始索引、帧数、window 和有限值检查全部通过；
- 旧 cache 共 4.801 GiB，新 cache 共 2.007 GiB，减少 58.2%；
- 1329 帧 episode 从 911.8 MiB 降到 185.8 MiB。

本地诊断 cache 是未提交工作树生成的，只用于数值验证。其签名不能用于 HPC 正式训练。正式 cache 必须由最终
clean commit 重新生成，并把任务日志输出的完整 SHA256 填入训练环境变量。

## 4. 为什么 goal 仍需要最近 64 帧 warm replay

使用同一 checkpoint、固定的 10 个 revisit、相同 seed 和 oracle-positive anchor，仅改变 goal append 路径：

| 路径 | 方向误差中位数 | P90 | pose quality 均值 |
|---|---:|---:|---:|
| 旧 dense cache + warm64 | 20.56 deg | 68.62 deg | 0.7380 |
| 官方 sparse cache + warm0 | 10.95 deg | 48.89 deg | 0.8871 |
| sparse global + dense warm64 | 1.99 deg | 14.64 deg | 0.9839 |

`warm0` 是官方稀疏流的严格控制组；它证明全局 keyframe 策略本身有效。`warm64` 进一步补足新插入 goal 所需的局部
连续视觉上下文。因此最终设计不是只选 sparse 或只选 dense，而是：

```text
全局：稀疏 keyframe，控制长程漂移和缓存规模
局部：最近 64 帧 dense replay，保证新 goal 的精细重定位
```

## 5. pose reliability 为什么默认退出 conditioning

30 个固定 revisit（3 个 position seed）上的 sparse+warm64 结果：

- 方向误差中位数 `3.989 deg`，P90 `13.121 deg`；
- 实际 pose quality 均值 `0.9892`，标准差 `0.0241`；
- 旧 reliability 预测均值 `0.9340`，标准差仅 `0.00437`；
- reliability 与实际 quality 的相关系数 `-0.089`；
- 旧 head Brier `0.00366`，常数 `0.99` 的 Brier 反而只有 `0.00058`。

因此旧 reliability head 基本是常数，既没有提供有效排序，也不应继续乘到 semantic revisit gate 上。新默认配置为：

```text
MEMNAV_USE_POSE_RELIABILITY_CONDITIONING=0
MEMNAV_W_POSE_RELIABILITY=0
```

head 和日志仍保留作诊断；旧行为也可显式开启。关闭时，pose code 的 reliability 槽固定为 1，effective gate 等于
semantic gate，reliability 参数不会收到 action/loss 梯度。

## 6. retrieval 诊断说明什么

20 个随机 seed 共 580 条样本，其中 200 条 revisit：

- strict positive hit rate：`0.745`；
- gate accuracy：`0.8966`；
- max raw cosine 对 strict hit 的 AUC：`0.728`；
- normalized entropy AUC：`0.644`；
- temporal local mass AUC：`0.625`；
- rank margin 基本无用。

strict covisibility 的“miss”中有不少只是选中了相邻帧，实际 goal pose 仍很好，因此 strict hit 会低估可用 retrieval。
唯一一次灾难性错误选择的 anchor 产生约 `171 deg` 方向误差，但 semantic gate 已降到 `0.104`，novel 分支能够主导；
旧 reliability 却仍为约 `0.938`。

当前证据不支持立刻增加一个 retrieval-confidence 小网络：小样本 grouped CV 不能泛化，而且标签没有区分“严格非正例但
几何仍可用”的邻帧。更合理的下一步是训练后增加几何容忍的 retrieval 指标，而不是在今晚任务前引入未验证的新 head。

## 7. 真实模型的本地 backward 与 resume

本地用真实 LingBot 权重、真实 sparse cache 和训练代码完成 batch size 1 的 GPU smoke test：

- 从 step 5 checkpoint 恢复到 step 10；
- optimizer、scheduler、RNG 和 dataloader skip 均正常恢复；
- step 6 revisit：retrieval loss `3.1842`，aux direction loss `0.00155`，raw pose direction error `3.12 deg`；
- step 8 revisit：retrieval loss `0.5236`，aux direction loss `0.00867`，raw error `7.44 deg`；
- 368 个其他可训练 tensor 在 step 5 到 10 之间发生变化；
- reliability tensors bit-exact 不变，证明关闭 loss/conditioning 后没有隐式梯度；
- 未出现 OOM、NaN 或 checkpoint metadata 不一致。

这证明代码路径可训练，但 10 步 smoke test 不代表最终收敛效果。

## 8. 防止依赖混用的提交前/任务前约束

新版 cache schema 为 v2，并保存：

- keyframe policy、interval、原始总帧数和 scale 帧数；
- aggregator/camera 的原始帧索引；
- sliding-window；
- InternNav commit、LingBot commit、模型权重 SHA256、脚本和 schema SHA256；
- 由全部配置形成的 precompute signature。

训练零步预检会扫描每个 selected episode 的 metadata 和 `.npy` header，并拒绝：

- legacy dense 与 sparse cache 混用；
- aggregator/camera 来自不同生成批次；
- interval、window、帧数或原始索引不一致；
- payload 截断；
- cache signature 与任务声明不一致；
- 部署 commit 不一致或部署 checkout 有 tracked modification；
- LingBot 依赖缺少 `_set_skip_append`。

提交顺序必须是：

```text
clean commit/push
  -> 独立 HPC 部署目录
  -> sparse precompute array
  -> 全量 zero-step cache preflight
  -> 短 backward/checkpoint smoke
  -> afterok 依赖的 8 小时正式训练
```

任何一步失败，后续任务不得开始。

## 9. 本地测试状态与结论边界

提交前已通过：

- 46 个 MemNav `unittest`；
- 8 个 pytest，另含 4 个参数化子测试；
- 所有改动 Python 文件 `py_compile`；
- 两份 Slurm 脚本 `bash -n`；
- `git diff --check`；
- 母目录 tracked worktree clean。

当前可以确认：

1. axis/extrinsic 与 angle wrap 修复仍在训练链中；
2. 长程主异常已经定位为 dense-KV 集成语义，并被官方 auto-keyframe 大幅缓解；
3. sparse cache 与官方诊断逐元素一致；
4. sparse-global + dense-local goal relocalization 是目前证据最强的结构；
5. 未校准 reliability 不再错误压低 revisit 分支；
6. 新路径能够真实 backward、保存和恢复。

尚需正式训练回答：

1. W&B retrieval、gate 和 aux direction 曲线是否稳定优于旧 400-step run；
2. non-oracle 固定评测是否减少灾难性方向错误；
3. 2-leg 提升是否保持，3-leg 是否把 pose 改善转化为 action/navigation 改善；
4. 最终最佳 checkpoint 位于多少 step，而不是预设“epoch 越多越好”。
