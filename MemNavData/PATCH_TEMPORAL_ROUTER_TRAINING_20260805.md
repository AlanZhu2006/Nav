# Patch-temporal reliability router：本机验证与 HPC 决策

日期：2026-08-05

代码范围：所有修改只位于
`/home/asus/Research/Nav-axis-uturn`。母目录
`/home/asus/Research/Nav` 仅作为固定 episode、LingBot 代码和权重的只读来源。

## 1. 目标与边界

当前线上可靠性 router 的流程是：

```text
DINO 检索一个历史 anchor
  -> SIFT 局部匹配
  -> Essential Matrix RANSAC
  -> 通过才允许 memory controller
```

本实验测试能否用冻结 LingBot DINO 的 patch 对应关系和 retrieval 的时间邻域，训练一个
很小的 selective head，在极高或极低置信区间跳过部分几何验证。它不是新导航 policy，
也不是用 learned score 直接替换 retrieval。所有导出模型继续标记
`deployment_approved=false`。

## 2. Teacher、数据角色与泄漏控制

Teacher 完全复用已有 SIFT/Essential Matrix 判定：

- ratio-test matches >= 20；
- recoverPose inliers >= 12；
- inlier ratio >= 0.50。

模型看不到 Habitat pose、action、episode success、goal phase 或 GT gate。

场景角色固定为：

- 训练：`17DRP5sb8fy`、`1LXtFkjw3qL`、`1pXnuDYAj8r`、
  `Uxmj2M2itWa`；
- 开发评测：`e9zR4mvMWw7`、`rqfALeAoiTq`、`s8pcmisQ38h`、
  `yqstnuAEVhm`、`zsNo4HB9uLZ`。

新增 session 显式标记为 `*_train` 或 `*_evaluation`。只有训练角色参与 C 的选择、
leave-one-training-scene-out OOF 和 selective threshold 校准。开发场景只参与报告。

这 5 个场景已经被用于多轮开发比较，因此现在应称为 **development-heldout**，不能再
冒充最终 pristine test。后续正式结论必须在新的、从未查看过的场景上复验。

默认训练只加入 `cross_episode_train`。`within_episode_return_*` 是可选 stress test，
默认关闭，因为它的 query 是当前 return frame，而部署时 query 是 image goal；本机消融
显示它会改善 return stress，却没有稳定改善部署口径。

## 3. 固定依赖

- base teacher SHA256：
  `7aa916080eeec15ad505ca6b8c2349ac2383a9846ee1bb20ed704c3df350c779`；
- exact DINO CLS cache SHA256：
  `5d920cf32756c26a45a3c854f1e18103cb6980cbba23684815584130db7a8d7b`；
- LingBot weight SHA256：
  `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`；
- LingBot commit：`7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`。

Runner 会校验上述哈希、LingBot commit、Nav commit、CUDA、Python import、编译、
14 个 router 单元测试以及工作树中的任务文件。`LINGBOT_REPO` 必须显式指定，避免子目录
缺少 LingBot checkout 时误用不存在的默认路径。

## 4. Sparse teacher 修复

旧 full expansion 会对每个 query 的整条轨迹逐帧运行 SIFT/RANSAC，绝大多数标签随后
又因为只训练 DINO top-32 而被丢弃。新 builder：

1. 保留全轨迹 DINO cosine，供 temporal feature 使用；
2. 只对确定性的 hard top-K 计算几何 teacher；
3. 其余行写 `teacher_pass=-1`；
4. 诊断器强制检查实际选中的 top-K 全部是 0/1，任何漏标立即终止。

本机 sparse smoke 新增 3,769 个 candidate，只做 168 次几何检查，用时 2.53 s；完整
split audit、top-4/grid-4 诊断均通过。

## 5. 中等规模本机正式结果

配置：query stride 64、每 episode 最多 4 个 query、candidate stride 1、teacher
top-32、8x8 patch grid。包含 return stress 的构建共新增：

- 280 个 session；
- 79,606 个 candidate；
- 8,960 个实际几何标签，其中 1,749 positive；
- teacher 构建 122.45 s。

随后固定为部署一致的 **base + cross-only** 口径：

- 190 个训练 top-1 session；
- 59 个开发 top-1 session，其中 22 positive、37 negative；
- 6,080 个训练 hard pair、1,888 个开发 hard pair；
- 5,597 张唯一图像；patch 提取 65.19 s，关系特征 10.21 s。

正式 report：

```text
.diagnostics/patch_temporal_router_20260805/
  medium_sparse_top32_cross_only_v1/patch_temporal/report.json
```

开发 top-1 结果：

| 特征 | ROC AUC | AP | Brier | 自动判断 | FA / FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| global cosine | 0.9091 | 0.8940 | 0.1402 | 5/59 (8.5%) | 0 / 0 |
| temporal | 0.9079 | 0.9035 | 0.1141 | 10/59 (16.9%) | 0 / 0 |
| patch | 0.9423 | 0.9441 | **0.0842** | 14/59 (23.7%) | 0 / 1 |
| patch + temporal | **0.9472** | 0.9438 | 0.0944 | 10/59 (16.9%) | 0 / 0 |

因此 patch+temporal 确实比单一 cosine 提供了额外信息，但目前只在 10 个自动样本上
观察到零错误。零次错误的 95% 二项上界仍约为 25.9%，不能解释成安全性证明。

## 6. 为什么暂时不能替换 RANSAC

旧的 20-session 评测存在明显组成捷径：10 个 `revisit_b` 全是 positive，7 个 negative
全部来自 10 个 `paired_swap_probe`。扩大 query 类型后，模型仍有提升，但稳定性审计没有
通过：

- 五场景 cluster bootstrap 中，patch+temporal 相对 cosine 的 AUC 增量中位数为
  +0.0381，但 95% CI 为 `[-0.0019, +0.0995]`；AP 增量 CI 也跨过 0；
- leave-one-training-scene-out 重训时，去掉 `17DRP5sb8fy` 后 coverage 跳到 49.2%，
  并产生 1 个错误，说明阈值和模型受单一场景支配；
- 4 个 cross-fitted head 全票时 coverage 为 0；放宽到 3/4 票即出现 1 个错误。

根因是 DINO 主要编码语义/外观相似，浅层 head 只汇总 patch 和时间统计；它没有像
RANSAC 一样，在推理时求解几十个局部点能否共同满足同一个相机运动模型。4 个独立训练
场景也不足以阻止模型把墙面、走廊风格等场景特征当捷径。

## 7. 已复核的 reranker 与 top-K 对照

本轮没有重复已有 reranker 调参。旧结果在同一 59-session 部署口径上为：

- DINO top-1：22/28 positive 命中，78.6%；
- learned pair classifier：top-1 75.0%；
- pairwise patch+temporal ranker：top-1 71.4%；
- 简单 temporal rerank 只提高 MRR/top-5，没有提高 top-1。

6 个 DINO top-1 漏检的第一个有效几何 anchor 排名分别是：

```text
6, 11, 11, 16, 20, 23
```

因此 top-2 到 top-5 完全没有新增召回。直接按 DINO 顺序逐个验证的代价为：

| 最大验证数 | positive recall | 总体 session accuracy | 平均几何调用 | 估算均值延迟 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 78.6% | 89.8% | 1.00 | 24 ms |
| 6 | 82.1% | 91.5% | 4.14 | 100 ms |
| 11 | 89.3% | 94.9% | 7.19 | 175 ms |
| 23 | 100% | 100% | 13.93 | 338 ms |

时间 NMS 和分段取峰也已用 850 个新增 SIFT 标签验证。训练侧选择的 gap=8、最多 4 次
验证，在开发集只把 verified anchor 从 22/59 提到 23/59，救回的是一个 paired probe，
没有救回 cross-episode。gap=48 可救回一个 cross-episode，但在训练集退化，不能作为
可泛化参数。由此排除“top-K 只是被相邻重复帧挤满”作为主要原因。

## 8. Go / No-Go 与 HPC 计划

当前结论为：

- **Go**：sparse teacher、角色隔离、patch feature 和分组审计代码可提交；
- **No-Go**：不把 learned head 接入 live policy；
- **No-Go**：不把当前同 4 个训练场景的 full run 当成主要八小时任务；更多相关帧不能
  解决独立场景数不足；
- **No-Go**：不继续调 learned reranker、temporal NMS 或 inference threshold。

HPC 只读盘点确认无需重新生成 Habitat 数据：固定 SquashFS overlay 中有 54 个场景、
736 条 2-leg 和 1,208 条 3-leg episode；现有 flowgate feature cache 覆盖其中 50 个
场景，并且每条 cache 含逐帧 `dino_cls`。另外 4 个 overlay-only 场景为
`2t7WUuJeko7`、`D7G3Y4RVNrH`、`HxpKQynjfin`、`RPmz2sHmrrY`。

多场景实验采用冻结 manifest：

- 40 个 train scene；
- 10 个 development scene；
- 上述 4 个 scene 完全保留为 final-reserved，不进入当前任务；
- 每个 scene 固定按名称取前 2 条 episode，防止 episode 多的场景支配训练；
- 原 4 个本机训练场景强制留在 train，其余 46 个由固定 salt 的 SHA-256 排序划分，
  与图像和标签结果无关。

第一阶段生成 exact CLS/base geometry teacher，第二阶段只给 cross-episode hard top-K
生成 sparse teacher，第三阶段训练和审计 patch+temporal。只有 development 上的
scene-jackknife、误差数和 coverage 同时稳定改善，才冻结模型并启用 4 个 final-reserved
场景；再通过最终盲测后才进入闭环导航 A/B。

若仍训练 learned verifier，下一版应保留二维 patch 坐标和显式 correspondence/cost
volume，加入可微几何一致性，而不是继续增加 shallow logistic 的参数。SIFT/RANSAC 保留
为 fail-closed fallback。

## 9. 入口与提交状态

代码入口：

- `MemNavData/build_router_cross_episode_pairs.py`；
- `MemNavData/diag_patch_temporal_router.py`；
- `MemNavData/patch_temporal_router.py`；
- `MemNavData/run_patch_temporal_router_long.sh`；
- `MemNavData/slurm_patch_temporal_router_long.sbatch`。
- `MemNavData/router_multiscene_split_20260805.json`；
- `MemNavData/run_patch_temporal_router_multiscene.sh`；
- `MemNavData/slurm_patch_temporal_router_multiscene.sbatch`。

旧 Slurm 脚本只用于复现四场景 diagnostic，不再作为主要长任务。新任务先提交
`MODE=smoke`；只有 smoke 的依赖、scene split、输出和完整三阶段 report 都通过，才提交
`MODE=full` 的八小时上限任务。最终记录必须包含 code commit、split/overlay/权重哈希、
Slurm job ID、依赖预检和结果目录。
