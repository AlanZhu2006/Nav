# 几何 Teacher 蒸馏与可靠性 Router：跨场景验证报告

日期：2026-08-04

工作目录：`/home/asus/Research/Nav-axis-uturn`

母目录 `/home/asus/Research/Nav`：只读取已有 episode、LingBot 代码和权重，未修改。

## 1. 这次要验证什么

当前自动 router 使用：

```text
DINO 检索候选
  -> SIFT + essential matrix 几何验证
  -> 连续两个 planning step 的候选均可靠
  -> latch memory controller
```

它的优势是 fail-closed 和跨场景可靠，缺点是首次验证需要 CPU 特征匹配。为同时改善
速度和方法完整性，本实验测试一个保守的 selective cascade：

```text
已有的 DINO CLS(goal, anchor)
  -> 极低置信度：自动拒绝 memory
  -> 中间区域：继续调用原几何 verifier
  -> 极高置信度：自动接受 memory
```

learned head 只允许在 scene-disjoint 校准中零经验错误的两端绕过几何；不确定样本继续
使用原 verifier。这个设计不会用导航成功标签、Habitat GT pose 或 episode phase 作为
推理输入。

## 2. 特征、teacher 与模型

### 2.1 输入特征

使用线上 retrieval 已经产生的冻结 DINOv2-L CLS，不新增视觉 backbone。对单位化后的
goal 向量 `g` 和 memory 向量 `m` 构造对称关系：

```text
[abs(g - m), g * m, cosine(g, m)]          # 2049 维
```

对称性避免“同一地点”因 query 顺序改变。对照模型只使用一个 cosine 标量。

### 2.2 几何 teacher

标签完全复用当前部署判定：

- SIFT，最多 4000 个特征；
- Lowe ratio `0.75`；
- episode 相机内参；
- essential matrix，RANSAC 概率 `0.999`、阈值 `1.5 px`；
- positive：`matches >= 20`、`inliers >= 12`、`inlier_ratio >= 0.50`。

因此这是“模仿线上几何可靠性”的实验，不是用 Habitat 位姿偷偷产生 GT gate。

### 2.3 分类和安全校准

- relation head：standardization + L2 logistic regression，`C=0.01`；
- cosine baseline：standardization + logistic regression，`C=1.0`；
- 类别加权；
- 阈值概率来自 inner leave-one-scene-out 预测，而不是模型见过的训练样本；
- 每个自动区间至少需要 100 个校准样本；
- accept 阈值必须高于所有校准 negative；
- reject 阈值必须低于所有校准 positive；
- 不满足时禁用该侧，回退几何。

“校准集零错误”只是保守的经验规则，不是统计保证，所以即使离线通过也必须再做未见
场景和闭环导航 A/B。

## 3. 数据划分

### 3.1 最初可行性实验

- 5 个未见 MP3D 场景；
- 10 条 2-leg episode；
- 每条构造 `revisit_b` 和 `paired_swap_probe` 两个 query session；
- 20 个 session、4731 个 goal-candidate pair；
- leave-one-scene-out。

`paired_swap_probe` 不是强行当作 negative。同场景的成对 episode 可能真的看到同一区域，
最终类别始终由几何 teacher 决定。

### 3.2 固定 train 4 scenes -> test 5 unseen scenes

训练/校准场景，共 32 条 episode：

- `17DRP5sb8fy`
- `1LXtFkjw3qL`
- `1pXnuDYAj8r`
- `Uxmj2M2itWa`

完全 held-out 测试场景，共 10 条 episode：

- `e9zR4mvMWw7`
- `rqfALeAoiTq`
- `s8pcmisQ38h`
- `yqstnuAEVhm`
- `zsNo4HB9uLZ`

合计：42 条 episode、84 个 session、22,267 个 pair、15,041 张唯一图像。模型拟合和
阈值校准只使用前 4 个场景；后 5 个场景只报告一次测试结果。

训练侧已有数据从母目录只读加载：

```text
/home/asus/Research/Nav/memnav_viz/validate_gated_extra/mp3d_2leg
```

新增两场景 episode 和所有输出都位于子目录的 `.diagnostics/`，该目录不进入 Git。

## 4. 结果

### 4.1 最初 5-scene LOO

| 指标 | 2049-D relation | cosine-only |
|---|---:|---:|
| pair ROC-AUC | 0.817 | **0.860** |
| pair AP | 0.602 | **0.768** |

relation selective cascade：

- 自动接受：0；
- 自动拒绝：626；
- 几何调用：4105 / 4731 = 86.77%；
- false accept：0；
- false reject：38；
- 真正会影响部署的 20 个 top-1 session 全部仍需几何。

它已经提示高维 CLS relation 会过拟合场景外观，且省下的主要是非 top-1 candidate，
不是线上真正验证的 anchor。

### 4.2 固定 train-4 -> unseen-test-5

训练侧共有 17,536 个 pair；测试侧共有 4731 个 pair，其中 teacher positive 1157、
negative 3574。

| held-out pair 指标 | 2049-D relation | cosine-only |
|---|---:|---:|
| ROC-AUC | 0.806 | **0.875** |
| AP | 0.594 | **0.778** |
| Brier，越低越好 | 0.256 | **0.174** |
| false-positive rate @ 0.5 | 40.5% | 31.4% |
| false-negative rate @ 0.5 | 15.6% | 15.1% |

严格 selective cascade：

| held-out 指标 | relation | cosine-only |
|---|---:|---:|
| 自动接受 | 0 / 4731 | 0 / 4731 |
| 自动拒绝 | 0 / 4731 | 0 / 4731 |
| 几何调用率 | **100%** | **100%** |
| false accept / reject | 0 / 0 | 0 / 0 |

为确认结论不是 `min_calibration=100` 人为造成，又把最低尾部数量降到 1 做了只读
敏感性诊断：

- relation 的校准尾部实际只有 19 个 reject 和 10 个 accept；到 held-out 后自动接受
  85 个 pair，其中 **10 个是 false accept**，只节省 1.8% 几何调用却引入 11.8% 的
  accepted error；
- cosine 的校准尾部为 43 个 reject、10 个 accept，但 held-out 没有任何样本达到这些
  极端阈值，覆盖率仍是 0%；
- 因此把最低数量改成 20、10 或 1 都不能得到兼具安全性和有效覆盖率的 cascade。

20 个 held-out top-1 session 中：

- 10/10 `revisit_b` 的 top-1 通过几何；
- 3/10 `paired_swap_probe` 也确有几何重叠，所以总 positive 是 13/20；
- relation top-1 AUC：0.758；
- cosine-only top-1 AUC：0.901；
- 两种 selective cascade 都是 20/20 回退几何。

### 4.3 时间

- exact DINO CLS：15,041 张图像共 178.4 s；这是离线批量提取，线上 CLS 本来就由
  retrieval 产生，不是新 router 开销；
- 离线几何 teacher：22,267 对共 289.5 s，13.00 ms/pair；该诊断预计算了 query
  SIFT，因此不能直接当作线上首次调用延迟；
- NumPy portable relation head：8.59 us/pair。

在 10 个真实 Revisit top-1 pair 上另测线上 verifier：

| 路径 | mean | median |
|---|---:|---:|
| 首次完整 SIFT/RANSAC | 24.283 ms | 20.467 ms |
| 相同 goal-anchor cache hit | 0.068 ms | 0.070 ms |

cache hit 平均约快 355 倍，且 matches、inliers、ratio 与首次结果逐项完全一致。当前两次
确认的意义是两个 planning step 都检索到稳定 anchor；相同图像对不需要重复求解同一
几何问题。

## 5. 可验证结论与部署决定

1. **不是单纯训练样本太少。** 从 5 场景 LOO 扩展到固定 4 场景训练、5 个新场景测试
   后，高维 learned relation 仍低于原始 cosine。
2. **AUC 好不等于能安全路由。** cosine top-1 AUC 达 0.901，但正负分布尾部在场景间
   仍重叠；一旦要求校准场景零错误，两侧阈值都只能禁用。
3. **当前不能宣称 learned router 提高了 SR 或速度。** 它没有进入 live policy，也没有
   闭环成绩；导出的 JSON 明确标记 `deployment_approved=false`。
4. **保留 geometry router。** 它仍是目前防止相似走廊误 loop closure 的关键安全层。
5. **启用结果缓存。** 缓存不改变任何路由结果，只消除同一 goal-anchor 在第二次确认时
   的重复 CPU 工作，因此不会引入新的泛化假设。

换句话说，这轮实验成功否定了一个看起来便宜、但跨场景不可靠的替换方案，同时给现有
几何方案做了无损加速。正式导航 SR 应保持不变；变化只应体现在第二次验证延迟。

## 6. 下一步最有价值的研究方向

若继续追求“更学习化但仍可靠”，不应继续调 CLS logistic 的 C 或阈值：

1. 使用 retrieval 已有的 DINO patch，构造 reciprocal patch correspondence、空间一致性
   和视差统计；这些特征与 essential-matrix teacher 的判据更直接；
2. 学习 selective utility，而非只模仿 overlap：只有“memory 相对冻结 NavDP 确实改善
   动作/价值”时才增加 memory residual；不可靠时严格恢复 base policy；
3. 扩展到至少数十个训练场景，并以完整 `Novel -> Revisit -> Novel` 三段闭环验证
   false activation、SR、SPL 和延迟；
4. 若 patch head 仍没有安全 coverage，就保留几何 verifier。约 20--25 ms 的一次性 CPU
   代价相对 LingBot/NavDP 规划很小，不值得用错误 loop closure 换取。

这比直接把 router 替换为黑盒分类器更有创新价值：核心问题变成“带 base-policy
保护的 selective memory utility”，而不是普通的 Novel/Revisit 二分类。

## 7. 代码与产物

代码：

- `MemNavData/reliability_router.py`
  - 对称 DINO relation；
  - 零经验错误 selective 阈值；
  - 无 sklearn 依赖的 portable NumPy logistic head；
- `MemNavData/diag_distill_geometry_router.py`
  - exact LingBot DINO 特征提取；
  - 多 episode root；
  - LOO 或固定 held-out scenes；
  - nested scene-disjoint 校准；
  - relation/cosine 对照和 top-1 报告；
- `NavDP/baselines/memnav/policy_agent.py`
  - per-episode `(goal_md5, anchor)` 几何结果缓存；
  - positive 和 negative 都缓存；
  - 每次 reset 清空；
- `MemNavData/eval_2leg_habitat.py`
  - 将 cache hit 和首次/当前验证耗时写入 plan 日志；
- `MemNavData/test_reliability_router.py`
- `MemNavData/test_retrieval_verification_cache.py`

诊断产物，不进入 Git：

```text
.diagnostics/router_distillation_20260804/full_exact/
.diagnostics/router_distillation_20260804/fixed_train4_test5_exact/
```

portable JSON 文件名刻意为：

```text
diagnostic_router_not_for_deployment.json
```

## 8. 依赖与完整性检查

本机诊断环境：

| 依赖 | 版本 |
|---|---|
| Python | 3.10.20 |
| NumPy | 1.26.0 |
| OpenCV | 4.9.0 |
| scikit-learn | 1.7.2 |
| PyTorch | 2.8.0+cu128 |
| pandas | 2.3.3 |
| pyarrow | 24.0.0 |

模型依赖被脚本 fail-fast 固定为：

- LingBot repository commit：`7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`；
- `lingbot-map-long.pt` SHA256：
  `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`；
- exact DINO state 必须包含 344 个 tensor，否则终止；
- embedding cache 同时核对完整 path 列表和权重 SHA，不匹配时拒绝复用；
- 重复 episode identity、缺失图像/内参、单一类别和非 scene-disjoint 数据均会终止。

最终提交前应重新执行：

```bash
PYTHONPATH=. /home/asus/miniconda3/envs/memnav/bin/python \
  -m unittest discover -s MemNavData -p 'test_*.py'

PYTHONPATH=InternNav:. /home/asus/miniconda3/envs/memnav/bin/python \
  -m unittest discover -s InternNav/tests/unit_test -p 'test_memnav*.py'

git diff --check
```
