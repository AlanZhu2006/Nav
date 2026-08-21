# CEC 延迟优化与严格等价性结果

日期：2026-08-18（Asia/Shanghai）  
性质：实现级 microbenchmark；不修改 retrieval、geometry rank、PnP、certificate
阈值、bearing、NavDP controller 或任何冻结 SR 结果。

## 1. 结论

CEC 的 20--45 秒长尾来自每个新 goal 第一次认证时，为 selected anchor 从第 8
帧开始顺序重放 dense LingBot depth。此次得到两类严格优化：

1. 默认启用、低风险的跨 goal 缓存：
   - reference SuperPoint feature LRU；
   - immutable selected-anchor 的最终 depth/confidence cache。
2. 默认关闭的低查询延迟模式 `--certified_eager_depth_cache`：
   - history 写入时维护一个与 sparse NavDP stream 隔离的 exact dense stream；
   - 每帧物化 causal depth/confidence；
   - goal 到来时直接读取 selected anchor，不再做全历史 replay。

真实 81 帧冻结轨迹上的端到端结果：

| 路径 | LightGlue→PnP→certificate→bearing | 决策 |
|---|---:|---|
| legacy lazy replay | `11.7408 s` | accepted |
| eager dense writer | `0.02284 s` | accepted |
| 比值 | `513.99x` | 全部结构化输出相同 |

两路的 selected anchor、PnP pose、certificate checks、scale-free bearing 和最终公开
结果逐项一致。该数字是单轨迹实现 microbenchmark，不是新的科学样本或 SR 声明。

## 2. Eager 模式的代价

同一输入、同一 seed、同一 81 帧轨迹：

| 指标 | lazy | eager | 增量 |
|---|---:|---:|---:|
| history ingest | `9.663 s` | `22.665 s` | `+13.002 s` |
| post-scale 每帧摊销 | — | — | `+0.1764 s/frame` |
| CUDA allocated after ingest | `10.274 GiB` | `17.862 GiB` | `+7.588 GiB` |
| CPU depth cache（73 帧） | `0` | `156.70 MB` | `2.147 MB/frame` |

因此 eager 不是“免费提速”，而是把一次不可接受的 goal-switch 长停顿摊到历史写入。
它适合有约 8 GiB 额外显存、且重视目标切换响应时间的部署；不应在 24 GiB 上与
Pi3X 等大模型未经共驻审计直接同时启用。

## 3. Sparse NavDP 状态没有被改变

Eager 使用 snapshot/restore 在同一个冻结 LingBot 上交替维护两个状态：

- primary：原有 flow-gated sparse navigation stream；
- secondary：只服务 certificate depth 的 dense stream。

对 81 帧完成独立状态摘要，lazy 与 eager 的以下内容完全一致：

- 81 个 DINO CLS；
- 81 个 camera poses；
- sparse anchor/camera frame indices；
- anchor K/V、camera K/V；
- `_last_tokens` 与全部 `_last_agg`。

两者统一 SHA-256：

```text
bd08bf5ce35f9034475ddbcac26b1a101a34af5783f0a0d49fc110d63a939d7a
```

这验证了额外 dense stream 没有污染 frozen NavDP 的在线输入状态。

### 3.1 三条不同长度、不同场景轨迹的等价性复验

在读完第一条结果后，又预先选取两个不同 MP3D scene 的冻结 actual-online RGB
history，分别检查正常写入和 eager 写入。三条轨迹的结果如下：

| scene / anchor | frames | legacy anchor depth | eager lookup | depth/conf | online state hash | eager ingest 增量 | CUDA allocated 增量 |
|---|---:|---:|---:|---|---|---:|---:|
| `gxdoqLR6rwA / 80` | 81 | `11.60 s` | `0.219 ms` | 逐元素相同 | 相同 | `+12.88 s` | `+7.588 GiB` |
| `pLe4wQe7qrG / 120` | 121 | `19.39 s` | `0.266 ms` | 逐元素相同 | 相同 | `+20.81 s` | `+7.650 GiB` |
| `yqstnuAEVhm / 160` | 161 | `27.09 s` | `0.215 ms` | 逐元素相同 | 相同 | `+28.80 s` | `+7.671 GiB` |

合计覆盖 363 个输入帧、三个 scene、三个 anchor 深度。所有轨迹均满足：

- `eager_writer_error = null`；
- depth 与 confidence 的 shape、有限性和每一个数组元素完全相同，最大绝对误差为 0；
- DINO CLS、camera poses、sparse frame indices、全部 live K/V 与最后 token/aggregate
  组成的状态 SHA-256 在 normal/eager 间一致。

因此第一条轨迹的结论不是偶然，但这仍属于实现等价性验证，不增加论文的独立导航
episode 数，也不能把三条 microbenchmark 当作 SR 泛化结果。

## 4. 默认跨 goal cache

### 4.1 Selected-anchor 最终 depth cache

同一 anchor 的第二次查询：

- legacy：`12.426 s`；
- exact cache：`0.128 ms`；
- depth/confidence 逐元素相同；
- 每个 anchor 约 `2.147 MB` CPU 内存。

但 Fresh20 double-Revisit 中，17 条同时产生 B/C anchor 的 episode 没有一条 B、C
选择同一 anchor。因此它对当前短 benchmark 的命中率有限，主要服务更长的 lifelong
场景和同地点重复目标，不能拿理论 speedup 代替实际端到端分布。

### 4.2 Reference SuperPoint feature LRU

同一 immutable history frame 对第二个 goal 再次匹配：

- `230.57 ms → 4.014 ms`，`57.45x`；
- 842 个 matches 及全部坐标/score 逐元素相同；
- LRU 默认上限 64 个 reference frames，文件大小与 mtime 参与 key，避免开发期路径
  复用返回 stale feature。

Fresh20 的 B/C top-8 只有 2/17 条存在候选交集，平均交集 `0.118`，所以该缓存同样是
lifelong 优化，不是当前 3-leg 分数的主要来源。

## 5. 已排除的“伪无损优化”

### 5.1 Special-token prefix + 32-frame suffix

想法是缓存 dense special K/V，后续只重放 LingBot 的 32 帧 full-token window。
真实轨迹结果：

- `12.46 s → 5.18 s`，约 `2.41x`；
- depth 最大误差 `0.2967`；
- confidence 最大误差 `1.4337`。

原因：窗口中每一帧的 K/V 又递归依赖它之前的完整窗口；只有 special prefix 无法恢复
窗口第一帧的原始状态。该实现已撤回，没有进入 runtime。

### 5.2 多帧 causal block replay

block size 2/4/8/16 的速度为 legacy 的 `1.45--1.89x`，但 depth 最大误差分别约
`0.105/0.106/0.163/0.223`，不是 precompute 使用的逐帧 streaming 运算。全部排除。

这两个负结果说明不能用“理论上同为 causal”替代真实 decision-equivalence 检查。

## 6. 当前启用策略

- 默认 runtime：
  - 保留原始 lazy first-anchor replay；
  - 启用 exact selected-anchor result cache；
  - 启用 bounded reference SuperPoint LRU；
  - 所有既有论文评测语义不变。
- 低延迟部署实验：显式传入 `--certified_eager_depth_cache`；
  - server status 会报告开关、失败原因与已缓存帧数；
  - eager 路径 OOM/异常时停止继续构建，已有 exact depth 保留，缺失 anchor 自动回退
    legacy replay；
  - 目前不设为默认，也不需要为验证它重新跑长闭环 SR。

## 7. 代码与证据

- runtime：`NavDP/baselines/memnav/policy_agent.py`；
- server flag：`NavDP/baselines/memnav/memnav_server.py`；
- SuperPoint LRU：`MemNavData/lingbot_pnp_localization.py`；
- microbenchmark：`MemNavData/benchmark_cec_dense_cache_equivalence.py`；
- 单元测试：
  - `MemNavData/test_policy_agent_graph.py`；
  - `MemNavData/test_lingbot_pnp_localization.py`；
  - `MemNavData/test_audit_final14_cec_latency.py`；
- 43 个相关测试通过。

关键原始结果：

- `.diagnostics/cec_latency_optimization_20260818/e2e_eager_seed17_anchor80_goal79_v2.json`；
- `.diagnostics/cec_latency_optimization_20260818/live_state_normal_seed17.json`；
- `.diagnostics/cec_latency_optimization_20260818/live_state_eager_seed17.json`；
- `.diagnostics/cec_latency_optimization_20260818/resource_normal_seed17.json`；
- `.diagnostics/cec_latency_optimization_20260818/resource_eager_seed17.json`；
- `.diagnostics/cec_latency_optimization_20260818/anchor_cache_and_block_anchor80.json`。
- `.diagnostics/cec_latency_optimization_20260818/pLe_anchor120_normal.json`；
- `.diagnostics/cec_latency_optimization_20260818/pLe_anchor120_eager.json`；
- `.diagnostics/cec_latency_optimization_20260818/yq_anchor160_normal.json`；
- `.diagnostics/cec_latency_optimization_20260818/yq_anchor160_eager.json`。

## 8. 下一步边界

三条不同长度、不同 scene 的冻结轨迹已完成严格等价性复验，足以支持当前实现判断：

- 不需要为了这个不改变决策的优化重跑 20-scene SR；
- 默认启用两个轻量 exact cache；
- eager 继续保持实验开关，除非部署机器明确接受约 7.6 GiB 额外显存和约
  `0.18--0.19 s/frame` 写入开销；
- 若论文要报告真实系统 latency，还需在目标部署卡上做 LingBot/NavDP/Habitat（以及
  若启用则 Pi3X）的共驻测量，不能用本次孤立 microbenchmark 替代。
