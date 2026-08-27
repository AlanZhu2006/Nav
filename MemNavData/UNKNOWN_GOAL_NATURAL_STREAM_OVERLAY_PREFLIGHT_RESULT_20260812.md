# Unknown-goal Natural-stream：正式 Train Overlay 预检结果

日期：2026-08-12（CST）  
状态：本机机制与 HPC 数据兼容性预检均通过；**尚未证明 learned router 有效，不授权长时闭环评测或 40-scene 全量采集。**

## 1. 本轮究竟验证什么

目标不是重新测 SR，而是确认以下数据链在原生 NavDP 自然运动中成立：

```text
自然 observation / planning stream
  -> causal memory top-8 候选与连续 DINO/RANSAC 证据（shadow，不接管）
  -> recorded pose 对齐
  -> 离线 Habitat depth 重渲染
  -> task-aligned goal-surface co-visibility 标签
```

本轮不训练模型、不选择阈值、不比较控制器，也不让 router 或 adapter 改变动作。

## 2. 本机已完成的机制验证

- 一条真实 3-leg NavDP natural rollout：99 个 planning decisions、604 个 top-8 candidate trials；
- router takeover = 0，adapter takeover = 0；
- causal teacher 能把 MemNav frame 严格映射到 leg/step、recorded pose 与 RGB 内容哈希；
- Habitat depth-only 重渲染与记录视图对齐；
- 15/15 单元测试通过，Python compile 与 sbatch 语法检查通过。

边界：本机使用的是旧 smoke episode，不是正式 128.9 GB train overlay。因此它证明代码机制，
不能证明正式训练域中的标签分布。

## 3. HPC 一场景两 episode 预检

场景：`17DRP5sb8fy`。两个 episode 因 Slurm `--export` 的逗号解析问题被分别运行；没有重复
计算第一条。

| Job | Episode | 状态 | 时间 | A/B/C | 说明 |
|---|---|---|---:|---|---|
| 15627928 | episode_0000 | COMPLETED 0:0 | 3m53s | 1/0/0 | B 失败，C 因果截断 |
| 15628460 | episode_0001 | COMPLETED 0:0 | 3m33s | 1/1/0 | A/B/C 均有可用自然流，C 尝试但失败 |

另有 job 15627758 在 14 秒启动审计时退出：共享 Conda 的 `python` 是正常 symlink，旧检查器
误将其拒绝；没有进入模型加载或评测。修复后解释器允许解析到存在的普通可执行文件，源码、
权重、overlay 仍维持 fail-closed 哈希检查。

两条有效 episode 合计：

| 项 | 数量 |
|---|---:|
| natural frames | 818 |
| planning decisions | 104 |
| 有候选的 plans | 81 |
| candidate trials | 277 |
| candidate positive / negative / ambiguous | 62 / 215 / 0 |
| top-K support positive / negative / ambiguous plans | 20 / 84 / 0 |
| router / adapter takeover | 0 / 0 |
| missing trace/image、malformed trials | 0 |

两条 shadow contract 均通过，最终 artifact receipts 独立 `sha256sum -c` 全通过。使用的标签阈值
是 train-only Phase-B 继承的 positive `0.5`、strict negative `0.2`，没有在本轮数据上调节。

## 4. 最关键的结果：管线有效，但当前样本太容易

标签在两条合并后不退化，说明正式 overlay、自然流、pose 和 teacher 的接口成立。但按现役
geometry baseline 的原始阈值重新计算：

```text
matches >= 20, inliers >= 12, inlier_ratio >= 0.5
```

- 62/62 positive candidates 全部 hard-pass；
- 215/215 negative candidates 全部 hard-reject；
- 20/20 positive plans 的 raw-DINO top-1 已经是 positive；
- 61/61 有候选的 negative B plans 没有 geometry pass。

因此这两条只能证明采集/标签管线，**不能证明时序模型能修复现有 baseline**。直接拿这种数据
长训，大概率只会复刻 DINO + RANSAC 的已有规则。

## 5. 已有 train-only 静态证据给出的困难 strata

对 40 train scenes、80 factual episode 的 `goal_b_t0` / `goal_c_t0` 重新分层：

- Revisit C：73/80 的 top-8 含 positive；其中 12 条 positive 没有 positive RANSAC hard-pass；
- Revisit C：7 条 top-8 含 positive、但 DINO top-1 不是 positive；
- Novel B factual：8 条 strict no-match 却出现 geometry hard-pass；
- 三类合并后共有 21 条 unique hard episodes。

代表性 train-only 目标：

| Scene / episode | 困难类型 |
|---|---|
| `1pXnuDYAj8r / episode_0001` | C positive geometry miss + DINO top-1 wrong |
| `YVUC4YcDtcY / episode_0000` | C positive geometry miss + B strict-no-match geometry false support |
| `JeFG25nYj2p / episode_0001` | C positive geometry miss + DINO top-1 wrong |
| `5ZKStnWn8Zo / episode_0001` | B strict-no-match geometry false support；C 为 easy control |

这些筛选只读取 train-only causal teacher 与 geometry evidence；development、consumed 20-scene
pool 和 blind 均未参与。

## 6. 冻结后的下一步

现在不应提交 6–8 小时闭环 eval，也不应立即训练 temporal head。下一步是一个小型、定向的
hard-stratum natural-stream pilot：

1. 从上面的困难类别中选 3–4 个跨 scene episode，并保留当前 easy episode 作 control；
2. 只收 shadow stream，不接管、不测方法 SR；
3. 检查静态困难是否会在自然移动后的多个 plan 中持续，而不是只存在于 t0；
4. 比较 DINO、现役 hard geometry 与 temporal evidence，但不使用 A/B/C role 特征；
5. 只有困难正例覆盖提高、strict negative 不恶化，才扩大到 scene-grouped train collection；
6. 只有 nested scene-OOF 门通过，才值得做正式 paired closed-loop eval。

这把昂贵评测放在最后：当前阶段每条 episode 的实测总时间约 3.5–4 分钟，且主要是数据采集，
不是多臂 SR 评测。

## 7. 产物

- immutable source bundle：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/unknown_goal_natural_stream_20260812/source_preflight_v2_20260812T0030CST`
- source receipt SHA256：`9c7a5140fcf0623bb3187235a30245fcdac362506b956c50fb13ad03b99adc09`
- launcher SHA256：`23b95fe7dbc385714670b88c6cc768578e5dbc436dfef292520b33738a224594`
- episode_0000 run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/unknown_goal_natural_stream_20260812/train17drp_ep01_preflight_v2_20260812T0040`
- episode_0001 run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/unknown_goal_natural_stream_20260812/train17drp_episode0001_preflight_v2_20260812T0050`

