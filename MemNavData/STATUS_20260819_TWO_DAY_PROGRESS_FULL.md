# 2026-08-18--19 两日完整进展总账

更新时间：2026-08-19（Asia/Shanghai）  
状态：**截至本文写入时的最新 source of truth**。本文 supersede
`STATUS_20260819_MONOCULAR_DUAL_TIMESCALE.md` 中“CEC+mono 正在运行”的状态描述，并把
Final14、Pi3X、CEC 延迟、单目控制、外部评测、真机与论文工作区统一到同一证据边界下。

本文只把已经读取原始结果、完成配对复算或能由现有文件直接审计的事实写成结论。没有把
partial count、smoke、基础设施失败或跨运行的数字包装成正式方法结果。

## 1. 两天后的项目结论

项目已经从多条互相竞争的“记忆、方向、learned localizer、controller replacement”路线，
收敛为一个清楚的主方法和一个已经闭环跑通的部署扩展。

### 1.1 主方法：Certified Episodic Compass（CEC）

主问题不是“历史里有没有相似图”，而是：

> 因果在线历史在什么条件下有资格干预一个冻结的 ImageGoal controller？

CEC 把这个问题实现为 **proof-before-control / open-set action authorization**：

```text
actual-online causal RGB history + current ImageGoal
        |
        v
DINOv2 temporally-diverse top-8 proposal
        |
        v
SuperPoint + LightGlue correspondences
        |
        v
Fundamental-MAGSAC support/ranking
        |
        v
LingBot historical reference depth + PnP-RANSAC
        |
        v
atomic certificate
  inliers >= 16
  query/ref hull coverage >= 5%
  reprojection RMSE <= 2 px
        |
        +-- reject/error --> exact native ImageGoal NavDP
        |
        `-- accept -------> scale-free unit bearing
                            x fixed 2.5 m residual
                            + original goal image
                            --> frozen NavDP
```

关键职责边界：

- DINO 负责提出地址，不拥有控制权；
- SuperPoint/LightGlue/PnP/certificate 提供运行时 witness；
- certificate reject 只表示“当前历史证据不足”，不等价于语义 Novel；
- CEC 不输出全局路径或 metric waypoint，只允许一个二维 scale-free bearing 穿过边界；
- NavDP 仍是唯一生成轨迹和动作的 policy；
- reject 或 runtime error 时回到相同 policy state、相同 diffusion seed 的 native request。

### 1.2 部署扩展：单目双时间尺度读出

这两天新增的单目路线不是第二个 controller，也不是 LingBot/NavDP action-level MoE。它让同一
个 frozen LingBot streaming state 服务两个时间尺度：

```text
causal monocular RGB stream
        |
        v
one frozen LingBot streaming geometry state
        |
        +-- dense short readout -------------------------------+
        |   first 40 observations: zero depth                  |
        |   RGB-only first-40 scale receipt                    |
        |   frame >= 40: relative depth x frozen scale         |
        |   -> unchanged NavDP RGB-D observation encoder       |
        |                                                      |
        +-- sparse long readout ---------------------------+    |
            CEC retrieval + geometric proof              |    |
            -> certified bearing or abstain               |    |
                                                         v    v
                                  frozen NavDP goal encoder / decoder / critic
                                                         |
                                                         v
                                                   one trajectory
```

最准确的概括是：**one causal stream, two time scales, one frozen policy**。

## 2. 两天内新增结果一览

| 结果 | 人口与结果 | 当前证据等级 |
|---|---|---|
| Final14 mixed-role CEC | 21 histories / 10 scenes；CEC `28/42`，raw fixed `21/42`，`+8/-1`，`p=.0391` | 论文主结果；但低于预注册 28-history target |
| Final14 Revisit | native `4/21` -> CEC `20/21`，`+16/-0`，`p=3.05e-5` | 论文主结果 |
| Pi3X learned proof | Revisit `19/21`、all `27/42`，但未通过 non-inferiority 与 proof-safety gate | 有效 learned baseline；不能替代 CEC |
| CEC latency audit | first-use natural median/p95 `3.404/26.348 s`；cached update median `0.152 ms` | 严格实现审计 |
| exact eager cache | 一条 81-frame trace `11.7408 s -> 22.84 ms`，结构化输出完全相同 | 无损 microbenchmark；默认关闭 |
| Gate C raw-depth selection | 639 valid samples；raw 在五项 frozen 指标上优于 zero 与 6.02M adapter | 支持选择 raw interface；不是 SR |
| Gate D Novel-A closed loop | metric `30/40`、raw mono `27/40`、zero `23/40` | 完整配对闭环；只通过工程继续门 |
| CEC+mono composition | conditional B：`7/28 -> 27/28`，`+20/-0`，`p=1.91e-6` | 强组件组合结果；已消费 supported-Revisit population |
| HM3D mixed-role extension | 21 个 Goal-A success，但 0 个 history 被 materialize，query eval 未运行 | 基础设施失败；无方法结论 |
| GOAT autonomous adapter | 多轮 action-contract smoke 仍未完成第一 ImageGoal | 迁移诊断；不进入论文正结果 |
| Go2 双机部署 | 4090 hub + Jetson RGB-D dry-run、断链 fail-safe 均通过；未启动底盘 | 真机 readiness；无真机 SR |
| WACV paper workspace | 主文、补充材料、本地 PDF 与远端 commit `255b839` 已建立 | 写作基础已就绪，数字仍需更新 |

## 3. Final14：CEC 与 learned Pi3X 的正式结算

Final14 是这两天最重要的论文级结算。冻结人口来自 14 个 untouched MP3D scenes；最终同时
满足 standard Revisit 与 natural Novel 构造合约的是 21 histories、10 scene clusters。

预注册目标是 28 histories / 10 scenes，因此 scene target 达到，history target 未达到。
所有统计有效，但论文必须明确标记 **underpowered relative to the frozen target**，不能打开
结果后追加 episode 或放宽构造条件。

### 3.1 Natural-direction 主协议

| arm | Novel | Revisit | role-balanced all |
|---|---:|---:|---:|
| native | `7/21` | `4/21` | `11/42` |
| raw fixed bearing | `2/21` | `19/21` | `21/42` |
| geometry fixed | `9/21` | `18/21` | `27/42` |
| learned Pi3X spatial proof | `8/21` | `19/21` | `27/42` |
| **CEC** | **`8/21`** | **`20/21`** | **`28/42`** |

CEC 对 native：

- Revisit：`+16/-0`，风险差 `+76.19 pp`，exact McNemar `p=3.05e-5`；
- Novel：`+1/-0`；
- all：`+17/-0`，风险差 `+40.48 pp`，`p=1.53e-5`。

CEC 对 always-on raw fixed：

- Revisit 只多 `1` 条，`20/21 vs 19/21`，不能声称提升 Revisit ceiling；
- Novel 为 `8/21 vs 2/21`；
- all 为 `28/42 vs 21/42`，配对 `+8/-1`，`p=.0391`，scene-cluster CI
  `[+2.78,+31.25] pp`。

这确立了 CEC 最有价值的作用：**不是在高支持 Revisit 上继续挤上限，而是保留 memory
utility 的同时控制 unsupported Novel interference。**

CEC 对 geometry fixed 只有 `+2/-1, p=1`。因此论文不能把创新写成“更强 matcher”或“更高
PnP 精度”；应写成 proposal、witness、authority、narrow control interface 与 exact
fallback 构成的完整运行时合约。

### 3.2 Novel fallback 的准确口径

Natural Novel 共 21 条：

- 19/21 完全 reject；
- 这 19 条在 requested/returned diffusion seed、selected trajectory 与 executed trace 上
  精确等于 native；
- 另有 2/21 certificate accept/takeover。

所以可以写“在已评测 rejection 上 exact fallback”，不能写“零 Novel takeover”“形式化
safety guarantee”或“CEC 能完美判断 Novel/Revisit”。

### 3.3 Pi3X learned proof 的结果与停止原因

Pi3X learned arm 不使用 CEC 的 SuperPoint、LightGlue、LingBot depth、PnP 与 atomic
certificate，目标是测试 learned spatial proof 能否替代显式 proof。

它不是完全失败：

- Revisit `19/21`；
- all `27/42`；
- 相对 native Revisit 为 `+15/-0`，`p=6.10e-5`。

但它未通过预先冻结的 primary promotion gate：

- 对 CEC 的 scene-cluster interval 穿过 `-10 pp` non-inferiority margin；
- 479 个 accepted plan bearings 中，median error `8.05 deg`，p95 `149.52 deg`，69 个
  超过 `90 deg`；
- Novel proof-safety gate 未通过。

这说明“闭环 SR 看起来接近”不足以证明 learned witness 可靠。NavDP 有时能从错误 bearing
中恢复，因此最终成功会掩盖 proof 的长尾错误。Pi3X 保留为有价值的 learned comparison，
不替代 CEC，也不再为追平一条 SR 继续调 proof head。

正式文件：

- `MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md`；
- `.diagnostics/learned_relocalizer_20260817/final14_attempt7_formal_result_20260818/`；
- 远端：`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/`。

## 4. CEC 延迟：问题定位与无损优化

### 4.1 真正慢在哪里

Final14 的 raw-plan 复算证明：CEC 不是每个 replan 都重新跑完整重定位。每个新 goal 只有
第一次 certificate request uncached，之后固定 absolute goal pose 或 sticky abstention，只
更新 current-relative bearing。

Natural-direction：

| 路径 | n | median | p95 | mean | max |
|---|---:|---:|---:|---:|---:|
| 首次 uncached certificate | 42 | `3.404 s` | `26.348 s` | `7.820 s` | `45.219 s` |
| 后续 cached bearing update | 1346 | `0.152 ms` | `0.428 ms` | `0.217 ms` | `0.539 ms` |

长尾主要来自 selected anchor 较晚时，从 scale block 到该 anchor 顺序重放 dense LingBot
depth。selected-anchor index 与 Natural first-use latency 的 Pearson `r=.817`。

审计还发现旧 summarizer 把第一次 uncached latency receipt 在后续 cache hits 上重复统计，
Natural 中把真实 42 个 first-use 值扩成 1388 个。该问题只影响 latency 口径，不影响 SR、
certificate decision 或任何方法比较；正式冻结结果文件没有被回写。

### 4.2 已启用的低风险 exact cache

- selected-anchor final depth/confidence cache：同一 anchor 第二次查询
  `12.426 s -> 0.128 ms`；
- reference SuperPoint feature LRU：同一 immutable history frame 再匹配
  `230.57 ms -> 4.014 ms`；
- key 包含文件大小与 mtime，避免开发期路径复用造成 stale feature。

它们是默认轻量实现优化，不改变 proposal、ranking、PnP、certificate、bearing 或 controller。

### 4.3 exact eager dense writer

可选 `--certified_eager_depth_cache` 在历史写入时维护一个与 sparse NavDP stream 隔离的
dense state，并逐帧物化 causal depth/confidence。

81-frame 真实冻结轨迹：

- lazy first certificate：`11.7408 s`；
- eager lookup：`22.84 ms`；
- selected anchor、PnP、certificate checks、bearing 与公开结果逐项一致；
- speed ratio `513.99x`。

又在 121/161-frame、两个不同 scenes 上复验；三条轨迹共 363 帧，depth/confidence
逐元素最大绝对误差为 0，在线 sparse state SHA-256 相同。

代价：

- history ingest `+0.1764 s/frame` 左右；
- CUDA allocated `+7.59--7.67 GiB`；
- 73 帧 CPU depth cache约 `156.70 MB`。

因此 eager 保持显式部署开关、默认关闭。它适合目标切换延迟比写入吞吐更重要、且有约
8 GiB 额外显存的机器；24 GiB 设备上不能未经共驻审计与 Pi3X 等大模型同时打开。

被排除的伪无损优化：special-token prefix + suffix replay 和 multi-frame block replay。两者
虽然更快，但产生非零 depth/confidence 偏差，均已撤回。

正式文件：

- `MemNavData/FINAL14_CEC_CACHE_LATENCY_AUDIT_20260818.md`；
- `MemNavData/CEC_LATENCY_OPTIMIZATION_RESULT_20260818.md`。

## 5. 单目短程分支：为什么 learned Adapter 被 raw depth 淘汰

### 5.1 原始问题

主 CEC 结果中，CEC sidecar 使用单目历史，但 frozen NavDP controller 仍消费 Habitat metric
depth。为了缩小部署差距，这两天测试了两种短程几何接口：

1. 6.02M Geometry Token Adapter：把 LingBot tokens 翻译成 NavDP 的 `[128,384]`
   observation latent；
2. 更小的 raw interface：LingBot relative depth 乘一次因果尺度，直接送给 NavDP 原生
   RGB-D encoder。

### 5.2 两个必须先修的数据错误

这条线没有直接相信第一次漂亮或奇怪的结果，而是先排除了两个会使结论失效的问题。

#### 深度单位错误

`generate_twoleg.py` 保存 `uint16 = metres * 10000`。旧 loader 漏掉 `/10000`，导致 NavDP
把绝大多数 teacher depth 当作 `>5 m` 后裁零，使 zero depth 与所谓 metric teacher 近似。
该批旧 Gate A/B/smoke receipt 已明确作废。

#### whole-episode scale 泄漏

旧 cache 的 ground scale 来自整条 episode。即使 RGB/KV 只注入 prefix，frame 40 的 scale
仍可能使用未来帧。正式合约修为：只 replay causal RGB `0..39` 和相应 camera pose 一次，
冻结不可变 first-40 scale receipt；禁止 whole-episode cache 与未来帧 fallback。

这两个修复是单目结果可信的前提，不是为了追一个更好的数。

### 5.3 Gate C：raw interface 胜出

正式人口：40 scenes、160 episodes、640 planned samples；其中 639 valid，1 个预先固定的
all-zero teacher sample 被 outcome-blind attrition。32 train scenes / 511 samples，8
validation scenes / 128 samples，scene overlap 为 0。

| validation 指标 | zero depth | raw LingBot depth | 6.02M Adapter |
|---|---:|---:|---:|
| RGB-D token cosine error ↓ | 0.3024 | **0.1812** | 0.3871 |
| diffusion epsilon MSE ↓ | 0.01152 | **0.00591** | 0.01553 |
| critic Spearman ↑ | 0.6250 | **0.7672** | 0.5328 |
| critic top-1 agreement ↑ | 0.6016 | **0.7266** | 0.5391 |
| critic MSE ↓ | 0.2010 | **0.06918** | 0.4327 |

只读 action diagnostic 同方向：

- raw：selected endpoint L2 `0.569 m`，heading error `23.14 deg`；
- zero：`0.948 m / 39.32 deg`；
- Adapter：`1.016 m / 45.23 deg`。

结论不是“训练数据还不够所以再训更久”，而是 frozen NavDP 的 depth interface 已高度校准。
小 Adapter 同时近似 observation latent、denoiser 与 critic，容易产生相互冲突的
off-manifold representation；保留正确空间结构的 raw depth 反而更容易被 frozen decoder
消费。Adapter 在五项冻结指标上全部差于 zero，因此停止长训是由正式 gate 决定的。

Gate C 是策略可消费性证据，不是闭环 SR。

正式根目录：

`/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_geometry_20260818/formal_v5_causal_first40_attrition1`

## 6. 因果在线 mono depth contract

当前实现已经把 raw interface 做成 fail-closed、可审计的运行时合约：

1. observation frame `0..39` 向 NavDP 输出逐像素全零 depth；
2. 恰好收到 40 张 causal RGB 后，只 replay `0..39` 一次并冻结 scale receipt；
3. frame index `>=40` 才输出 `LingBot relative depth * scale_hat`；
4. scale invalid 时输出 zero，禁止 pooled/teacher/oracle fallback；
5. depth PNG 与当前 JPEG SHA-256 绑定，stale query fail closed；
6. `monocular_sidecar` 忽略上传的 simulator depth，并显式报告
   `metric_depth_sensor_consumed=false`；
7. 同一 RGB stream 同时保留给 CEC，不启动第二份 LingBot map。

核心代码：

- `MemNavData/monocular_depth_runtime.py`；
- `NavDP/baselines/memnav/policy_agent.py`；
- `NavDP/baselines/memnav/memnav_server.py`；
- `NavDP/baselines/navdp/navdp_server.py`；
- `MemNavData/eval_2leg_habitat.py`。

真实 41-frame prefix smoke、NavDP wire smoke 与专用三臂 Habitat smoke 均通过。smoke 只
验证 wire/state/scale 生命周期，不进入 SR。

## 7. Gate D：mono controller 的 N=40 闭环

Gate D 在已消费的 MP3D 20 scenes / 40 Novel-A episodes 上，同机同进程旋转三臂，仅改变
NavDP observation depth：

| arm | SR | mean SPL | mean path |
|---|---:|---:|---:|
| metric teacher | `30/40 = 75.0%` | 0.7327 | 7.71 m |
| raw LingBot first-40 | `27/40 = 67.5%` | 0.6307 | 9.14 m |
| zero depth | `23/40 = 57.5%` | 0.5455 | 9.90 m |

配对结果：

- raw vs metric：`+2/-5`，风险差 `-7.5 pp`，`p=.4531`，scene-cluster CI
  `[-20,+5] pp`；
- raw vs zero：`+6/-2`，风险差 `+10 pp`，`p=.2891`，CI
  `[-2.5,+22.5] pp`；
- zero vs metric：`+2/-9`，风险差 `-17.5 pp`，`p=.0654`，CI
  `[-32.5,-2.5] pp`。

部署审计：

- 40/40 raw episodes 不消费 simulator metric sensor；
- 40/40 到达 frame 40；
- 40/40 得到有效且实际被消费的冻结 scale；
- 2/40 scale 命中预注册 clamp；
- 20 scenes / 120 arm records 完整，独立 verifier `verified=true`。

冻结决定为 `continue_to_cec_on_monocular`，即通过工程继续门。但 10 pp paper
non-inferiority 没有通过，因为 raw-vs-metric CI 下界为 `-20 pp`。正确结论是：

> raw mono 对 zero 的点估计为 +10 pp，且与 Gate C 方向一致，足以进入组合实验；但 N=40
> 既不能排除对 zero 的零效应，也不能排除相对 metric teacher 的较大损失。

不得写“mono 与 metric 等价”或“主 CEC 结果已经完全单目化”。

正式结果：

`/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_raw_depth_gate_d_20260819/formal_20260818T171542Z_b8d2ffb5/POSTHOC`

## 8. CEC + mono composition：最新完成结果

这是旧状态文档之后真正新增的正式结算。

### 8.1 冻结问题与人口

问题是：

> 在完全相同的 raw-mono Goal-A history 和完全相同的 raw-mono NavDP controller 下，打开
> CEC sparse long-horizon readout 是否改善 supported Revisit Goal-B？

人口是 Fresh160 immutable manifest 的每场景前两条，20 scenes / 40 episodes。该 population
已经被以前的实验消费，因此本实验是严格的 **component compatibility / composition
experiment**，不是新的 held-out confirmation。

两臂：

- `raw_native`：monocular sidecar + native ImageGoal；继续写同一 LingBot stream，但禁止
  memory takeover；
- `raw_cec`：相同 monocular sidecar，只打开已经冻结的 CEC 与 fixed-2.5m bearing。

每条 episode 的 raw-mono Goal-A 只运行一次，再由 SHA 绑定 trace 给两臂重放。两臂共享同一
loaded LingBot/NavDP server pair，不读取 role 标签。

### 8.2 完整结果

Goal-A 成功 28/40，因此 conditional B 分母为 28：

| arm | B given shared raw-mono A | ITT joint |
|---|---:|---:|
| raw native | `7/28 = 25.0%` | `7/40 = 17.5%` |
| raw CEC | `27/28 = 96.43%` | `27/40 = 67.5%` |

Conditional contrast：

- `+20/-0`；
- 风险差 `+71.43 pp`；
- exact McNemar `p=1.9073486e-6`；
- scene-cluster 95% CI `[+53.57,+88.0] pp`。

ITT contrast：

- `+20/-0`；
- 风险差 `+50 pp`；
- scene-cluster 95% CI `[+35,+65] pp`；
- McNemar 同为 `p=1.9073486e-6`。

审计：

- 20/20 array tasks 完成；
- 80 arm records 完整；
- CEC requests `391`、accepts `391`、runtime failures `0`；
- primary summary 与 independent verifier 完成；
- 独立 distance artifact 从 80 个原始 `final_dist_B` 按 `<1.0 m` 复算全部 success，
  `verified=true`、`authorized=true`、`known_gap=null`。

调度记录中 distance job `16010883` 显示 FAILED，是因为同一路径的 distance verification
输出已经存在，脚本 fail-closed 拒绝覆盖；不是复算失败。现有只读 artifact 已完成 80 条
raw-distance 重算，其 SHA-256 为
`9f191c0c00523a3725f70db37506aea10da7e216e9187f041bcae2a4f2e18533`。

权威目录：

`/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_cec_composition_20260819/formal_20260819T055600Z_624f9fa9`

### 8.3 这个结果能说明什么

它强力支持：

- raw mono 的短程 depth readout 与 CEC 的长程 proof/bearing 没有结构性冲突；
- 相同 mono controller 下，supported Revisit 的主要差异确实来自 CEC；
- CEC 能把 raw-mono Goal-A 成功后的 B 从 25% 提到 96.43%；
- 全程 query rollout 没有消费 simulator metric depth。

它不能支持：

- mono-CEC 的 Novel safety；这批全是高支持 Revisit；
- certificate abstention/fallback 的新增证据；391/391 requests 全部 accept，说明本人口没有
  覆盖 reject path；
- fresh scene generalization；人口已消费；
- mono 与 metric controller 的直接因果差；Gate D 与该实验不是同一运行，不能跨机器或
  跨进程直接相减；
- 完全单目主论文结论；Final14/HM3D headline 仍是 metric RGB-D NavDP controller。

还要注意：本次 shared raw-mono A 为 `28/40`，Gate D 独立运行中的 raw 为 `27/40`。正式
composition 只能使用本次同进程 shared-A 分母 28，不能为了统一数字跨运行替换。

## 9. HM3D：已成立的 transfer 与未完成的 mixed-role extension

### 9.1 已成立 Revisit transfer

既有 held-out HM3D val9 结果仍有效：

- native conditional Revisit B：`7/21`；
- CEC：`19/21`；
- paired `+12/-0`，`p=.000488`；
- gains 分布在 8/9 scenes；
- joint `7/36 -> 19/36`。

它证明 Revisit utility 跨 MP3D -> HM3D transfer，但没有 paired Novel query，因此不证明
外部 mixed-role safety。

### 9.2 mixed-role extension 的真实状态

8 月 18 日提交了同 9 scenes 的 Novel/Revisit role-hidden 扩展：construction array
`15947671` 的 9 个 tasks 全部完成；population seal `15947673` 随后报
`HM3D mixed-role population is empty`，下游 eval/summary/verifier 没有运行。

这不是“HM3D 上不能构造 Novel/Revisit”的科学结论。原始 inventory 显示：

- source scenes：9；
- source episodes：36；
- saved native Goal-A successes：21；
- materialized histories：0。

21 个 eligible success histories 全部在 materialization 阶段遇到相同类型的 asset alias
错误。例如代码查找：

```text
.../asset_alias/HaxA7YrQdEC.basis/HaxA7YrQdEC.basis.glb
```

`.basis` scene identity 被重复当作目录层与文件 stem，导致 `FileNotFoundError`。其余 15 条
是预期的 `native_a_failed` attrition。因为 21 条成功 history 都未被 materialize，role-pair
builder 没有尝试任何 query，正式 SR 为“未产生”而不是 0。

权威 run root：

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_mixed_role_20260818/hm3d_mixed_role_20260818T105403Z`

下一次只能做路径解析的不可变 repair，不能改变场景、query contract、threshold 或读取
query outcome 后筛样。

## 10. GOAT：为什么暂时不再作为正向主评测

### 10.1 已有正式结果

34-scene frozen sequential-Revisit 评测中，official GOAT 与 CEC 都为 `4/34`，paired
`+0/-0`。但 CEC 没有执行一次 motion override：5 次 certificate accept 全部发生在
official policy 已经输出 `SUBTASK_STOP` 的同一步，冻结协议要求原样保留 STOP。

因此该结果是 `degenerate_noop_no_executed_intervention`，不能解释为 bearing 无效，也不能
当作 CEC 的外部正结果。

### 10.2 这两天的 NavDP autonomous adapter 诊断

为了检查能否把整套 architecture 迁入 GOAT，又逐层修了：

- ImageGoal/current camera intrinsic 与颜色接口；
- 0.25 m / 30 deg 离散动作适配；
- collision recovery；
- 低-critic lateral search 的原子执行；
- STOP 与 terminal matcher fail-closed。

单条诊断 episode 上，修复确实恢复了运动能力；atomic-search 300-step smoke 行走约
16.4 m、触发 38 次 search，终点仍距目标约 3.50 m，CEC 两次 terminal search 均因
`precheck_fundamental_inliers` 拒绝，ImageGoal success 仍为 0。更改 critic threshold 的
后续 smoke 只属于 post-hoc interface diagnosis，不是可报告方法比较。

MP3D 与 GOAT 的输入控制合约并不相同：GOAT goal image 是 object-centric instance photo，
goal/current camera 的高度、俯仰、FOV 与 NavDP 训练分布差异大；GOAT 动作为 0.25 m/30 deg
离散原子，而 NavDP 原实验使用更细连续轨迹执行。修接口能恢复一部分局部运动，不能自动
弥补 object-centric goal conditioning 分布差异。

冻结决定：

- 不把 GOAT smoke 写成 benchmark score；
- 不在 held-out GOAT 上继续调 threshold；
- GOAT 保留为 external limitation / adapter stress test；
- 当前论文外部正证据使用 HM3D Revisit transfer，而不是 GOAT。

## 11. Unitree Go2 双机真机部署

### 11.1 架构

```text
Unitree Go2 / Jetson Orin NX                  RTX 4090 workstation
───────────────────────────                  ───────────────────────
D435i synchronized RGB-D
  -> existing NavDP ROS adapter --SSH--> unified CEC hub
  <- 24-point local trajectory              |- MemNav causal buffer
  -> local trajectory tracking              |- DINO/LightGlue/PnP/CEC
  -> depth fail-closed                       `- frozen NavDP
  -> 0.35 s watchdog / gamepad priority
  -> SportClient.Move()
```

TopoFocus 只用于参考双机职责与 fail-closed 设计，没有复用其 model、planner 或 navigation
method。4090 只输出局部轨迹，没有 Unitree SDK 权限；Jetson 保留传感器、速度限幅、depth
急停、watchdog 与遥控器优先权。

### 11.2 已完成

- 统一 hub/adapter 单元测试 21 passed；
- MemNav、NavDP、hub 在 4090 同时加载；
- 端口 `8888/18888/18889` 只监听 loopback；
- Jetson 到 4090 RTT 约 2.3 ms，SSH tunnel 实际连通；
- D435i + disabled adapter 的无运动 dry-run 成功；
- 20 秒 38 个状态全部 `enabled=false`，0 error、0 非零命令；
- inference p50/p95/max `0.638/0.681/0.760 s`；
- 杀 tunnel 后进入 fail-safe，观察到 `vx=0,wz=0`，恢复后重新规划；
- 整个验证没有启动 `go2_cmd_bridge`。

### 11.3 尚未完成

- 相机光轴到 Go2 `base_link` 的 bearing 符号校准；
- GPU 独占条件下的长时间 p99；
- 启动底盘后的系绳低速直行；
- 多目标 closed-loop 真机 SR；
- 将 simulator 中的 raw-mono depth extension 换入当前 D435i RGB-D 真机链。

所以当前是 deployment readiness，不是真机导航结果。

截至本文写入时，本机 realworld 三服务仍在运行：`127.0.0.1:8888/18888/18889`；RTX 4090
约占 5.46 GiB、GPU utilization 0%。如果短期不做真机，应显式停止以释放显存，不能把
“空闲但已加载”误写成正在评测。

完整文档：`MemNavData/REALWORLD_GO2_DUAL_MACHINE_DEPLOYMENT_20260818.md`。

## 12. 论文工作区进展

已建立独立 WACV 2027 Applications Track workspace：

```text
paper/
  main.tex
  sec/
  tables/
  figures/
  supplementary.tex
  supp/
  EVIDENCE_LEDGER.md
  FIGURE_PLAN.md
  STYLE_GUIDE.md
  TEMPLATE_PROVENANCE.md
```

当前标题：

> Certified Episodic Compass: Geometrically Verified Memory for Continual
> ImageGoal Navigation

已完成：

- 使用官方 WACV 2027 style 建立匿名 main/supplementary；
- 主文、补充材料与 bibliography 均可编译，本地已有 `main.pdf` 与
  `supplementary.pdf`；
- 参考 `AlanZhu2006/Memnav_Paper` 的写作节奏、图表位置和 appendix discipline，但未复制
  技术内容或文字；
- 参考模板 provenance 已单独记录；
- 远端 `AlanZhu2006/Memnav_Paper` main HEAD 为
  `255b839694aabf26385f3d438393ffd32a77b25e`。

当前 paper draft 已写入 Final14、HM3D、NNR、Pi3X、negative results 与方法主体，但
`paper/EVIDENCE_LEDGER.md` 仍把 CEC+mono 写成“running”。在把新结果放进正文前必须先更新
ledger，并把它限定为 consumed composition evidence；不能把 `27/28` 当成新的 Final14，
也不能把 headline CEC 改写成 fully monocular。

## 13. 目前证据分层

### A. 可以进入论文主表/主结论

1. Final14 role-free mixed-role CEC vs native/raw；
2. HM3D held-out Revisit transfer；
3. actual-online N--N--R；
4. 原始 geometry memory `4/40 -> 19/40` 作为早期 causal evidence；
5. 方向 oracle `28/40 -> 40/40`，但只能写 mechanism upper bound。

### B. 可以进入 ablation、analysis 或 deployment section

1. Gate C raw-vs-zero-vs-adapter；
2. Gate D metric/raw/zero closed-loop；
3. CEC+mono `7/28 -> 27/28` composition；
4. CEC latency 与 exact cache；
5. Train40 certificate precision/recall/FPR；
6. Fresh160 high-support ceiling 与 online observability。

### C. 只作工程 readiness / limitation

1. Go2 no-motion dry-run；
2. GOAT autonomous adapter smoke；
3. HM3D mixed-role alias-path failure；
4. Replica constructibility failure；
5. HPC launch、dependency、overlay、H200 runtime incidents。

## 14. 当前不能说什么

- 不能说 CEC 有形式化 safety guarantee；
- 不能说 CEC 从不在 Novel 上接管；Final14 有 2/21 Novel takeover；
- 不能说 CEC 在 Revisit 上显著超过 raw 或 geometry；
- 不能说 Pi3X learned proof 已替代 explicit certificate；
- 不能说 GOAT 验证了本方法；
- 不能说 HM3D 已验证 mixed Novel/Revisit safety；该任务没进入 query eval；
- 不能说 raw mono 与 metric RGB-D non-inferior；
- 不能把 Gate C offline metric 当 SR；
- 不能把 CEC+mono 叫 fresh confirmation 或 Novel safety test；
- 不能说 primary CEC headline 是 fully monocular；
- 不能把 training-free 写成“不包含 pretrained learned models”。CEC 没有任务特定训练，
  但 DINO、SuperPoint/LightGlue、LingBot 与 NavDP 都是 pretrained models；
- 不能说已有真机 SR；目前只有无运动与 fault-injection readiness。

## 15. 两天工作带来的最重要认知变化

### 15.1 论文贡献不应写成“把几个模块拼起来”

DINO、LightGlue、PnP 和 NavDP 都不是单独的新模块。真正可识别的贡献是：

1. 把历史复用表述为开放集 action authorization，而不是 closed-set role classifier；
2. 将 proposal、geometric witness、control authority 明确分开；
3. 只暴露 scale-free bearing，限制历史模块的控制能力；
4. 在同状态/同 seed 下定义 exact fallback；
5. 用 mixed-role 配对实验直接测 utility/interference frontier。

### 15.2 当前 learned 失败不是简单由“PT1 数据少”解释

Pi3X 与 Geometry Token Adapter 分别暴露了两个不同问题：

- learned proof 可以取得高 SR，但长尾角误差没有被 SR 暴露；
- learned latent bridge 同时拟合多个 frozen downstream functions，会破坏已校准的 depth
  manifold，即使参数量和优化 loss 都能下降。

现有证据不支持“把 PT1 全部塞进去长训就会自然解决”。若以后重启学习路线，目标应是更
窄的 proof confidence/calibration，而不是端到端替换 retrieval、matcher、PnP、depth 与
controller。

### 15.3 单目扩展现在是有效架构，不是 headline 已替换

Gate C、Gate D 与 composition 已经形成完整链条：

```text
raw depth 更可消费
    -> raw mono 能闭环运动
    -> CEC 能在同一 mono controller 上恢复 supported Revisit
```

但 Final14/HM3D 的 mixed-role与 transfer headline 尚未在 full-mono 条件下重做。因此最诚实
的论文结构是：CEC 为主方法，mono dual-timescale 为部署扩展；只有新的 mixed-role/fresh
或真机证据到位后，才考虑把 fully monocular 放进标题或第一贡献。

## 16. 下一步优先级

### P0：更新论文数字与证据边界

- 把 CEC+mono 完整结果写入 `paper/EVIDENCE_LEDGER.md`；
- 在方法/实验中增加 causal first-40 mono contract；
- 在结果中把 Gate C、Gate D、composition 分成三个不同 estimands；
- 不跨运行相减 Gate D 与 composition；
- 保留 headline CEC 的 metric-controller边界。

### P0：修复 HM3D mixed-role 资产别名

这是当前最便宜、最直接的外部 Novel safety 缺口。只允许修复 `.basis` 目录/stem 解析并创建
新的 immutable repair bundle；先 outcome-blind 复算 materialized history 数，再决定是否
提交 query eval。不得改变 9 scenes、query contract 或 certificate threshold。

### P0：真机最小闭环

按冻结顺序完成：

1. bearing 左/右静态符号校准；
2. GPU 独占 dry-run 与 p99；
3. tunnel/watchdog 再验证；
4. 0.5--1.0 m 系绳低速直行；
5. 最小 Novel/Revisit pair，而不是立即做大规模真机 benchmark。

### P1：单目 mixed-role 的严格归因

2026-08-19 已冻结并提交 Final14-style query-depth factorial：

1. mono native；
2. mono raw-fixed bearing；
3. mono CEC；
4. metric native；
5. metric CEC。

旧 Final14 已消费，所以它只能作为 same-history controller-depth attribution，不能冒充 fresh
confirmation。原始 Goal-A 仍来自 metric-depth NavDP；五臂只在相同 causal RGB replay 后改变
query controller 的 depth source / CEC authorization。21/21 histories 均长于 40 帧，因此 mono
query 全部必须使用 active causal first-40 scale，不允许 bootstrap。

Attempt 6 当前提交链：

- smoke：`16020635`，history 0、80-step guard；
- formal：`16020636`，21-history array，依赖 smoke 成功；
- summary/verifier：`16020637`，依赖整个 formal array 成功；
- formal root：`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_mono_factorial_20260819/formal_20260819T124820Z_5690569a`；
- immutable source receipt：`5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216`。

Attempt 1--5 均未产生 formal method result：依次暴露了 PT1 overlay 遗漏、preflight 字段
推导错误、Habitat Python 3.9 不支持 `zip(strict=True)`、只读 bundle pycache 写入、以及提交
脚本未纳入 bundle identity 五个执行问题。Attempt 6 已把 overlay、全部 21 个 source parquets、
Python 3.9 实际函数测试、临时 pycache 和 incident ledger 纳入 source receipt；详见对应
`FINAL14_MONO_FACTORIAL_ATTEMPT*_INCIDENT_20260819.json`。
最初 Attempt 6 的三个 2-hour pending jobs 在任何输出创建前取消，并用相同 bundle/root 以
1-hour GPU TimeLimit 重提；这只是 outcome-blind backfill 调度修正，见
`FINAL14_MONO_FACTORIAL_ATTEMPT6_SCHEDULER_AMENDMENT_20260819.json`。

它的主检验是 mono CEC 相对 mono native / mono raw fixed；metric 两臂及 difference-in-differences
只用于拆分“mono 局部控制损失”和“CEC 授权收益”，不是新的独立确认。

### P1：系统延迟报告

- 在实际部署 GPU 上报告 lazy/eager 的端到端 p50/p95/p99；
- 同时报告显存、CPU cache 与 per-frame ingest cost；
- 不用孤立 microbenchmark 代替完整 LingBot+NavDP+CEC 共驻延迟。

### 明确停止

- 不继续 Geometry Token Adapter 长训；
- 不在 Final14 或 GOAT held-out 上调 threshold；
- 不把 active-glance、X-NavDP controller、graph rescue 拉回主线；
- 不继续追求 candidate-free full-history Transformer 替代显式 retrieval；
- 不为了“看起来更 learned”牺牲已经测得的 proof reliability。

## 17. 当前运行状态

截至 2026-08-19 20:54（Asia/Shanghai）本文审计时：

- Final14 mono Attempt 6 smoke `16020635` 因 Priority 排队；formal `16020636` 与 summary
  `16020637` 正确处于 Dependency；
- CEC+mono array `16009201` 的 20 个 tasks 全部 `COMPLETED`；
- summary `16009253` `COMPLETED`；
- HM3D mixed construction `15947671_[0-8]` 全部完成，seal `15947673` 因空 materialized
  population fail closed；
- 本机 RTX 4090 utilization 0%，约 5.46 GiB 被 realworld MemNav/NavDP/hub 占用；
- 本机没有 Habitat closed-loop eval 在运行。

2026-08-20 补充：Final14 mono formal `16020636` 已完成 `19/21` histories；indices 2、4
在脚本开始前被系统于 `gh011` 取消，没有创建输出。缺失任务以相同 immutable bundle/root
补交为 `16026422_[2,4]`，新的 summary/verifier 为 `16026423`。在两条补齐并通过独立
verifier 前，不报告部分 SR。详见
`FINAL14_MONO_FACTORIAL_ATTEMPT6_MISSING_TASK_REPAIR_20260820.json`。

## 18. 精准代码与结果入口

| 责任 | 路径 |
|---|---|
| CEC server/runtime | `NavDP/baselines/memnav/memnav_server.py` |
| proposal/witness/cache/lifecycle | `NavDP/baselines/memnav/policy_agent.py` |
| certificate 与 bearing contract | `MemNavData/certified_relocalization_runtime.py` |
| role-free mixed evaluator | `MemNavData/eval_shared_online_role_pairs.py` |
| Final14 formal result | `MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md` |
| Final14 latency audit | `MemNavData/FINAL14_CEC_CACHE_LATENCY_AUDIT_20260818.md` |
| exact cache microbenchmark | `MemNavData/CEC_LATENCY_OPTIMIZATION_RESULT_20260818.md` |
| mono depth runtime | `MemNavData/monocular_depth_runtime.py` |
| Gate C protocol/status | `MemNavData/MONOCULAR_DUAL_TIMESCALE_EXPERT_PROTOCOL_20260818.md`, `MemNavData/STATUS_20260819_MONOCULAR_DUAL_TIMESCALE.md` |
| Gate D evaluator | `MemNavData/eval_mdtec_raw_depth_gate_d_habitat.py` |
| Gate D summary/verifier | `MemNavData/summarize_mdtec_raw_depth_gate_d.py`, `MemNavData/independent_verify_mdtec_raw_depth_gate_d.py` |
| mono composition evaluator | `MemNavData/eval_mdtec_monocular_cec_composition_habitat.py` |
| mono composition summary/verifier | `MemNavData/summarize_mdtec_monocular_cec_composition.py`, `MemNavData/independent_verify_mdtec_monocular_cec_composition.py` |
| Final14 mono factorial protocol | `MemNavData/FINAL14_MONO_FACTORIAL_PROTOCOL_20260819.md`, `MemNavData/final14_mono_factorial_protocol_20260819.json` |
| Final14 mono factorial runner | `MemNavData/run_final14_mono_factorial_episode.py`, `MemNavData/run_final14_mono_factorial_history.sh` |
| Final14 mono factorial summary/verifier | `MemNavData/summarize_final14_mono_factorial.py`, `MemNavData/independent_verify_final14_mono_factorial.py` |
| Final14 mono active submission receipt | `MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT6_SUBMISSION_RECEIPT_20260819.json` |
| Final14 mono execution incidents | `MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT1_OVERLAY_INCIDENT_20260819.json` through `MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT5_BUNDLE_IDENTITY_INCIDENT_20260819.json` |
| HM3D completed Revisit result | `MemNavData/HM3D_HELDOUT_VAL10_FORMAL_RESULT_20260817.md` |
| HM3D mixed submission | `MemNavData/HM3D_MIXED_ROLE_SUBMISSION_20260818.md` |
| GOAT formal limitation | `MemNavData/GOAT_SEQUENTIAL_REVISIT_FORMAL_RESULT_20260815.md` |
| GOAT autonomous adapter | `MemNavData/goat_autonomous_multigoal_pilot.py` |
| Go2 hub | `MemNavData/realworld_cec_hub.py` |
| Go2 deployment doc | `MemNavData/REALWORLD_GO2_DUAL_MACHINE_DEPLOYMENT_20260818.md` |
| architecture source of truth | `MemNavData/ARCHITECTURE_20260819_PAPER_SOURCE_OF_TRUTH.md` |
| paper evidence ledger | `paper/EVIDENCE_LEDGER.md` |
| paper source | `paper/main.tex`, `paper/sec/`, `paper/tables/`, `paper/supp/` |

## 19. 最终判断

两天前，项目的主要风险是“CEC 看起来像有效的工程组合，但 learned replacement、部署延迟、
单目闭包与外部证据都没有收口”。两天后，状态变成：

1. Final14 mixed-role 已经给出 CEC 相对 raw memory 的显著 utility/interference 优势；
2. learned Pi3X 有真实 utility，但显式暴露出 proof long-tail，合理地没有替代 CEC；
3. CEC first-use latency 已被定位，并有 decision-equivalent 的缓存方案；
4. learned Geometry Token Adapter 被正式 gate 淘汰，raw causal mono depth 成为更小的部署接口；
5. raw mono 不仅通过 offline gate，也完成 N=40 Novel-A closed loop；
6. CEC+mono 在同一 controller/history 上把 supported Revisit 从 `7/28` 提到 `27/28`；
7. 真机双机链已达到无运动与断链安全 readiness；
8. 论文工作区已经成形。

项目现在不是“效果都不太好”。更准确的说法是：**主 Revisit 方法效果已经很强，剩余短板
集中在 fully-monocular headline 的 fresh/mixed-role 证据、外部 Novel safety、系统 first-use
latency 代价和真机闭环。** 接下来应围绕这四个明确缺口补证据，而不是重新打开已经被负结果
淘汰的 learned decoder、active scan 或 controller replacement 分支。
