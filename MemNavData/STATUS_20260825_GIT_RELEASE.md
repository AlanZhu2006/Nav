# Nav / MemNav 项目权威总账与 Git release（2026-08-25）

更新时间：2026-08-25 19:06（Asia/Shanghai）

目标远端：`AlanZhu2006/Nav`

目标分支：`feat/memnav-graph-blind-20260806`

当前已推送基线：`fa44c44`（`Release full-mono CEC lifelong and deployment stack`）

本文是本次 Git 提交的导航入口。它统一描述当前方法、记忆实现、正式结果、负结果、代码
入口、真机边界、HPC 状态与下一步。若数字冲突，优先级始终是：

```text
independent raw-file verifier
  > sealed summary / protocol
  > 本文
  > 运行时口头汇报或 partial count
```

## 1. 当前项目到底是什么

项目当前研究的问题是：

> 一个只接收因果单目 RGB 的持续 ImageGoal 导航系统，如何利用跨目标历史，同时避免错误
> 历史干扰一个冻结的扩散导航策略？

当前主方法是 **Certified Episodic Compass（CEC）**。它不是重新训练 NavDP，也不是构建
一张供 planner 搜索的显式地图。它让历史先提出 Revisit 假设，再要求该假设提供可审计的
几何 witness；只有 certificate 通过时，历史才获得一个很窄的控制权限：向 frozen NavDP
提供 scale-free bearing。失败、拒绝或异常时逐动作回退 native NavDP。

论文架构短语冻结为：

> **one causal stream, two time scales, one frozen policy**

方法概念冻结为：

> **proof-before-control / open-set action authorization**

## 2. 当前记忆不是显式建图，也不只是 LingBot KV cache

最准确的名称是 **retrieval-addressed latent episodic memory**。它有四层状态。

### 2.1 可持久化的长期情景记忆

- 实际执行过程中因果写入的 RGB/JPEG 历史；
- 每帧的时间索引、SHA-256 和 goal-session/candidate-ceiling 元数据；
- 每帧冻结 DINO descriptor 供数百帧范围的内容寻址，近期窗口另保留 dense patch feature；
- 真机两阶段协议中，首次 survey 的 JPEG dataset 是 durable memory，第二次运行通过重放
  恢复内存状态。

这部分回答“过去哪一帧最可能对应当前目标”。它不是 occupancy grid、点云或 node-edge
拓扑图。

### 2.2 LingBot 在线隐式几何状态

- frozen LingBot GCT 以 streaming 方式处理同一条 RGB 流；
- KV cache 保留近期完整 patch 上下文，旧帧压缩为 anchor/special tokens；
- 每帧产生 camera/pose encoding，并可按需物化相对 depth/confidence；
- 最近窗口维持连续相对几何，历史 anchor 的 dense depth 可通过精确 replay 或 cache 恢复。

它回答“当前相机与历史 anchor 在同一个隐式坐标支架里如何相对排列”，不是一张显式全局
地图。KV cache 单独无法在数百帧中可靠内容寻址，也不能提供 LightGlue/PnP 所需的原始
参考图；candidate-free GCT 已被闭环结果 `5/20` 对 DINO-addressed `18/20` 否决。

### 2.3 运行时几何 proof

```text
current ImageGoal + causal RGB history
  -> DINO temporally-diverse top-8 proposal
  -> SuperPoint + LightGlue correspondences
  -> Fundamental-MAGSAC support/ranking
  -> LingBot historical anchor depth + PnP-RANSAC
  -> atomic certificate
```

当前 certificate 冻结门限：

- PnP inliers `>=16`；
- query/reference hull coverage 均 `>=5%`；
- reprojection RMSE `<=2 px`。

通过后得到目标在 LingBot 隐式坐标中的 pose；用最新 current pose 相减，输出归一化
`[forward, left]`。LingBot 单目 translation norm 不被当作可靠 metric distance。

### 2.4 极窄控制接口

```text
certificate accept
  -> unit bearing [forward,left]
  -> frozen 2.5 m residual
  -> original ImageGoal + PointGoal token
  -> frozen NavDP diffusion decoder + critic

certificate reject/error
  -> exact native NavDP
```

NavDP 仍是唯一生成 trajectory/action 的 policy。CEC 不输出全局路径，不在两个 action
experts 中投票，当前主方法也不使用 graph rescue。

## 3. 单目双时间尺度架构

同一个 frozen LingBot stream 有两个 readout：

```text
causal RGB stream
  |
  +-- short range
  |     frames 0..39: zero depth
  |     exact replay causal frames 0..39 once
  |     predicted floor height + known camera height
  |     -> immutable scale receipt
  |     frame >=40: relative depth x frozen scale
  |     -> unchanged NavDP RGB-D observation encoder
  |
  +-- long range
        RGB/DINO episodic retrieval
        + LightGlue/LingBot-depth/PnP witness
        -> certified bearing or abstain
  |
  -> one frozen NavDP goal encoder / diffusion decoder / critic
```

严格合约：

1. 只允许 actual causal RGB；
2. first-40 scale 冻结一次，此后不读 future frames；
3. 不允许 Habitat depth/pose、pooled/oracle scale 或第二条 LingBot stream；
4. replay 前后 snapshot/restore streaming cache；
5. RGB、depth payload 与 scale receipt 通过 SHA 绑定；
6. scale 无效时 fail closed；
7. 真机可保留独立 depth collision safety，但导航 policy 本身仍为 monocular。

这里的“training-free”只表示没有为 CEC 训练或微调新任务模型。NavDP、LingBot、DINO、
SuperPoint 和 LightGlue 都是冻结的预训练模型，不能写成“完全没有 learned model”。

## 4. 当前已经成立的核心结果

### 4.1 Final14：proof-before-control 主结果（MP3D，metric-depth 控制变量）

21 histories / 10 scene clusters，Natural Novel 与 Revisit 严格配对：

| arm | Novel | Revisit | 合计 |
|---|---:|---:|---:|
| native | `7/21` | `4/21` | `11/42` |
| raw fixed memory | `2/21` | `19/21` | `21/42` |
| geometry fixed | `9/21` | `18/21` | `27/42` |
| learned Pi3X | `8/21` | `19/21` | `27/42` |
| CEC | `8/21` | `20/21` | `28/42` |

CEC 对 native：Revisit `+16/-0`，`p=3.05e-5`；全部查询 `+17/-0`。CEC 对 raw：
`+8/-1`，`p=.0391`。优势主要来自减少 unsupported Novel interference，不是继续提高已经
接近饱和的 Revisit ceiling。Natural Novel 中 `19/21` 被拒绝且逐动作 exact fallback，另有
`2/21` 接管，因此不能声称零 Novel takeover。

### 4.2 Final14 mono factorial：单目 query control 没有削弱授权效果

- mono CEC `28/42`，mono native `10/42`，paired `+19/-1`，`p=4.0e-5`；
- mono CEC 相对 mono raw 为 `+5/-0`，增益全部来自 Novel；
- mono CEC `28/42`，metric CEC `26/42`，只有 `+2/-0, p=.5`，不能写 mono superior；
- 两种 depth 下 certificate decision byte-identical；
- 该实验的 Goal-A history 来自原 metric Goal-A causal RGB replay，不是 full-mono Goal-A
  population。

### 4.3 MP3D supported-Revisit Full-Mono composition

20 scenes / 40 episodes，共享 mono Goal-A history：

- conditional B：mono native `7/28` -> mono CEC `27/28`；
- paired `+20/-0`，`p=1.9e-6`；
- ITT：`7/40 -> 27/40`；
- 391/391 certificate accepts，0 runtime failures，0 simulator metric-depth reads。

这是高支持 Revisit 组合证据，不是 mixed-role safety 证据。

### 4.4 HM3D reused-scene Full-Mono mixed-role

8 histories / 7 scenes，16 queries：

| arm | Novel | Revisit | 合计 |
|---|---:|---:|---:|
| mono native | `2/8` | `0/8` | `2/16` |
| mono raw fixed | `3/8` | `8/8` | `11/16` |
| mono CEC | `2/8` | `7/8` | `9/16` |

CEC 对 native `+7/-0, p=.015625`；Novel `8/8` reject 并 exact fallback。但 CEC 没有超过
raw fixed（`9/16 vs 11/16`）。该结果建立完整 RGB-only integration，不建立 fresh-scene
generalization 或 CEC-vs-raw superiority。

### 4.5 Fresh HM3D Full-Mono mixed-role

28 outcome-blind histories / 21 scene clusters，56 queries，scene 从未进入此前 metric-controller
实验：

- mono native `17/56`；
- mono CEC `32/56`；
- paired `+16/-1`，`p=2.7e-4`；
- scene-cluster CI `[+15.5,+37.5] pp`；
- Revisit：`9/28 -> 24/28`，同样 `+16/-1`；
- CEC 与 raw fixed 的 Revisit 都是 `24/28`；
- Novel：native/CEC `8/28`，raw fixed `4/28`，说明 always-on memory 仍有 interference；
- 6,262 个 Goal-A 与 6,555 个 query plan receipts 均为零 simulator-depth read；
- independent verifier `verified=true`。

这是目前最重要的完整单目外部场景结果。它证明 fresh-scene Revisit utility 和 role-free
fallback，不证明 Novel 本身被解决。

### 4.6 多目标与持续记忆

- actual-online N--N--R：Goal-C `5/19 -> 16/19`，paired `+11/-0`，
  `p=.0009766`；
- 原三臂 lifelong 18 episodes / 7 clusters：forced reject `4/18`、initial-leg-only
  `6/18`、all-prior `11/18`；all-prior 对 forced `+8/-1, p=.0391`；
- 这建立了端到端 retained-history treatment 的单调结果；但旧 arms 在 C 段独立执行，
  因而不能把 B2 差异完全归因于相同物理前缀下的历史范围；
- 当前 shared-C 实验正冻结同一 A/B/C prefix，再只在 B2 分叉，以移除该混杂。正式 B2
  结果尚未产生。

### 4.7 机制与次级结果

- 最早 geometry memory：`4/40 -> 19/40`，`+15/-0, p=6.1e-5`；
- Fresh160 high-support Revisit：native `27/120`、raw `106/120`、CEC `112/120`；CEC 对 raw
  `+9/-3, p=.146`，不显著；
- HM3D metric-controller Revisit：`7/21 -> 19/21`，`+12/-0, p=.000488`；
- Novel oracle bearing：`28/40 -> 40/40`，`+12/-0, p=.000488`，但 oracle 不可部署；
- Final14 support spectrum：CEC authorization 随 support 为
  `2/21 -> 19/21 -> 21/21`；supported 档 raw/CEC 近饱和，差异主要在 unsupported 档。

## 5. 已严格停止或降级的路线

| 路线 | 结果 | 当前定位 |
|---|---|---|
| wider top-K | `18/40 vs 18/40, p=1` | 候选宽度不是瓶颈 |
| learned CDEC | top-1 接近 geometry，但 actionable `115 vs 122` | 不进入长闭环 |
| candidate-free GCT | `5/20 vs 18/20` | 长程必须显式内容寻址 |
| learned residual | `76/80 vs 74/80, p=.5` | 不推广 |
| Pi3X proof | SR 接近 CEC，但 bearing 长尾与 Novel gate 失败 | learned baseline |
| active glance | 最好 `25/40`，native `31/40` | 停止原地扫描 |
| X-NavDP | `21/26 vs 20/26, p=1` | controller 不是主瓶颈 |
| graph rescue | 多个正式人口无稳定增益 | 从主方法移除 |
| GOAT adapter | certified success `0/20`，目标/相机/到达合约不匹配 | 暂停，不作反证 |
| Geometry Token Adapter | 6.02M adapter 五项 Gate-C 指标均劣于 zero | 选择 raw mono depth |

## 6. 本次 Git release 的新增代码

### 6.1 CEC 与真实持续生命周期

- `policy_agent.py` 新增只读 goal support、goal-session replay、RoPE cap fail-closed、
  scale-free terminal direction 与 first-40 local scale receipt；
- `memnav_server.py` 暴露 goal-session replay、candidate support 和 local pose 端点；
- `realworld_cec_hub.py` 升级为显式 `memory_recording -> revisit_query` 两阶段状态机；
- goal candidate 在 recording 时注册但不写入 memory；
- `begin_revisit` 原子重建 NavDP 短期 FIFO，并在真正 query 起点后才冻结 goal session；
- query/recording 可以多次切换而保持长期历史。

这里必须区分两个代码层级：Nav 仓库保留论文/仿真侧的研究实现；当前真机权威运行栈维护在
独立仓库 `/home/asus/Research/Memnav_Realworld`。后者已经进入 protocol-v3、两次运行的
sealed survey/formal lifecycle 和 direct-bearing-v2，不能把 Nav 内较早的 hub 副本当作
Jetson/RTX 部署真值。

### 6.2 Full-Mono lifelong/shared-C

- result-blind A/B 构造、actual mono factual-B collection、prefix construction；
- factual C 只运行一次并 hash-seal；
- `all_prior / initial_leg_only / forced_reject_native` 从同一 A/B/C 起点比较 B2；
- aggregation 与 independent raw-file verifier；
- zero-history attrition、dependency repair、content-addressed bundle 和 exact Slurm receipts；
- deferred launcher 现在从 sealed population 读取精确数组长度，禁止再把 260 source 上界
  当成实际 evaluation population。

### 6.3 Controller portability

- GNM、NoMaD、ViNT、iPlanner、ViPlanner 统一经过 CEC accept/fallback contract；
- RGB-only controller 接收经过 SHA 绑定的 certified history anchor；
- PointGoal controller 只能消费 proof-bound normalized bearing；
- forced-reject arm 运行相同 proof，但不允许 takeover；
- collision/FIFO short reset 不再错误清除 active goal identity；
- shared-C replay 恢复 goal-session boundary，而不追加 memory 或运行 diffusion。

这些代码建立接口可移植性，不等价于五个 controller 已经有可比较的正式 SR。

### 6.4 HPC 与审计

- immutable source-bundle selftest；
- 每臂 `contract_dry_run`；
- 分区/QOS lint 与 `PYTHONDONTWRITEBYTECODE`；
- exact-index repair，不覆盖已有完整结果；
- shared SSH socket/master 双重检查；
- aggregate 与 verifier 从 raw CSV/plans/compute identity 重算；
- array 现在必须由 sealed population 精确确定长度。

## 7. 精准代码入口

| 职责 | 入口 |
|---|---|
| CEC causal memory / proof / cache | `NavDP/baselines/memnav/policy_agent.py` |
| MemNav HTTP runtime | `NavDP/baselines/memnav/memnav_server.py` |
| LingBot streaming backbone | `NavDP/baselines/memnav/policy_backbone.py` |
| temporal candidate contract | `NavDP/baselines/memnav/router_candidates.py` |
| certificate 与 bearing 数学 | `MemNavData/certified_relocalization_runtime.py` |
| LingBot depth + PnP | `MemNavData/lingbot_pnp_localization.py` |
| causal mono scale/depth | `MemNavData/monocular_depth_runtime.py` |
| 2-leg evaluator | `MemNavData/eval_2leg_habitat.py` |
| 3-leg evaluator | `MemNavData/eval_3leg_habitat.py` |
| lifelong evaluator | `MemNavData/eval_shared_online_lifelong_nnr.py` |
| shared-C pure contract | `MemNavData/lifelong_shared_c_contract.py` |
| HM3D lifelong pure contract | `MemNavData/hm3d_fullmono_lifelong.py` |
| HM3D shared-C B2 evaluator | `MemNavData/eval_hm3d_lifelong_shared_c_b2.py` |
| controller portability hub | `MemNavData/cec_controller_portability_hub.py` |
| controller contract/proxy | `MemNavData/controller_portability_contract.py`, `controller_portability_proxy.py` |
| 真机 RTX hub | `MemNavData/realworld_cec_hub.py` |
| 真机 goal candidate scorer | `MemNavData/score_realworld_revisit_goal.py` |
| 尺度无关到达 shadow 合约 | `MemNavData/realworld_visual_convergence_contract.py` |
| 真机到达物理标定协议 | `MemNavData/REALWORLD_SCALE_FREE_ARRIVAL_CALIBRATION_PROTOCOL_20260825.md` |
| HPC 操作手册 | `MemNavData/HPC_HARDENING_20260821.md` |

真机运行时的权威入口与状态则在 sibling repo：
`/home/asus/Research/Memnav_Realworld/CURRENT_STATUS.md` 和
`/home/asus/Research/Memnav_Realworld/TWO_PASS_REVISIT_RUNBOOK_20260825.md`。

## 8. 当前 HPC 状态快照

审计时间：2026-08-25 07:05 EDT / 19:05 Asia/Shanghai。

### 8.1 HM3D actual-full-mono shared-C

- run root：`formal_20260824T171704Z_2ce2ae67`；
- result-blind sealed population：8 histories / 6 scenes；目标 24 / 15，明确 underpowered；
- 8 条 factual-C collection 均正常完成；C success `5/8`，覆盖 4 scenes；
- 该 `5/8` 只是决定谁能进入共同 B2 起点，不是 CEC-vs-baseline SR；
- 已提交 immutable bundle 的 collection array 错误使用 `0-259`；截至快照已越过约 199 个
  indices，其中只有冻结 population 内的 8 个是有效工作，其余为空任务；`199-259` 仍等待
  `Resources`；
- seal、true-stack smoke、B2 evaluation、aggregate 和 verifier 均尚未开始；
- 因此当前没有新的 multileg SR。

统计能力已经可以先验判定：factual C 只有 `5/8` 成功，所以严格 shared-C B2 最大
`N=5`。即使 all-prior 相对 initial-leg-only 达到完美 `+5/-0`，exact two-sided McNemar 也
只能到 `p=.0625`。因此这批运行只能作为共享物理前缀的机制 pilot，不能再称作 formal
confirmation；继续完成的价值是验证因果实现和估计效应方向，而不是追求显著性。

Git 中的 deferred launcher 已修复为精确 sealed-population array，但不会伪装成已经改变了
远端旧 immutable job。

### 8.2 ViNT shared-C controller portability

- 18 个 source collection 后，factual-C success population 仅 4 histories / 1 scene；
- paired B2 job `16289955_[0-17%2]` 尚未启动，reason=`QOSMaxGRESPerUser`；
- aggregate/verify 等待依赖；
- 无正式 SR，而且即使完成也只能视为极小机制样本。

## 9. 真机状态与证据边界

当前 Nav 侧已经具备：

- 单目 CEC/NavDP RTX hub；
- recording/query 两阶段目标生命周期；
- goal candidate 与 memory 排除；
- NavDP FIFO warm-up 与 goal-session receipt；
- scale-free direct-bearing handoff；
- reset-required、watchdog 和 fail-closed transport contract。

Jetson/Go2 下位机正式实现维护在独立 `Memnav_Realworld` 仓库，本次 Nav commit 不把它复制
回来。当前真机已经验证 transport、disabled shadow、深度事务和 Go2 转向死区修复，但三次
powered trial 都仍是失败，尚未建立一个完整自主 ImageGoal 到达：

- Q->R CEC Revisit：移动 `3.01 m`，以 path-length safety abort 结束；
- R->Q native Novel：旧速度门控导致左右 hunting，之后已修复控制合约；
- S->Q full-mono：真实最近距离 `0.993 m`，经过高共视窗口但没有可靠到达判定，最终
  operator stop。

S->Q 的 frames 325--328 虽通过 LightGlue/LingBot-depth/PnP chain，预测距离最低却只有
`0.125 m`，相对独立物理最近距离至少低估 `7.9x`。因此现在只允许 PnP bearing 获得控制权，
metric norm 只写诊断，不能授权 STOP。

新增的只读 visual-convergence audit 扫描了 431 帧，仅 15 帧通过已有 two-view precheck；
最强 frame 326 有 331 matches、299 fundamental inliers、query/reference hull coverage
`0.712/0.398` 和 normalized identity flow `0.0613`，但仍对应物理未到达。这说明“高共视”
也不能从单条失败轨迹后调成 STOP 阈值。

本次新增的 pure contract 只消费 proof-conditioned image-space residual，并要求
`request_hold -> K 个连续静止观测 -> shadow_stop`。它明确不读取 metric translation，且
`runtime_stop_authorized` 永远为 false；没有做任何机器人运行时或电机权限变更。

准确状态是“软件框架和安全合约基本齐全，自动视觉收敛/到达判定仍是物理闭环缺口”，不能
写成真机实验已完成。

## 10. 当前最强论文叙事

1. **冲突**：持续视觉历史同时包含 utility 与 interference；相似历史不应天然拥有控制权。
2. **方法**：用 proposal -> witness -> authority -> narrow control interface 把 loop closure
   变成可审计的动作授权。
3. **统一单目架构**：一个 causal geometry stream 同时服务逐步 depth 与跨目标 episodic
   bearing，最终只有一个 frozen diffusion policy 生成动作。
4. **证据分解**：Final14 隔离 memory authority；Gate D 隔离 mono depth；Fresh HM3D
   验证 full-mono external composition；lifelong 实验检验跨目标累积。
5. **核心发现**：raw memory 在 supported Revisit 上已经接近 ceiling；CEC 的可识别价值是
   在保留 Revisit utility 的同时减少 unsupported-history interference。

这比“DINO + LightGlue + LingBot + NavDP 拼接”更准确：贡献不在发明每一个 backbone，而在
历史证据如何获得、限制并失去控制权限。

## 11. 当前禁止的主张

- CEC 形式化保证安全或永不在 Novel 上接管；
- CEC 在高支持 Revisit 上显著超过 raw DINO；
- CEC 解决了 Novel ImageGoal navigation；
- 当前方法完全不含 learned model；
- 所有 headline 实验都是 end-to-end mono；
- mono depth 与 metric RGB-D 已 non-inferior；
- Pi3X/learned localizer 已替代显式 geometry proof；
- controller portability smoke 证明 controller-agnostic SR；
- GOAT 或 Replica 已验证外部 benchmark 泛化；
- 真机已经有闭环成功率；
- 当前正在排队的 shared-C partial count 是正式方法结果。

## 12. 下一步优先级

### P0：物理标定尺度无关的真机到达证据

保持 scale-free 原则，构造独立的连续多帧 visual-convergence proof；bearing 负责接近/对齐，
STOP 不读取未经验证的单目 metric norm。先在 3--4 个地点按预声明距离/朝向网格采集物理
标签；calibration/confirmation 按地点隔离，冻结 rule SHA 后才能读取 confirmation。之后依次
进入 disabled shadow、系绳低速和正式 SR。当前禁止直接接 runtime STOP。

### P1：完成 shared-C B2 因果 pilot

让旧 job 的多余空索引自然结束，完成 seal/evaluation/verifier。唯一问题是：在严格相同
A/B/C 物理前缀后，`all_prior` 是否优于 `initial_leg_only`。由于最大 `N=5`，无论结果多好
都只报告 pilot；若方向值得继续，再冻结一个有统计能力的新 population，而不是复用本批。

### P2：论文消融与效率

- proposal-only / witness-only / certificate / exact fallback；
- first-use latency、cached latency、显存和历史长度曲线；
- full-mono / metric control attribution；
- supported/unsupported utility-interference frontier。

### P3：Controller portability 只保留为接口证据

除非获得足够的 shared-C population 和完整 verifier，不把 ViNT/iPlanner/NoMaD 等小样本
提升为主表 SR。主论文应继续围绕 frozen NavDP 展开。

## 13. 本次 release 的验证与排除项

提交前验证：

- 修改/新增 Python 全部 `py_compile` 通过；
- 修改/新增 shell 与 sbatch 全部 `bash -n` 通过；
- 此前 release 的 MemNav 合约：147 tests passed；
- Habitat 构造/聚合：21 tests passed；
- 本次 scale-free arrival contract 与真机 hub focused regression：`37 passed`；
- 扩展的 15-suite pure-contract regression：`171 passed`；Habitat-linked
  `test_paper_single_revisit_contract.py` 在当前 `memnav` 环境因缺少既有 `quaternion` 依赖未
  进入 collection，不把它误记为断言失败或本次回归通过；
- 431-row 真实 audit CSV schema smoke：431 行兼容、15 个 frozen precheck pass、15 个被
  deliberately broad smoke rule 读通，`metric_translation_consumed=false`；该 broad rule
  只验证数据接口，不是冻结阈值或方法结果；
- `git diff --check` 通过；
- 未发现私钥、GitHub token 或设备登录 PIN。

明确不提交：

- `goal.jpg`、`image.jpg`、`input_image.jpg`：本机临时调试图；
- `paper/`：独立 `AlanZhu2006/Memnav_Paper` 工作区，不嵌入 Nav；
- `.diagnostics/`、rollouts、scene assets、weights、checkpoints 与机器人日志。
