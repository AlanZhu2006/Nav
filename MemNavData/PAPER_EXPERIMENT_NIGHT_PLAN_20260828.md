# 论文未完成实验总账与 2026-08-28 夜间执行计划

更新时间：2026-08-28（Asia/Shanghai）。本文件区分“已有论文证据”、
“已提交但尚无结果”、“只完成准备”和“当前不应运行”。排队、构造完成或 smoke
成功都不记为论文结果。

## 1. 今晚真正要关闭的三个问题

### 1.1 Final14 五行 depth ablation 的缺失行

问题：在同一 21-history、42-query population 上，完全不给 NavDP 有效 depth 时，
Novel、Revisit、Overall SR/SPL 和路径长度分别是多少？

唯一新增臂为 `zero_native`。它与既有 metric native、mono native、metric CEC、
mono CEC 共享 Goal-A replay、query、seed、budget、checkpoint 和 1 m 成功定义。
该汇总脚本已经直接计算 SPL 与 mean path，不需要再跑一套 episode。

正式 DAG：

```text
16499686 smoke (COMPLETED, 0:0)
  -> 16499701 formal array 0-20%2
  -> 16499709 summary + independent raw verifier
```

不可提前写入论文的内容：任何 partial SR、未通过 verifier 的汇总，以及旧 Novel-A
Gate-D 的 `23/40`。后者不是同一 mixed-role population。

### 1.2 CEC 的 proposal-matched authority ablation

问题：当 proposal、local matching、PnP、bearing adapter 和 frozen NavDP 全部相同，
完整 certificate 相比“只要产生有限 PnP witness 就授权”究竟改变什么？

两臂：

```text
mono_cec                  strict operational certificate
mono_unthresholded_witness finite PnP witness, no certificate thresholds
```

这是 authority-only ablation；unthresholded witness 仍使用 DINO、LightGlue、
Fundamental-MAGSAC、LingBot historical depth 和 PnP，不能写成 geometry-free 或
retrieval-only。

提交前已经通过：

- 本地 20 个 authority/handoff tests；
- 本地与远端各 45 个 policy/router tests；
- 容器内 strict 与 unthresholded 两条 route contract dry-run；
- GPU 和 CPU Slurm `--test-only`；
- immutable bundle 全量 hash。

正式 DAG 被串在 zero-depth verifier 后：

```text
16499709
  -> 16501311 authority smoke
  -> 16501313 formal array 0-20%2
  -> 16501320 summary + independent raw verifier
```

本实验的主要价值是 Novel false authorization / Revisit rejection 与 SR 的风险—覆盖
权衡，而不是保证 strict CEC 一定取得更高总 SR。

### 1.3 HM3D 中“新经历是否成为新记忆”

问题：B 原本对 A history 是 Novel；agent 实际到达并观察 B 后，在同一个封存 C
前缀之后，允许 B2 检索新增 B history，是否优于把 memory ceiling 永久锁在 A？

原 Natural-V4 的 40-history power gate未通过。独立 verifier 确认：

- factual-B rollouts `99`；成功 `27`；
- 最终 supported population `22 histories / 15 scenes`；
- strong support `18/22`；
- population SHA-256
  `ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5`；
- `0` simulator metric-depth reads；
- C/B2/C2 尚未执行，query outcomes 未读取。

因此另行冻结 underpowered amendment：全部 22 条先各跑一次 factual C；成功 C
在任何 B2 之前封存；随后从完全相同 C prefix 配对 `all_prior`、
`initial_leg_only` 和 `forced_reject_native`。无论 p 值如何，都只称为
“underpowered external continual-memory mechanism test”，不能改称原 powered
confirmation。

提交链：

```text
16501320
  -> 16501659 CPU deferred launcher
  -> factual-C array 0-21%2, each <= 1 h
  -> C population seal
  -> one B2 true-stack smoke
  -> B2 formal array over successful C histories, %2, each <= 1 h
  -> aggregate
  -> independent raw verifier
```

动态 job IDs 会由 launcher 写入远端
`underpowered_deferred_submission/{collect,evaluate}.json`。C2 不在本次 clean
estimand 中：B2 treatment 后 C2 起点已分叉，不能再冒充严格 paired comparison。

## 2. 会议表格逐项状态

| 论文项目 | 已有可用证据 | 今晚动作 | 仍缺什么 |
|---|---|---|---|
| Cross-controller / cross-dataset | NavDP+CEC 已有 fresh full-mono HM3D；MP3D 有受控 Final14 | 不提交新的 ViNT SR | ViNT 必须先实现可部署 bearing consumption，再用 fresh outcome-blind population；协议匹配的 MP3D 四行仍缺 |
| HM3D continual by leg | MP3D 18 条 retained-history dose response 已成立；HM3D A/B population 已封存 | attempt-2 smoke-gated 精确修复 DAG 已提交 | 先完成 factual-C seal、node-affine B2 与独立 verifier；若会议坚持完整 Leg-3 Novel/Revisit 主表，仍需另冻相同前缀的 Novel-C 对照 |
| Real robot | transport/hash/fail-stop 和静态接口证据 | 不做无人值守运动 | 现场 paired Novel/Revisit、自动 arrival/STOP、视频与路径口径 |
| Depth ablation | 同一 Final14 上 metric native、mono native、metric CEC、mono CEC | 正式运行 zero native | verifier 通过后即可形成五行 SR/SPL/path 表；必须注明 Goal-A history 来自 metric replay |
| CEC mechanism | raw/CEC、certificate ladder、known-role diagnostics 已有 | 正式运行 matched authority arm | verifier 后再决定主表用 raw / unthresholded / CEC / known-role 哪些行，不把不同信息 oracle 混为一谈 |
| Length bins | 当前任务构造距离主要为 2--9 m，episode raw receipt 可审计 | 不提交 GPU | 20--30 m 与 30--50 m 目前不可构造/样本为空；先报告 constructibility，不制造小分母表 |

## 3. ViNT：为何今晚不再提交正式 SR

第一轮 formal adapter 是负结果：Novel exact fallback 正常，但 Revisit 从 native
`5/28` 降到 CEC `0/28`。审计发现 adapter 只把 goal JPEG 换成历史 anchor，却丢弃
了 CEC 的 certified bearing；28/28 第一段 ViNT horizon 与 bearing 相反，中位方向误差
约 165 度。

随后在五个已知 loss cases 上做的机制测试显示，理想化零平移 yaw alignment 可把
首段 heading<=30 度和 first-horizon closer 从 `0/5` 提到 `5/5`；anchor-aligned
成功 `5/5`。但这是 outcome-aware subset 加 ideal yaw，只证明根因和可修复性，不是
论文 SR。

下一步必须是：

```text
CEC accepted bearing
  -> bounded 30-degree physical turns
  -> each turn obtains a fresh observation
  -> shadow-update controller/history exactly once
  -> discard pre-turn ViNT horizon
  -> run unchanged ViNT on the verified anchor ImageGoal
```

只有本地/单元 contract 通过，且 fresh outcome-blind HM3D/MP3D population 在结果前
冻结后，才允许提交 ViNT formal。继续在五个 loss cases 上调动作会构成 outcome tuning。

## 4. 不应今晚烧 GPU 的项目

- **真机正式实验**：需要人在场、安全绳/急停、arrival/STOP 规则和外部视频，不能用
  HPC 替代，也不能无人值守启动。
- **20--50 m distance bins**：当前 source task 的 shortest-path construction 不覆盖
  这些桶；应先换 benchmark contract，而不是重复跑 2--9 m episodes。
- **GOAT**：NavDP 与 GOAT object-centric target/camera/discrete-action contract 不同，
  已有 adapter 失败不能当作 CEC 外部泛化；当前不进入主实验队列。
- **新的 learned relocalizer 长训**：CDEC、candidate-free GCT、small residual 与 Pi3X
  已提供足够负结果；在没有新 supervision/architecture hypothesis 前不再用长训替代
  闭环证据。
- **ViNT outcome-aware formal**：理想对齐 N=5 只能当机制结果。

## 5. 依赖与调度防错清单

今晚三个 DAG 均满足：

- 只使用 `ssh -G alantorch` 给出的共享 ControlPath，并验证远端身份为 `yz11502`；
- immutable source bundle + SHA-256 receipt；
- 复用旧 bundle 前全量 `sha256sum -c`；
- 容器内 import/route dry-run；
- H100/A100 only；
- GPU array element time limit `01:00:00`；
- GPU 并发最多 2；
- 通过 `afterok` 串成一条链，任一 verifier 失败会阻止后续科学实验；
- summary 使用 `afterany` 时必须在脚本内检查所有 raw outputs，避免数组部分失败被误汇总；
- runtime role label 不进入 controller；
- same-process pairing；
- 不覆盖既有 run root，不删除失败尝试。

截至 2026-08-28 12:37（Asia/Shanghai）的提交链为：

```text
Final14 zero depth
  16499686 smoke (COMPLETED)
  -> 16499701 formal 0-20%2
  -> 16499709 summary + verifier

Final14 authority-only ablation
  16499709
  -> 16501311 smoke
  -> 16501313 formal 0-20%2
  -> 16501320 summary + verifier

HM3D continual underpowered amendment
  16501320
  -> 16501659 deferred launcher
  -> factual-C array -> population seal -> B2 smoke -> B2 paired array
  -> aggregate -> independent verifier
```

本地最终静态/契约回归为 `116 passed`，并额外通过 Habitat Python 3.9
`py_compile`。ViNT bounded executor 已完成实现与本地 dry-run，但没有新鲜、
outcome-blind population，因此仍是 `prepared only`，未提交正式 SR。

### 5.1 端口竞争后的权威 replacement DAG

原 zero array 的 index 19 在 evaluator 启动前发生 TCP 端口竞争；其 partial 目录已
只读归档，另外 20 个完成单元未重跑。原 analysis 及其所有下游均未执行并被 Slurm
取消。加入 node-local port-block lock 后，权威链更新为：

```text
16502265 exact zero repair index 19 (COMPLETED 0:0)
  -> 16502270 replacement zero summary/verifier (COMPLETED 0:0)
  -> 16502418 portsafe authority smoke
  -> 16502420 portsafe authority formal 0-20%2
  -> 16502421 authority summary/verifier
  -> 16502570 portsafe HM3D deferred launcher
```

Zero-depth verifier 已通过：Novel `3/21`、Revisit `1/21`、overall `4/42`，
`verified=true`；详见 `FINAL14_ZERO_DEPTH_RESULT_20260828.md`。authority 与 lifelong
仍然是已提交、未出结果状态。

### 5.2 authority dependency-provenance repair

`16502418` 的 38 个启动测试全部通过，但第一个 query 在调用
`/retrieval_probe_step` 前失败，未产生任何 arm outcome。根因不是模型、数据或端口：
authority overlay 漏装 `MemNavData/monocular_depth_runtime.py`，Python namespace
package 因而从旧 Final14 base bundle 解析该模块；新 server 所需的
`bind_monocular_depth_transaction` 在旧文件中不存在。其 dependent formal、verifier
和 lifelong launcher 均被 `afterok` 自动取消。

修复没有改变 population、arm、threshold、seed、budget 或 controller。新的 immutable
bundle 显式包含 monocular runtime 及其测试，并在 GPU 启动前断言
`module.__file__ == <current bundle>/MemNavData/monocular_depth_runtime.py`。本机与远端
分别通过 `28` 个 authority/mono tests、`45` 个 policy/router tests、port-lock test，
随后两个正式容器 route dry-run 通过。当前权威链为：

```text
16503212 provenance-locked authority smoke
  -> 16503217 formal array 0-20%2
  -> 16503241 summary + independent verifier
  -> 16503597 HM3D underpowered deferred launcher
```

Authority bundle receipt 为
`18fe24537b840871017dfc8c5e9cc34a141dfa5eca64c73f4c36395570979d10`；lifelong
继续使用已验证的 portsafe bundle `9207614aadf20b62`。旧任务及失败 smoke 只作为
基础设施审计记录，不进入科学分母。

### 5.3 authority receipt-schema repair

`16503212` 成功执行了两个 80-step smoke arms，证明依赖 provenance 修复有效，但在
post-run proposal audit 处 fail closed。两个 endpoint 实际分别使用
`strict_certificate` 和 `pnp_pose_available`，proposal 顺序、anchor 与候选列表也一致；
问题是 `eval_2leg_habitat.py` 的 generic plan serializer 没有把 response 中已有的
`certified_relocalization_authority{,_policy}` 写入 JSON，审计因此读到 `None`。

修复仅增加收据字段，不改变 matcher、PnP、authority decision、bearing、trajectory 或
success。序列化逻辑被拆成无 NumPy/OpenCV 等第三方依赖的
`cec_authority_receipt.py`，并分别用 memnav 与 Habitat Python 测试；这避免为 evaluator
引入新的 `cv2` 依赖。当前链为：

```text
16504303 receipt-complete authority smoke
  -> 16504304 formal array 0-20%2
  -> 16504307 summary + independent verifier
  -> 16504366 HM3D underpowered deferred launcher
```

新 bundle receipt 为
`3f5783aca521b0a57c9e27d7278fa386b3390fca2660023716db98e58717d756`。
`16503217/16503241/16503597` 在任何 formal/lifelong query 前被依赖取消。
新 smoke `16504303` 已 `COMPLETED 0:0`，并确认 prefix equality、两种 role 的
proposal equality 与 runtime role invisibility；formal `16504304` 已自动开始。未读取
任何 partial formal outcome。

### 5.4 authority 完成与 HM3D factual-C 精确续跑

authority replacement DAG 已全部完成并由 `16504307` 独立验证。21 个 history 的
42 对 proposal 全部一致；strict CEC 与 unthresholded finite-PnP witness 的 overall
SR 分别为 `28/42` 与 `25/42`（paired `+4/-1`，+7.14 pp，exact McNemar
`p=0.375`）。Novel 为 `8/21` 与 `5/21`（`+4/-1`），Revisit 两臂均为
`20/21`。这个结果只回答 certificate authority 的 matched ablation，不提升为新的
fresh-generalization claim。

下游 launcher `16504366` 因 `safe_sbatch` 把默认 GPU partition 放在显式
`cpu_short` 之前而失败；它此前已创建 factual-C array `16505696`。该 array 有 16 个
index 完成，失败集合精确为 `0,1,7,9,11,13`。其中 `0,13` 是 Goal-B 前已有
Goal-A FIFO 时的错误空队列审计假设，`1,7,9,11` 是跨节点 Habitat RGB 哈希漂移。
没有读取成功 factual-C 的导航 outcome，也没有产生或读取 B2 outcome。

修复 bundle receipt 为
`250a4891728ee4c784f86b73e292e846c679c0a13f7fb66f95c5437d0da28631`。
6 个失败 partial 已移动到只读 incident archive，16 个完成目录保持原位；当前 DAG：

```text
lane 0: 16509621 index 0 @ gh005
          -> 16509634 index 7 @ ga005
          -> 16509637 index 11 @ ga028
lane 1: 16509627 index 1 @ gh001
          -> 16509636 index 9 @ ga003
          -> 16509642 index 13 @ ga002
both lanes -> 16509644 retained-output SHA barrier
           -> 16509648 factual-C seal
           -> 16509649 node-affine smoke/evaluation launcher
```

B2 的每个 sealed-C 样本必须回到其 collection `compute_identity.host` 执行；动态
launcher 用两条确定性 lane 保证最多并发 2 个 GPU job。最终仍必须经过 aggregate 与
independent raw-file verifier，且无论数值如何都只能报告为 underpowered external
continual-memory mechanism test。

所有 job 创建后，首次远端 submission-receipt heredoc 因 shell 引号丢失而失败；它没有
修改或重复提交 DAG。本地已先生成的权威 receipt 随后通过已验证的 `yz11502` socket
逐字节复制到远端并锁为只读，SHA-256 为
`fff95fb18a6bfc17046f970607577e8c6badd039a0aa6c3a81dadf63b97b8501`。
该纯收据修复记录在
`HM3D_LIFELONG_UNDERPOWERED_COLLECT_REPAIR_RECEIPT_REPAIR_20260828.json`。

### 5.5 HM3D repair attempt 1 启动闭包故障与 attempt 2 提交

独立复查发现，attempt-1 lane 0 的 `16509621/16509634/16509637` 均在 evaluator
启动前以 `2:0` 退出，日志完全相同：旧 frozen hub 不认识 runner 传入的
`--reject-policy shared_native_exact`。它们分别对应 index `0/7/11`，运行节点仍严格为
`gh005/ga005/ga028`，所以不是新的跨节点漂移，也没有产生导航 outcome。lane 1 和下游
当时尚未开始；`16509627/16509636/16509642/16509644/16509648/16509649` 已按精确
job ID 于 2026-08-28 11:08:55 EDT 取消，没有终止任何 running job。

根因是 immutable dependency closure 不完整：repair overlay 提供了新 runner，但 runner
仍从旧 task bundle 启动 hub。旧 hub SHA-256 为
`fb249dae4f865bd17e12bd1673156403651592b870ac1a216de1ffca3d36b7d5`；它没有该 CLI
参数，却已在同一 `ComparisonPlan` 中字面冻结
`reject_policy="shared_native_exact"` 与 `fallback_controller="navdp"`。attempt 2 不替换
hub，只加入 dependency-free AST 审计：新 hub 显式传参；旧 hub 仅在证明上述两个
literal 同时成立时省略冗余参数；其他 policy、无法证明的旧 hub 或旧 hub 的 handoff
模式一律在模型启动前 fail closed。controller、hub 行为、weights、数据、seed、动作、
memory、导航阈值与 certificate 阈值均未改变。

本地 27 个针对性测试、Slurm wrapper/port 测试和 shell/compile 门通过；远端为 26
passed、1 个 minimal-overlay expected skip，并通过真实 frozen hub AST、容器 overlay
来源、pre-archive audit 与六类 `sbatch --test-only`。新的 immutable bundle receipt 为
`899141ad23b4ff0ca3012bb68b9bc6aa0e5a8e1ee45bbfe4abcf8fa98ab89f26`。
三个 attempt-1 启动 partial 已移动到独立只读 archive（恰好 `0/7/11`，mode `0555`）；
原 6 个失败 partial archive 和 16 个完成 factual-C 目录均未改动。

正式 attempt-2 DAG 为：

```text
16514058 index-0 collect compatibility smoke @ gh005
  -> 16514066 outcome-blind structural smoke verifier
     -> lane 0: 16514071 index 0 @ gh005
                  -> 16514136 index 7 @ ga005
                  -> 16514153 index 11 @ ga028
     -> lane 1: 16514101 index 1 @ gh001
                  -> 16514150 index 9 @ ga003
                  -> 16514157 index 13 @ ga002
both lane tails -> 16514159 retained-output/runtime integrity barrier
                -> 16514162 factual-C population seal
                -> 16514165 node-affine B2 resume launcher
                -> B2 smoke -> two formal lanes -> aggregate -> raw verifier
```

collect smoke 写入独立 root，不进入科学分母；只有 `16514066` 验证结构、hash、节点、
hub health 与 policy 后，六个正式 repair 才能开始。每条 lane 使用 `afterok` 和
invalid-dependency cancellation fail fast，最大科学 GPU 并发为 2。提交后独立
`squeue/sacct` 复核 11 个 job 和全部依赖边均正确；截至 2026-08-28 23:46
（Asia/Shanghai），`16514058` 因 `Priority` 等待，Slurm 的非承诺预测启动时间为
2026-08-30 02:33（Asia/Shanghai），其余均为依赖等待。submission receipt 本地/远端
一致且只读，SHA-256 为
`85fef2876e71ca80d5327b1be66621357b70942a1ea836ba2fc5e87c14f09d5a`。

### 5.6 attempt-2 health receipt 误判与 2026-08-29 精确恢复

collect smoke `16514058` 最终在冻结节点 `gh005` 完整结束，但 CPU gate `16514066`
只因 `/healthz` 缺少 `reject_policy/reject_controller` 两个字段而 fail closed。只读检查
表明这是旧 hub 的既有 schema：它报告 `schema=cec_controller_portability_hub_v2`、
`controller=navdp`、`initialized=true`、`reset_required=false`；同一 smoke 的
hash-bound `compute_identity.json` 已记录
`cli_contract=legacy_shared_native_exact` 与 `reject_policy=shared_native_exact`，启动前
AST audit 又证明同一 `ComparisonPlan` 中的 fallback 为 NavDP。因此失败属于 verifier
重复要求新 health schema 字段，不是运行时 authority 改变。

在此判断前后均未读取 factual-C success/SPL/distance/action outcome，也没有任何 B2
outcome。六个 scientific repair 和 seal/B2 下游均由依赖取消，未开始。修复只让 verifier
接受“legacy health + compute identity + AST”这一等价收据组合；hub、controller、weights、
population、seed、threshold、budget、node map 与六个 repair index 全部不变。已完成 smoke
不重跑且不进入科学分母。

本机 25 个相关测试、远端 10 passed/1 expected skip、旧 smoke 实物审计、bundle 全量
hash 和五类 Slurm `--test-only` 均通过。新 verifier bundle receipt 为
`0facfeadd85286e783a52b44af2f553275c8225dbb70166eaddd10d5256ff896`。正式恢复链为：

```text
16521565 health-receipt smoke gate (COMPLETED 0:0)
  -> lane 0: 16521578 index 0 @ gh005
               -> 16521614 index 7 @ ga005
               -> 16521647 index 11 @ ga028
  -> lane 1: 16521597 index 1 @ gh001
               -> 16521638 index 9 @ ga003
               -> 16521653 index 13 @ ga002
both lane tails -> 16521666 22/22 retained/runtime integrity barrier
                -> 16521671 factual-C seal
                -> 16521679 node-affine B2 resume launcher
                -> B2 true-stack smoke -> paired formal lanes
                -> aggregate -> independent raw verifier
```

提交后复核时，`16521578/16521597` 已在各自冻结节点 `COMPLETED 0:0`，index 9
正在 `ga003` 运行，index 7 正常等待 QOS，其余等待依赖；没有新的 early failure。机器可读边界见
`hm3d_lifelong_underpowered_collect_health_receipt_repair_20260829.json`，收据修复说明见
`HM3D_LIFELONG_UNDERPOWERED_COLLECT_HEALTH_RECEIPT_REPAIR_20260829.md`。无论最终数值，
本链仍只能作为 22-history/15-scene underpowered continual-memory mechanism test。

## 6. 结果晋级规则

只有同时具备以下四项才能进论文数字表：

1. episode-level raw receipt 完整；
2. success 从 raw final distance 重新计算；
3. paired gain/loss、exact McNemar 与 scene-cluster CI 可复算；
4. independent verifier `verified=true`。

zero-depth 和 authority 使用 consumed Final14，只能用于 paired ablation；HM3D
underpowered amendment 只能用于 continual mechanism。它们都不能被重新命名为新的
fresh generalization 结果。
