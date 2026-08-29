# MP3D Table-1 controller portability formal result（2026-08-29）

## 1. 冻结问题与 population

本实验在同一份 outcome-blind MP3D population 上，分别估计 CEC 对两个冻结
ImageGoal controller 的组内 paired effect：

- 42 条 actual-mono causal histories；
- 25 个 scene clusters；
- 每条一个 unsupported Novel 和一个 supported Revisit，共 84 queries；
- Novel 的初始路径方向 front/side/rear 为 `14/9/19`；
- runtime 不读取 Novel/Revisit role；
- 同一 controller 内 native/CEC 共享 query、history、checkpoint、seed、600-step
  budget 与 1 m success radius；
- NavDP 使用 causal-monocular depth sidecar；ViNT 使用 first-certified bounded-turn
  bearing adapter；
- NavDP 与 ViNT 的绝对 SR 不是 paired controller-superiority estimand。

benchmark manifest SHA-256：
`a33f210fdd0cfa84e82c4d403ac79056dcc7959cd1ce84bf62bec8c5632deb69`；
construction verifier SHA-256：
`618c409f7c7c62ad739687935cdd6f2e564e96aed6ccf6059d887d795c3e953e`。

## 2. 正式四行

| Controller | Arm | Novel | Revisit | Overall |
|---|---|---:|---:|---:|
| NavDP | native | 17/42 | 9/42 | 26/84 |
| NavDP | +CEC | 17/42 | 37/42 | 54/84 |
| ViNT | native | 9/42 | 2/42 | 11/84 |
| ViNT | +CEC | 9/42 | 24/42 | 33/84 |

### NavDP primary paired result

- Revisit：`9/42 -> 37/42`，paired `+29/-1`，risk difference
  `+66.67 pp`，two-sided exact McNemar `p=5.7742e-8`，scene-cluster
  bootstrap 95% CI `[+51.06,+81.40] pp`；
- Novel：`17/42 -> 17/42`，paired `+0/-0`，risk difference `0 pp`；
- Overall：`26/84 -> 54/84`，paired `+29/-1`，risk difference
  `+33.33 pp`，`p=5.7742e-8`，scene-cluster CI
  `[+25.53,+40.70] pp`。

NavDP 次要量：

| Role | Native SPL | CEC SPL | Native mean path | CEC mean path | Native mean steps | CEC mean steps |
|---|---:|---:|---:|---:|---:|---:|
| Novel | .2895 | .2895 | 13.82 m | 13.50 m | 391.0 | 383.9 |
| Revisit | .0801 | .6620 | 15.06 m | 4.21 m | 426.5 | 128.9 |
| Overall | .1848 | .4758 | 14.44 m | 8.85 m | 408.7 | 256.4 |

### ViNT paired result

- Revisit：`2/42 -> 24/42`，paired `+22/-0`，risk difference
  `+52.38 pp`，two-sided exact McNemar `p=4.7684e-7`；
- Novel：`9/42 -> 9/42`，paired `+0/-0`；
- Overall：`11/84 -> 33/84`，paired `+22/-0`，risk difference
  `+26.19 pp`，`p=4.7684e-7`，scene-cluster CI
  `[+19.00,+33.78] pp`。

| Role | Native SPL | CEC SPL | Native mean path | CEC mean path | Native mean steps | CEC mean steps |
|---|---:|---:|---:|---:|---:|---:|
| Novel | .1757 | .1757 | 12.48 m | 12.64 m | 338.7 | 343.5 |
| Revisit | .0261 | .5714 | 14.05 m | 6.24 m | 380.8 | 176.4 |
| Overall | .1009 | .3736 | 13.27 m | 9.44 m | 359.8 | 260.0 |

两套结果共同支持同一 proposal--witness--authority contract 可被 NavDP 与 ViNT
消费，并在 MP3D 上保留显著 Revisit 增益。它们不能用于比较 NavDP 与 ViNT 谁是更强
controller；每一项因果 estimand 都只在 controller 内 paired。

## 3. Novel authority 与 exact fallback

两条 controller 链的 certificate decision 一致：

- 42/42 Revisit queries 获得 takeover；
- 1/42 Novel query 获得 takeover；
- 其余 41/42 Novel queries 全部拒绝并 exact fallback；
- runtime failure plans：0；
- metric-depth sensor reads：0。

因此正式口径是“Novel SR interference 为零、41/42 exact fallback、1/42 takeover”，
不能写成“Novel 零接管”。唯一 takeover 是
`kEZ7cmS4wCh / episode_0004 / pair_00_novel`。它的 construction
`max_online_a_covis=0.063203`，低于冻结 Novel 门 `0.10`，但高于 certificate 的
5% coverage 量级。运行时 witness 为 32 PnP inliers、query/reference hull coverage
`.10435/.05666`、reprojection RMSE `1.329 px`，因此通过冻结 certificate。

该 takeover 没有改变二值结果：NavDP 与 ViNT 的 native/CEC 均失败。它也不是一个
简单的“错误几何一定伤害导航”案例：NavDP 的 final distance/path 从
`20.19 m / 22.31 m` 改善到 `4.85 m / 8.70 m`，而 ViNT 的 final distance 近似不变、
path 从 `14.92 m` 增至 `21.78 m`。最保守解释是 construction 的 Novel support 门与
operational certificate coverage 门之间存在一个弱支持边界；不能据此把 CEC 描述成
Novel/Revisit 分类器或形式安全证明。

## 4. 独立复算与 provenance

结果打开顺序为：exact completion 全齐 -> aggregate -> controller-specific raw-file
verifier -> SHA-pinned joint seal。最终作业：

- NavDP exact retry / aggregate / verifier：
  `16559033_27 / 16559034 / 16559035`，均 `COMPLETED 0:0`；
- retained ViNT exact retry / aggregate / verifier：
  `16558665_24 / 16558668 / 16558669`，均 `COMPLETED 0:0`；
- joint seal：`16559083`，`COMPLETED 0:0`。

run root：
`/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_controller_portability_20260829/formal_20260829T085025Z`。

- NavDP summary SHA-256：
  `63ae5d4b06e4e8acc07e9275f07a018585ee02a7d38ce610b30ea5651a7e75f6`；
- NavDP independent verifier SHA-256：
  `d87853fade9c932ca1541ad830b2a0745048adb09e8081c99d954a8198ed6244`；
- ViNT summary SHA-256：
  `7b0b18e2d94fb371ad3a3df055b974245f642218904f7dafc5f29fa8a3e9d76a`；
- ViNT independent verifier SHA-256：
  `9596e85fa78a94070d3cd0e21ce93b5b80207bf1d868db4e7652ca92d116c3bd`；
- joint receipt SHA-256：
  `e63ce0d30347784b5b328a1ca7158b7462c3d12fef1081727d3b98d0981c235f`。

除两个正式 independent verifiers 外，又直接从 168 条 NavDP raw metric rows（重组为
84 个 paired queries）和 84 条 ViNT pair-audit query rows 独立复算成功、gain/loss、
McNemar 与 scene-cluster bootstrap；
计数和检验值与封存结果一致。Joint receipt 明确记录
`cross_controller_absolute_sr_is_not_a_paired_estimand=true`。

## 5. 基础设施事件不进入科学结果

原 arrays 的缺失 cell、ViNT TCP 端口冲突、NavDP identical-frame transaction cache、
以及 attempt-1 的旧 authority endpoint 都由冻结后的 additive repair 处理；已完成
history 从未重跑。诊断期间的 outcome-visibility 披露、只读失败归档、source receipt
与 scheduler seal 修正见：

- `MP3D_TABLE1_CONTROLLER_EXACT_REPAIR_PROTOCOL_20260829.md`；
- `MP3D_TABLE1_NAVDP_AUTHORITY_CACHE_COMPOSITION_REPAIR_20260829.md`；
- `MP3D_TABLE1_NAVDP_AUTHORITY_CACHE_SUBMISSION_INCIDENT_20260829.md`。

这些事件只解释为何需要 exact repair，不改变上述 population、method、threshold、
seed、budget、success radius 或 estimand。
