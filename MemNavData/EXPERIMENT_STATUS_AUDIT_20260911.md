# 近期实验状态与结果复核：2026-09-11

本轮按用户要求暂停 Table II 新设计，检查此前实验，不启动新评测或训练。
实时队列快照：北京时间 2026-09-11 01:13；Table I 完整单元读出：01:12。
远端使用共享 `alantorch`，实际身份 `yz11502`、登录节点 `torch-login-a-0`。

GEM 是论文名称，封存代码/结果中的 `cec` 是同一条 strict 方法。
`cec_no_coverage` 是显式消融，不是已替换默认 GEM 的新主方法。

## 1. 总体状态

| 实验 | 当前状态 | 完成量 / 主要结果 |
|---|---|---|
| 去面积条件的四臂共视消融 | 完整结束，逐项收据复算通过 | 159 queries / 636 rollouts；低共视 GEM 28/40→32/40 |
| Table I 修复补跑 + NoMaD | 未完成 | 163/210 完整单元；2 失败；45 等待；每单元四 rollout |
| 修复版 actual-mono A + mixed queries | 已完成 | 112 A 源任务；45 合规历史；GEM 60/90、native 33/90 |
| 原三臂共视分层 | 已完成 | 159 queries / 477 rollouts；与新四臂为同一已消费查询池，不是独立新场景 |
| DINO 历史图直接替换 ImageGoal | 本机完成 | 2 scenes、4 queries、16 rollouts；替换臂 0/4 |
| 固定 raw 图像对：LightGlue / MASt3R | 本机完成 | 两种 matcher 各 159 pairs；没有新导航 SR |
| Learned anchor 关系定位 | 本机完成 | 两种输入×三种子；未达到替换几何的证据要求 |
| 历史在线深度复用 | 本机完成 | 16 rollouts；两种来源 Revisit 均 3/4；首次证书调用显著缩短 |
| Final14 精确 SPL 补跑 | 已完成 | 21 histories / 210 rollouts，远端全量 verified=true |
| Table II A/B 差距归因 | 只读审计完成 | 已核验 379 条 A/B 记录；新 mixed-B/C 仍只是设计草案 |
| 长程 route-tangent | 旧版本实验完成 | 23 histories：native 0/23、endpoint 0/23、tangent 4/23 |
| 长程 16-history oracle 归因 | 失败、未补齐，目前未运行 | 原数组失败，summary 失败、verifier 取消；没有完整总体 SR |

本轮最新队列只有 Table I GPU 数组及其 CPU summary。01:13 时 GPU 元素均等待
`QOSGrpGRES`，没有 RUNNING 元素；这不是 SSH 失联。状态随后可能变化。

## 2. 新完成：去面积条件的四臂消融

完整 159 查询、636 次导航已于北京时间 2026-09-10 19:45 完成汇总。
GPU jobs `17297864` 两项、`17297865` 157 项全部 COMPLETED；summary `17297866` COMPLETED。

本轮独立读取全部 159 份 `archive_receipt.json` 和 `independent_verification.json`，
检查完成状态、两份验收副本一致、任务身份及完整四臂，从逐条记录复算 SR 和配对增损。
结果与封存 summary 一致。本轮没有重新下载或重新哈希所有大型原始归档；
原 GPU 归档与最终汇总已经完成逐成员/整包校验。

### 2.1 预先指定的主要分组

| 组别 | Native | Raw fixed | Strict GEM | 去面积 GEM |
|---|---:|---:|---:|---:|
| 低共视 [0.1,0.5)，N=40 | 14/40 | 39/40 | 28/40 | 32/40 |
| 较高共视 [0.5,1.0]，N=91 | 24/91 | 88/91 | 77/91 | 79/91 |
| 原 Novel 对照，N=28 | 6/28 | 5/28 | 6/28 | 7/28 |

- 主比较：低共视去面积相对 strict 为 +4/−0，+10 pp，exact McNemar p=0.125；
  scene-cluster CI 为 [+2.38,+19.51] pp。不能只拿 CI 而隐去配对 p。
- 较高共视：+2/−0，p=0.5。
- 原 Novel：+1/−0，p=1；strict 0/28 接管、28/28 exact native；
  去面积 2/28 接管、26/28 exact native，没有丢失 native 的成功例。
- 排除已披露的原 Novel 边界例后仍为 native/strict 6/27、去面积 7/27；
  去面积只剩 1/27 接管。这里的 Novel 不是完美零支持标签。
- 去面积在低共视仍低于 raw：32/40 对 39/40，+1/−8，p=0.0390625。

描述性全体合计：native 44/159、raw 132/159、strict 111/159、去面积 118/159；
去面积相对 strict +7/−0，p=0.015625。这个人工支持配比总体不是自然部署 SR，
该合计比较也不替代事先指定的低共视主比较。

### 2.2 按 GEM 实际候选历史域的五档结果

| 共视档 | N | Native | Raw fixed | Strict GEM | 去面积 GEM |
|---|---:|---:|---:|---:|---:|
| [0.1,0.3) | 14 | 5 | 13 | 8 | 9 |
| [0.3,0.5) | 26 | 9 | 26 | 20 | 23 |
| [0.5,0.7) | 23 | 5 | 22 | 17 | 17 |
| [0.7,0.9) | 32 | 7 | 32 | 27 | 29 |
| [0.9,1.0] | 36 | 12 | 34 | 33 | 33 |

以上按 frame≥8 的 GEM 实际候选域注释；不是原 frame≥39 的 23/27/28/28/25 分档。
原分档敏感性分析也保存在 summary，不能将两种分母交叉引用。

结论：面积条件确实拒绝了一部分可转化成闭环收益的方向；去掉它不是低共视问题的完整解决，
更不说明剩余所有误差都属于阈值问题。没有依据本次结果自动修改 strict 默认方法。

源文件：

`/scratch/yz11502/Research/Nav-axis-uturn-results/coverage_ablation_20260910/run_1df157b0b3d26b4f/evaluation/paired_summary.json`

SHA256：`c443a1984d8a47de2e13028b4f53562fb3403c4d017692c8c37f8fa01e085e20`。

## 3. Table I 三 controller × 两 dataset：完整单元快照

总体 210 cells / 840 rollouts；一个 cell = 一个 controller、一条实际历史、
Novel/Revisit 两目标 × native/GEM 两臂。已完成集成六项包含在总体内。

01:12 逐项检查得到 163 个完整可验证 cell，即 652 个完整配对集内的 rollouts。
不能用失败 cell 中先完成的三臂填入表格。

| 数据集 | Controller | 完整 cell / 预期 | Novel native→GEM | Revisit native→GEM | Revisit 增/损 |
|---|---|---:|---:|---:|---:|
| HM3D | NavDP | 28/28 | 6/28→6/28 | 10/28→27/28 | +17/−0 |
| HM3D | ViNT | 28/28 | 5/28→5/28 | 5/28→25/28 | +21/−1 |
| HM3D | NoMaD | 28/28 | 3/28→3/28 | 7/28→24/28 | +19/−2 |
| MP3D | NavDP | 41/42 | 13/41→13/41 | 9/41→40/41 | +32/−1 |
| MP3D | ViNT | 37/42 | 5/37→5/37 | 2/37→32/37 | +30/−0 |
| MP3D | NoMaD | 1/42 | 0/1→0/1 | 0/1→1/1 | +1/−0 |

HM3D 三子组已经完整；MP3D 三组分别缺 1、5、41 个完整 cell。
后三行是完成子集快照，不能当六组完整正式结果或比较 controller 排名。
正式六组汇总及多重比较还未生成。

HM3D 三 controller 的 GEM Novel 均无接管、全部 exact native。
MP3D 已完成 NavDP/ViNT 各有一个 Novel 接管，当前成功标签没有改变；
因此不能将它们写成「全部 Novel 物理轨迹相同」。

这是修复 query 执行/输入后的比较，仍复用原 actual A，不是三个 controller 各自自主采集 A。
NoMaD 使用发布权重的 goal-conditioned 接口；ViNT/NoMaD 的 GEM 接收认证历史 anchor 图，
不是 NavDP 的 PointGoal 接口，也不是各自官方 topomap 全系统复现。

### 3.1 当前两个失败

| 任务 | Cell | 失败位置 | 已归档的直接错误 |
|---|---|---|---|
| `17306820_110` | MP3D / NavDP / history26 | 最后一个 Revisit-native 臂 | `MemNav planning append received different JPEG bytes` |
| `17306820_158` | MP3D / ViNT / history32 | 最后一个 Revisit-native 臂 | `Controller did not consume the issued image` |

两项都是 ExitCode 1:0，运行约 13–14 分钟，不是 OOM、时限耗尽或导入依赖失败。
前三臂均已执行，但整个 cell 未通过完整配对检查，必须保留为缺项。
已从原始归档内读取 `task/logs/revisit_native.log`，不是仅从父进程 CalledProcessError 猜测。
当前能确定故障发生于输入/消费回执的一致性检查；JPEG 差异的具体成因尚未复现。
不能直接取消检查、把缺项算零或拼接部分结果。失败归档均保留、可恢复。

### 3.2 调度

- 首批 `17306819`：六项完整完成。
- 正式数组 `17306820`：157 项成功、2 项失败、45 项等待；
  加上首批六项，两批合计 163/210 项成功。
- 最新等待原因为 `QOSGrpGRES`；01:13 没有运行中 GPU 元素。
- summary `17306821` 等待依赖；现有两个缺项若不补，最终检查会报告不完整总体。
- 本轮没有重提、取消、修改时限/并发，也没有改为其他 GPU 分区。

运行根：

`/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/run_7b6bffa77891ef4f`

## 4. 本机已完成项目

### 4.1 DINO 历史 JPEG 替换目标图

2 scenes / 4 queries / 16 rollouts，独立 verifier=true：

| 方法 | Novel | 低共视 Revisit |
|---|---:|---:|
| native | 1/2 | 1/2 |
| DINO-imagegoal | 0/2 | 0/2 |
| raw fixed | 0/2 | 2/2 |
| strict GEM | 1/2 | 1/2 |

只说明这四条小测中图像替换未兑现收益；不能据此否定完整相关方法或所有 image-goal controller。
该实验不是单变量纯 bearing 消融，因为 raw 与新图像替换臂的候选范围也有已披露差异。

### 4.2 LightGlue 与 MASt3R 固定 raw 配对

本机日志已 `COMPARISON_COMPLETE`，两个 matcher 各 159 pairs。
旧状态文档所说的「下载中、comparison 未生成」已经过时。

| 描述性指标 | SuperPoint/LightGlue | MASt3R |
|---|---:|---:|
| 最低共视 14 pairs 的 F 内点中位数 | 13.5 | 94 |
| 原 Novel 28 pairs 的 F 内点中位数 | 9 | 48 |
| F 内点区分原支持角色的 AUC | 0.8776 | 0.8896 |
| F 内点区分 raw 方向误差≤30°的 AUC | 0.8589 | 0.8591 |
| 单图像对匹配总时间中位数 | 0.0201 s | 0.1160 s |

MASt3R 输出的匹配更密，绝对内点数不能直接当作更高准确率；原 Novel 对也会增多。
两个 AUC 均是已消费关联样本的描述性读出，没有独立确认或新阈值拟合。
时间为本机实现的匹配函数计时，不是完整导航延迟；MASt3R 使用官方 PyTorch RoPE 实现。
这不是 MASt3R-SLAM、联合位姿优化或 learned 授权器的闭环成功，新增导航 SR 为零。

结果：`.diagnostics/raw_match_verification_20260910_IJGnxv/comparison_v1.json`，
SHA256 `e3fa73453ef99f64aa2a8c272361a66ce8050cd0a00c03eafada84f75b164b94`。

### 4.3 Learned 与深度复用

- Anchor 定位：两臂、三种子全部训练完。未见场景 20 pairs，平均位置误差约
  1.0704 m（视觉）对 1.0396 m（视觉+XYZ），没有新导航 SR或替换部署的证据。
- 固定 top-8 联合 PnP：41 sessions，≤0.5 m 单帧 14/41、联合 12/41；不是改进。
- 在线历史深度复用：4 scenes /16 rollouts，Revisit 两臂均 3/4、Novel 均 0/4；
  首次证书函数中位耗时 20.330 s→0.126 s，不含完整检索、规划、通信延迟。
- 当前无这些实验的训练/导航进程。01:08 本机 GPU 利用率 0%，占用约 7.6 GiB；
  驻留的是其他工作线的真机 MemNav/NavDP 服务，本轮未调用、停止或修改。

## 5. 已完成的主要旧批次与未完成项

- 修复版完整单目：112 源 A，88 成功；45 合规历史提供 90 个查询。
  Novel native/raw/GEM=15/45、9/45、15/45；Revisit=18/45、45/45、45/45。
  这是近距离高支持查询，不能与 Table I 的 28-history 总体混算。
- Final14 精确 SPL：原 21 histories /210 rollouts 全部完成，summary job `17089991`
  COMPLETED，`summary_verified.json` 为 verified=true。
  活动论文的 Table III 已是含末步的 SPL 单值，不再是旧 bounds；它仍不是 9 月 10 日修复执行器的同版重跑。
- Table II：当前 A=131/196、B=54/183；条件 C-Novel native/GEM=4/20；
  C-Revisit=8/20→17/20。最近做的是原始记录归因，不是新 mixed-B/C 导航。
- 长程 route-tangent：旧执行版本、受控 causal RGB survey，不是实际 NavDP-A。
  native/endpoint/tangent=0/23、0/23、4/23；+4/−0，p=0.125，未通过既定确认条件。
- 后续 16-history 长程 oracle：本轮 sacct 再确认 `16817050_32..47` 全部 FAILED，
  `16817051` FAILED、`16817052` CANCELLED；不能声称已隔离并解决漂移/路线/controller 的归因。

### Final14 精确 SPL 最终读出

本轮读取远端 `summary_verified.json` 并与活动论文 Table III 核对一致：

| 条件 | Novel | Revisit | 合计 SR | 实际路径 SPL |
|---|---:|---:|---:|---:|
| Metric native | 6/21 | 3/21 | 9/42 | 0.1061 |
| Zero-depth native | 3/21 | 1/21 | 4/42 | 0.0602 |
| Mono native | 6/21 | 6/21 | 12/42 | 0.1320 |
| Metric GEM | 7/21 | 20/21 | 27/42 | 0.4593 |
| Mono GEM | 8/21 | 20/21 | 28/42 | 0.4941 |

Mono GEM 相对 mono native 总体 +17/−1，p=0.00014496；相对 metric GEM +2/−1，p=1。
这不是旧 26/42 metric GEM 与旧 native 分母读出的原样拷贝，也不能把新 SPL 和旧 SR 拼接。
目标 A 历史仍来自原 metric A；补跑解决末步计量，不是新 full-mono A 或新物理执行确认。

源：`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_table3_exact_spl_20260907/summary_verified.json`。
SHA256：`ba78af3f84f4eecacf5226805e0c540d87340fe4f5c374d82d7fbe13698b4e7f`。

## 6. 本轮结论与操作边界

最新实质进展是：三种 controller 的 HM3D 记忆收益均已在修复 query 链完整复现；
去面积消融全量完成，定位了可恢复的一部分拒绝损失；MASt3R 组件读出已经结束。

尚需收尾的是 Table I 的 45 个排队项和两个图像绑定失败项；旧长程归因仍未完成。
没有启动新的 Table II、增加模型、改阈值、重新开始学习或把快照写进论文。
本轮只新增此汇总文件，不覆盖旧时点文档、不 commit/push。
