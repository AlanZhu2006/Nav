# CEC 项目完整复审：2026-09-06

审计时间：2026-09-06 05:00–05:45，Asia/Shanghai。  
范围：当前实际运行的架构、论文与会议清单、关键原始结果、最新 HPC 任务、版本边界与复现风险。  
方式：只读审计；未修改方法或论文，未提交/取消 HPC 作业，未操作机器人，未 commit/push。只新增本报告与本地审计产物。

## 0. 结论先行

**项目的主线有效，但当前稿件还不能直接作为数值全部核验完成的投稿版。** 原因不是 training-free，也不是缺少更复杂的公式，而是本次发现了一个真实的路径长度/SPL 计量问题，另有一项重要机制补实验因基础设施错误未完成。

本轮重新查证的六件事：

1. **主表的 SR 没有被推翻。** 从原始逐查询 CSV 独立复算，Table I 四组跨 controller/dataset 配对和 Table II 的 C 查询，成功数、增减数、McNemar 值全部与稿件一致；fresh full-mono HM3D 也一致。共检查 404 个 CSV、808 条 arm-role 记录，不把它们当作 808 个独立场景或 episode。
2. **新发现：旧 `path_len_m` 不总是实际位移。** 旧 pursuit executor 返回指令步长，navmesh snapping 后仍按该步长记账。Table II Revisit 的 CEC SPL，旧字段复算为 0.674960，而逐帧位置加真实末位姿复算为 0.761541。它不改变 SR，但要求检查和修正论文的路径/SPL 数字。本报告没有自动替换表格。
3. **HPC 此刻没有排队或运行中的作业。** authority-spectrum 是 20/28 完成、8/28 失败、分析取消；不是“还在慢慢跑”。
4. **最新 U-turn/rear-alignment 已跑完，却没有解决长程。** 9 histories / 6 scenes，3/9→4/9，+2/−1，p=1.0；转向臂发生 4 条 geometry stop。这是已消费失败集上的机制结果，不是 fresh confirmation。
5. **会议核心四张仿真表确实存在。** 但不能说会议任务全部完成：真机正式结果仍缺；长度研究使用了替代的 controlled causal survey，并且存在距离与跨层结构混杂。
6. **主仓库同步正常，开发工作树与论文工作树仍有未发布改动。** AlanZhu2006/Nav 对应 `fork`，不是 `origin`。论文 PDF 编译通过，但 Fig. 1 存在真实视觉重叠。

建议顺序：**修正计量 → 补齐 authority 缺失任务 → 整理论文图表与版本 → 再决定是否投入长程新方法。** 不建议再用训练、更多 controller 或新 benchmark 分散当前工作。

## 1. 审计对象与证据等级

### 1.1 三个目录不能混用

| 对象 | 真实路径/版本 | 本轮状态 |
|---|---|---|
| 研究工作树 | `/home/asus/Research/Nav-graph-blind`，`2a309b6` | 22 个 tracked 修改、272 个 untracked 文件；大量长程分支未提交 |
| 已发布代码 main | `/home/asus/Research/Nav`，`1aac77e` | 干净；实时查询 AlanZhu2006/Nav 的 `fork/main` 同为此提交 |
| 当前论文 | `/home/asus/Research/Memnav_Paper`，`2a8b75b` | 远端 main 与本地提交一致；另有 8 个 tracked 文件未提交 |
| 旧论文副本 | `Nav-graph-blind/paper/` | 会议计划仍从这里读取，但其 LaTeX 不是当前稿件 |

代码仓库 `origin` 指向 glbreeze/Nav 的旧上游；其 main 为 `088d370`。这不能用来判断 AlanZhu2006/Nav 是否同步。此次用 `git ls-remote fork refs/heads/main` 确认了目标仓库。

当前论文编译入口为 `main.tex`，bibliography 为 `main.bib`。Method/Problem 实际读取 `sec_lg/3_problem.tex`、`sec_lg/4_method.tex`，而不是闲置的 `sec/3_problem.tex`、`sec/4_method.tex`。`main_lg.tex` 不是本次编译入口。

保护项：标题、Abstract、Introduction 均未编辑；现有 dirty changes 全部保留。

### 1.2 本轮核验到哪一层

- **原始记录复算：** Table I 四组、Table II C、fresh full-mono HM3D 的 SR/配对；Table II C 的逐帧路径；最新 HPC 队列、退出状态、失败日志。
- **原始汇总 + verifier 核对：** latest rear-alignment、metric-distance、same-floor longrange、Final14 attribution。
- **既有实验档案复核：** 更早的 Fresh160、learned ablations、Novel oracle、X-NavDP、GOAT/Replica 等。没有重跑所有旧实验，也不把旧档案复述称为本次独立复现。
- **当前实现诊断：** 核心 CPU 测试、canonical 路径代码、论文重新编译和 PDF 人工查看。没有启动新的 GPU 闭环评测。

证据优先级：原始 trace/CSV 与冻结运行代码 > 独立 verifier > 汇总 JSON > 状态文档 > 历史聊天。

## 2. 当前真正的主架构

一句话：**由同一条因果 RGB 流维持可寻址的视觉历史和冻结的流式几何，提供短程深度与经验证的历史方向，接入一个冻结导航策略。**

```text
causal RGB observations
  ├─ explicit RGB archive + raw DINOv2 descriptors
  │      └─ goal-to-history top-8 retrieval, temporal NMS
  └─ frozen LingBot streaming geometry
         ├─ current relative depth × first-40 camera-height scale
         │      └─ NavDP observation depth
         └─ historical depth + historical/current camera estimates
                └─ SP/LG + epipolar ranking + one PnP witness
                       ├─ certificate accept → directional record
                       └─ reject → native request unchanged

NavDP: current RGB + estimated depth + original ImageGoal
       + optional fixed-radius PointGoal 2.5 b
       → frozen goal encoders / diffusion / critic → trajectory
```

### 2.1 必须分清的三种状态

1. **显式历史 archive/index：** 保存已经观察到的 RGB、DINO descriptor、帧索引和几何关联。DINO 检索的是它，不是 LingBot 的一个网络 head。
2. **LingBot streaming state：** 维护因果几何上下文，供深度、相对姿态和历史 witness 使用。它不是一个直接支持 ImageGoal 内容检索的数据库。
3. **NavDP observation FIFO 与推理状态：** 控制器自己的短上下文、goal 和随机数状态。CEC reject 的一致性不仅关乎最后一个向量，还关乎这些状态是否保持原生语义。

不能写成“所有长程记忆都在 KV cache 中”。也不应把 rosbag 当成核心在线寻址结构；rosbag 是录制/回放载体。

### 2.2 Dense 分支

`MemNavData/monocular_depth_runtime.py` 和 `policy_agent.py` 的 first40 路径：

- 第 0–39 帧建立唯一的因果尺度估计；
- 第 40 帧起使用 `clip[0.8,6.0](1.15 × camera_height / estimated_height) × relative_depth`；
- warmup 或尺度无效时给显式 zero depth；
- RGB、深度和 transaction 绑定，不允许旧帧深度混入新请求；
- 此处的“单目”指没有深度传感器输入，不是内部完全没有深度或没有 metric calibration。

40 是本项目冻结的尺度估计协议，不是 LingBot 在第 40 帧才开始理解几何的网络定理。高度先验解决公共尺度问题，并不保证每一段累计相对位姿准确。

### 2.3 Sparse 分支

Canonical 参数和控制接口与稿件匹配：

- raw DINOv2-L CLS、1024 维；top-8，四帧 temporal NMS；
- 当前目标首次查询时冻结可用历史边界，不将追逐该目标时后来收集的帧冒充已有 Revisit 支持；
- SuperPoint/LightGlue 建局部匹配，Fundamental-MAGSAC 排序；
- 最高候选构成唯一待检验 place hypothesis，不是逐个 PnP 到成功为止；
- 历史 LingBot depth lifting + PnP-RANSAC；
- `inliers ≥ 16`，双侧 hull ≥ 0.05，RMSE ≤ 2 px；
- 接受后从当前 camera frame 得到 `[forward, left]` 单位方向，NavDP 半径固定 2.5 m；
- 正常拒绝返回 native ImageGoal 请求；geometry stream 故障是 stop，不等于正常 reject。

旧 `gatecurr` checkpoint/learned retrieval 代码仍在大类中被加载或产生诊断量，但 canonical certified 候选与授权不由该 learned gate 决定。因此当前方法可称 **pretrained and frozen / no task-specific training**，不能因此声称“没有 embedding”或“所有组件都是手写算法”。清理 legacy 加载是后续 release 工作，不是本轮擅自删除权重。

### 2.4 ViNT 与 NavDP 不是同一个低层接口

NavDP 消费原始 ImageGoal + 可选 2.5 m directional PointGoal。ViNT 没有相同 PointGoal 接口，使用有新观测的 ≤30° 分步朝向调整，然后把认证的历史 anchor image 作为 ImageGoal。

所以 Table I 证明 **相同记忆证据可适配两个冻结 controller**，不是两个 controller 逐字节接收相同输入，也不是 NavDP 和 ViNT 的绝对 SR 排名。当前 `sec_lg/4_method.tex:164` 已正确解释这一点，无需反复追加防守段落。

## 3. 现有主要结果：哪些真正成立

### 3.1 Table I：跨 controller × dataset

以下全部从原始 CSV 重新配对；场景数为 HM3D 21、MP3D 25。

| 数据集 / controller | Novel，Base→CEC | Revisit，Base→CEC | 全部查询，Base→CEC | Revisit +/− | exact p |
|---|---|---|---|---|---|
| HM3D / NavDP，28 histories | 6/28→6/28 | 8/28→25/28 | 14/56→31/56 | +18/−1 | 0.000076294 |
| HM3D / ViNT，28 histories | 3/28→3/28 | 3/28→19/28 | 6/56→22/56 | +16/−0 | 0.000030518 |
| MP3D / NavDP，42 histories | 17/42→17/42 | 9/42→37/42 | 26/84→54/84 | +29/−1 | 0.000000057742 |
| MP3D / ViNT，42 histories | 9/42→9/42 | 2/42→24/42 | 11/84→33/84 | +22/−0 | 0.000000476837 |

原始 `reached` 与 `<1 m` 的记录距离没有不一致；所检查的 runtime-failure/metric-depth-read 字段均为零。此处检查的是已有记录，不宣称重新运行 simulator 验证每一个位姿。

HM3D 没有 Novel takeover。MP3D 的 1/42 weak-support Novel 被两种 adapter 接受，但二元 outcome 未改变；另外 41/42 精确回退。**Novel SR 一样不等于所有 Novel 都从未被干预。**

### 3.2 Fresh full-mono HM3D：完整传感器链

28 histories / 21 scene clusters，实际 mono Goal-A 形成历史；与 Table I 是不同 query population，不能混写为同一组 native 数字。

| Arm | Novel | Revisit | 合计 |
|---|---|---|---|
| mono native | 8/28 | 9/28 | 17/56 |
| mono raw fixed | 4/28 | 24/28 | 28/56 |
| mono CEC | 8/28 | 24/28 | 32/56 |

CEC 对 native：+16/−1，p=0.000274658。对 raw：合计 +6/−2，p=0.289063；Revisit 单独为 +1/−1。这里有完整 mono composition 的有效证据，没有 CEC 显著抬高 raw-Revisit 上限的证据。

### 3.3 Table II：连续第三目标，不是无条件三段 joint

- factual A：131/196；
- factual B candidate trials：54/183；多个候选共享 A，所以不能与上一比例直接相乘；
- 成功且有支持的 A+B 后，20 histories / 13 scenes：
  - Novel-C：4/20→4/20；
  - Revisit-C：8/20→17/20，+10/−1，p=0.01171875；
  - balanced C queries：12/40→21/40。

C 的以上 SR 已重新读原始 CSV 复算。**SPL 则见第 4 节新发现，不能再用旧 verifier 通过替代实际路径核验。**

它支持“多个目标之后仍能利用累计历史”，不支持直接宣传 17/20 的 end-to-end 三段全成功率，也不是完整五段自主导航的确认结果。

### 3.4 Tables III–IV：Final14 归因

共同的 21 histories / 10 scenes / 42 queries；历史 A 来自 metric-depth rollout。因此是 query-stage attribution，不是 full-mono 系统总表。

Depth 五臂：metric native 11/42；zero native 4/42；mono native 10/42；metric CEC 26/42；mono CEC 28/42。两种 CEC 的 Revisit 都是 20/21。另一个 first-goal 比较 mono 27/40、metric 30/40，未达到预定义 non-inferiority 门。

Sparse 三个重点臂：raw 23/42，finite-PnP 25/42，strict CEC 28/42；Revisit 都是 20/21。Novel intervention 分别是 21/21、18/21、2/21。

CEC 对 raw +5/−0、p=0.0625；对 proposal-matched finite-PnP +4/−1、p=0.375。机制归因方向清楚，但不能把它们写成已确认的额外 SR 优势。不同 proposal 的 raw top-1 与相同 top-8 的 finite-PnP 不是同一个控制变量对照。

### 3.5 旧成果怎样保留

| 旧方向 | 合理定位 |
|---|---|
| geometry memory 4/40→19/40 | 记忆有用的早期基线证据，不替代最新 mono/formal 表 |
| Fresh160：27/120→112/120，raw 106/120 | 高支持 Revisit 内部复现；CEC 对 raw +9/−3、p=0.146，不是确认性上限提升 |
| Novel oracle 28/40→40/40 | 特权方向可恢复性的机制上界；不是可部署 Novel 方法 |
| CDEC、GCT、Pi3X 替代 | 分别暴露 actionability、长程寻址、错误 bearing 授权的问题；不是“学习不可能”的证明 |
| X-NavDP/MPC 小规模对照 | 未建立额外 controller 收益；不代表普遍能力等价 |
| GOAT / Replica | GOAT 存在目标与动作/STOP 协议差异；Replica 是当前构造合同不可满足；不能编为外部成功结果 |
| graph rescue / 各种 route clock | 开发机制；未通过独立确认，不并入 canonical endpoint CEC 的正式 SR |

## 4. 本轮最重要的新发现：路径长度和 SPL

### 4.1 根因已经落到代码，不只是猜测

冻结运行代码中，`pursuit_step` 执行：

```text
cand = pos + v * forward
snap = pathfinder.snap_point(cand)
return snap, new_yaw, v
```

若允许 snapping，则第三个返回值仍是 `v`（creep 分支是 `0.3*v`）。但实际移动长度应是 `||snap_xz - pos_xz||`。随后 `path_len += dl`，导致在贴墙/吸附/部分阻挡时记账与实际轨迹分离。

本轮不仅查看 HEAD，还读取了实际 authority runtime bundle `hm3d_table1_navdp_authority_transaction_repair_51ee9a4ca063c7f1` 中的旧实现，确认它返回指令步长。当前研究工作树 `MemNavData/eval_2leg_habitat.py:2815` 已有返回实际位移的修改，**但修改当前代码不会自动改正旧结果**。

原有 stage-SPL verifier 校验 `CSV path == trace.path_len`，再用该字段重算 SPL。两者来自同一个旧累加器，故能一致地通过；它没有独立对轨迹 positions 求和。

### 4.2 Table II C 的原始轨迹重算

使用所有 80 条 arm-role 记录，逐帧检查 trace 从 step 0 开始且连续，并从 `end_x_m/end_z_m` 补齐最后执行一步。末位姿字段 **80/80 存在**。按当前平面执行路径口径：

| 组别，每组 N=20 | 旧 CSV 路径所得平均 SPL | 逐帧位移 + 真实末位姿所得平均 SPL | SR 是否改变 |
|---|---:|---:|---|
| Native Novel-C | 0.172272538 | 0.175580274 | 否，4/20 |
| CEC Novel-C | 0.172272538 | 0.175580274 | 否，4/20 |
| Native Revisit-C | 0.142183244 | 0.148914175 | 否，8/20 |
| CEC Revisit-C | 0.674960108 | 0.761541225 | 否，17/20 |

CEC 的 17 条成功 Revisit 中，14 条的旧路径与实际轨迹相差超过 0.10 m。全部 Revisit 中最大差异约 8.20 m。因此不能归因于浮点误差或漏一个终止帧。

**上表是本轮诊断重算，不是已经替换并发布的论文结果。** 后续需对 A/B、Tables III–IV 和其他实际报告 SPL 的 population 采用同样的独立计量流程，并说明 planar/3D path 约定。不能只更新对 CEC 有利的单个数字。

Fresh full-mono 中也能看到同类旧路径偏差；但它较早的 query CSV 没有同样完整的末位姿列，需要追溯其他原始产物，不能拿少一帧的折线长度直接当最终精确 SPL。

### 4.3 影响与最小修复

- 已复算的 SR、gain/loss、McNemar 不因该记账错误改变。
- 路径长度、SPL、基于记录路径的效率解释需要重新审计。
- 不能据此宣布所有导航轨迹无效；也不能据此保留错误 SPL。
- 优先 **只读原始轨迹离线重算**，无需重新执行全部 GPU 策略。
- 更新 verifier 的独立性：位置增量求和应与旧计量字段分别产生，不再仅复制同源字段。
- 若某组原始末帧不足，先报告无法精确复算，再由作者选择恢复原始产物、给出界限或暂不报告该 SPL；不得推测填值。

## 5. HPC 实时对账

### 5.1 连接没有权限故障

遵守 `MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md`：使用 `alantorch` 的现存 ControlMaster/socket，身份为 yz11502。无 PTY 的 command channel 可能挂起，但共享连接上的交互 PTY 可用。本轮没有让用户重新登录，没有关闭 master；审计 shell 退出后 master 仍存活。

`squeue -u yz11502`：本轮查询为空。本机 GPU 未启动新任务。

### 5.2 Authority-spectrum：20/28 完成，8 条是运行错误

冻结 population：28 histories / 21 scenes，四臂 native/raw/finite-witness/CEC。

- smoke `16929024`：完成；
- formal array `16929030`：20 completed、8 failed；
- 失败 indices：`2,3,6,7,10,13,18,21`；
- analysis `16929036`：因依赖失败取消。

其中七条为同一根因：**array 按 history 分片，但 runtime 目录按 scene 命名**。`slurm_hm3d_table1_authority_spectrum.sbatch:50` 把 history 转成 scene rank；`run_hm3d_fullmono_server_scene.sh:127` 生成 `eval_${SCENE_INDEX}`。同场景第二条 history 因目录已存在而退出，不必并发也会复现。

index 2 是另一条已有接口问题：finite-witness 臂回到 native 请求时，NavDP 的 cached monocular depth 不属于当前 transaction，HTTP 500。实际 server bundle 与已修好的 runtime closure 未统一，日志为 `cached monocular depth belongs to a different transaction`。

这不是 GPU 利用率低、CUDA 随机性或数据不足。需要明确的两项修复：每 history 独立 runtime 身份；复用已验证的 transaction 实现并验证实际 import 来源。

下一次执行应保存完成的 20 条，仅重跑缺失的 8 条完整同进程配对块；先验证同场景两条 history 和重复图像/新 transaction 的真实服务路径，再提交缺失数组。**本轮没有实施修复或读取未完成总体的科学 SR。**

### 5.3 Rear-alignment：已完成的结果

- gate `16929826_8` 完成；
- formal `16929827` 的 9 条全部完成；
- analysis `16929828`、independent verifier `16929829` 完成；
- verifier `verified=true`。

| 条件 | 成功 |
|---|---:|
| route tangent，unaligned | 3/9 |
| route tangent，rear-aligned | 4/9 |

配对 +2/−1，p=1.0；6 scenes。5 条 parent-stuck 里救回 1 条；4 条 parent-success 中 +1/−1。54 个 ≤30° yaw atom 都有 fresh observation 和 control-edge receipt，转向期间没有错误调用 visual PnP；但 aligned 后续出现 4 条 geometry stop。

汇总内写明 `navigation_claim_allowed=false`、`significance_claim_allowed=false`、`fresh_confirmation_required=true`。它的 `advance...` 决策仅表示小机制门满足，不表示已建立导航收益。

原始结果：

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_rear_alignment_20260904/consumed_a86e3d9b07234fd6/result/summary.json`

本地证据副本：`.diagnostics/project_reaudit_20260906/hpc/rear_alignment_summary.json`，SHA-256 `4beb775d140535d14fec554c849cac4188e87d136a5fdb86c7563af95610c9d9`。

## 6. 长程为什么失败：目前能说到哪里

### 6.1 不能把三个不同问题混在一起

| 问题 | 已有证据 | 尚不能推出 |
|---|---|---|
| 公共尺度 | 高度标定后，中短程 metric-distance audit 的距离误差较小 | 任意长历史的相对位姿都不会漂移 |
| 路线信息 | endpoint bearing 在长、绕行、跨层路线不足；local tangent 在同层集救回 4 条 | 固定半径本身是唯一问题 |
| 执行/定位闭环 | 后向 PointGoal 有 forward clipping；route-clock 有低视差 PnP 与进度漂移问题 | 都是原始 LingBot KV 漂移，或一个 U-turn 就能解决 |

高度先验只校准尺度；共同 Sim(3) 不变性只消掉共同 gauge。二者均不消除不同时间的相对旋转/平移误差。

NavDP 仍会局部规划避障，2.5 m bearing 不是强制直线运动。但反复给终点方向不提供“先背离终点、绕墙/找楼梯”的历史路线顺序，这与 local controller 是否会短程避障是不同问题。

### 6.2 现有长程结果的边界

原始 48-history controlled survey 的 Revisit：近段 native 2/16、CEC 4/16；中段 1/16、2/16；远段 2/16、0/16。它不是 NavDP 实际完成 A 后的历史。距离越长，跨楼层比例也显著增高，所以不应视为隔离了距离的纯因果消融。

后来的同层 23 histories / 8 scenes，初始 geodesic 约 20.37–24.66 m：native 0/23、endpoint 0/23、monocular route-tangent 4/23；+4/−0、p=0.125。14 条 tangent geometry failures 中，13 条是 PnP 不成立，1 条不足 16 inliers。这里相当部分是新增相邻帧 route-motion estimator 在低视差/纯旋转下失败，不能全归咎于 LingBot 主 stream。

9 条没有 geometry stop 的运行中有 4 条成功、5 条 stuck；至少两个后向 deadlock 与 `process_pointgoal` 将 forward 裁剪到非负有关。最新 rear-alignment 又表明：修复这一动作问题不能自动修复转向后的几何。

### 6.3 Metric-distance 的负结果不应过度解释

在中短程 28-history 对照上：fixed bearing 25/28，metric distance 24/28，+1/−2、p=1.0。这支持“恢复距离暂未带来可验证额外控制效用”，不支持“metric 一定比 bearing 差”。也不能拿这 28 条证明 20–30 m 的方向/位姿没有误差。

### 6.4 下一步长程研究应验证什么

先不要改变 controller、再加 matcher、再扩大几十条新曲线。优先读现有日志分清：

1. 初始 witness 正确、当前位姿仍可靠，但终点方向无法绕行；
2. witness 正确但后续 current-to-history 变换失真；
3. route target 合理，但 frozen controller 的后向/障碍处理不兑现；
4. 新增 route-clock 在自身几何失败时停止。

若需要新机制对照，最有信息量的是固定同一历史、目标、几何估计和 controller，只比较 endpoint cue 与正确局部路线 cue，并把 privileged 方向臂明确标为诊断。不要把 oracle 好结果重新包装为可部署方法。

本轮不建议立即扩大 rear-alignment：先解释 4 条 geometry stop 是否为同一可修复因果链，否则扩样只是更精确地测当前失败。长程应保留为独立研究分支，不让它重写已验证的短/中程 episodic-memory 主线。

## 7. 会议清单的实际完成程度

控制文件：`paper/会议实验优先级与执行清单_2026-08-27.md`，不是旧状态文档中随时变动的建议。

| 会议项 | 当前判断 | 剩余问题 |
|---|---|---|
| Table I，NavDP/ViNT × HM3D/MP3D | 完成，SR 本轮全量复算 | 保持 within-controller 解释；不是万能 controller 结论 |
| Table II，A/B/C 连续目标 | shared-prefix 版本完成 | C 是条件结果；SPL 必须修正 |
| Dense 五臂 attribution | 已完成并在 Table III | query-stage，不混称 actual-mono A；SPL 扩展审计 |
| Retrieval/CEC/Oracle + mechanism | Table IV 已有；matched finite-PnP 已有 | 新 HM3D authority 补实验尚缺 8 histories；不声称所有机制尚未做过 |
| 长度分桶 | survey stress 结果已有 | 替代历史合同、跨层混杂；不是原始 actual-online 长度主表完全闭合 |
| 正式真机 | 尚无合格闭环正式结果纳入 | 按用户要求，本轮不操作、不扩展真机工作 |

所以，“四张核心仿真表齐了”是对的；“会议要求全部完成”“长程已解决”“五段完全自主 SR 已确认”都不准确。

`Memnav_Paper/CONFERENCE_EXPERIMENT_MATRIX.md`、`EVIDENCE_LEDGER.md` 仍包含旧队列状态，应更新成封账结果/失败结论。无需把 job ID 塞进正文。

## 8. 按人类审稿注意力重看论文

本轮使用 `icra-human-review` skill，采用先看标题/Abstract/Introduction/主图/主表/结论，再验证关键疑问的顺序。只保留三项可能改变判断的问题；这是作者侧、同上下文复审，不是独立或真实审稿意见。

### 8.1 首读贡献可以稳定复述

**把冻结流式几何与同源 RGB 索引组成可查询的 episodic memory，用 dense monocular depth 和 sparse verified direction 接到冻结 goal-conditioned policy。**

它不是新 matcher、不是一个“训练一定更差”的理论、不是动作级 mixture-of-experts，也不是 lifelong 全覆盖证明。当前 Abstract/Introduction 和最新 Method/Conclusion 的主题基本一致；不需要再围绕 proof/gating 改写整篇，更不需要为了像论文而堆高阶公式。

三个实际优点：

1. 两个数据集、两个 frozen controller 的配对增益清楚且已复算。
2. full-mono 的实际 A-history 证据已补上，不再只有 query-time mono。
3. 隐藏 role、共享 prefix、原生请求保持的实验设计，能定位记忆贡献，不是简单跨实验拼 SR。

### 8.2 三个重要问题及比例适当的处理

**问题一：效率指标的计量正确性。**

- 位置：`sec/5_experiments.tex:19`、Table II、`sec/6_results.tex:36`。
- 事实：正文称 SPL 来自 executed path；本轮发现旧记账与真实位移不一致。
- 审稿影响：这是实质 numerical correctness，不是表述偏好。发现后不处理会削弱整份 evidence ledger 的可信度。
- 最小处理：离线统一重算实际路径，更新受影响表/文，不换方法，不优先全量重跑策略。

**问题二：方法相对最近简单替代的独立价值。**

- 位置：Table IV、`sec/6_results.tex:62`、`sec/7_analysis.tex:23`，以及 Related Work。
- 事实：CEC 对 native 增益强；对 raw/finite-PnP 的额外 SR 尚未确认。AnyImageNav 已提出 training-free geometric query registration/self-certification，并明确有 depth/odometry 依赖。
- 审稿影响：如果贡献写成“几何认证本身全新”或“全面优于简单 memory”，会很容易被反例挑战；当前稿已经较好地避免了后一种夸大。
- 最小处理：完成既定 HM3D matched authority 对照，保持传感器/状态/控制接口差异清楚；不要用不同任务的 published SR 横减，也不必突然补十个无关 planner。
- 来源：[AnyImageNav 正文，尤其 §3、§5](https://arxiv.org/html/2604.05351v3)。本轮没有上传本稿或用未公开内容搜索。

**问题三：使用范围与系统代价。**

- 位置：`sec/7_analysis.tex:32`、`sec/8_limitations.tex`。
- 事实：主查询约 2–9 m、高支持 Revisit；first recall P90 约 16–19 s；没有正式物理闭环；空间长程未解决。
- 审稿影响：若被读成任意长程、实时、完整自主 arrival 的通用系统，证据不足。但这些边界当前已经在 limitation 说明，不需要每段再重复。
- 最小处理：保留 episodic reuse 的中心，给出可读的系统图和真实 recall/执行视频；是否做远程/真机扩展由作者决定，不把它当作所有系统论文的统一硬门。

作者侧暂评：按当前 ICRA rubric 为 **B− / 3.5（borderline）**，不是录用概率。置信度中等：主结果和代码证据充分，但最近外部方法没有同协议闭环对比，且新发现的计量问题还未正式修复。最影响下一轮判断的是正确的 SPL 封账与最近简单基线的完整比较，而不是再增加数学密度。

当前官方分档依据：[IEEE RAS ICRA reviewer guidelines](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-reviewers/)。

### 8.3 真正需要修的排版，和无需再折腾的内容

- **Fig. 1 在 PDF 第 4 页的 state / dense readout 矩形重叠。** `figures/cec_architecture.tex:37` 附近的固定坐标与自动宽度造成遮挡；不是编译器问题。应重新布局并保持字号，不用缩小来遮掩。
- RGB/Goal 到 policy 的直接通路目前主要靠 fusion 标签表达；补清楚连线比加文字更有价值。
- Table I 虽占双栏 float，却使用 scriptsize 且表体较窄，caption 较长。放大表体、精简重复说明即可；不是缺表。
- Tables II–IV 在第 6 页均存在。布局较密但未发现越界或裁切，不需要再次整体恢复旧表。
- Abstract 偏长且有“non-overlap image carries nothing”等绝对化动机句。它们可以留待作者做小幅语气确认；本轮未改 Abstract/Introduction，也不以写作偏好判定技术无效。
- 当前 35 个活动 citation keys、20 个 labels 均解析；37 个 BibTeX entries 中 2 个未用，这不是投稿 blocker。没有为凑相关工作新增引用。

## 9. 构建、测试、规则与匿名性

### 9.1 当前稿件独立构建

在隔离副本使用已存在的 TeXLive Docker 镜像、禁用网络：

```sh
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
```

宿主未安装 latexmk；副本使用标准 `placeins.sty`，没有改项目模板或升级依赖。

- 成功，8 页，US Letter，PDF 1.4；
- 0 undefined citations/references，0 duplicate labels；
- 最终 log 没有 LaTeX Warning / overfull / underfull；
- 字体嵌入，未发现活动稿件身份信息；
- 人工查看首页、方法/公式页、架构图页、表格页与参考文献起始页；图形内部重叠仍存在，说明“无日志警告”不等于排版已通过。

审计 PDF：`.diagnostics/project_reaudit_20260906/paper/main.pdf`；SHA-256 `8720fc9f2f2111f7ddcb264753439749425031f99a23cc96b8deeaf546edaf1a`。这是未修改稿件的审计构建，不是可直接上传的推荐终稿。

### 9.2 核心测试

15 个核心测试模块，共 **172 passed / 0 failed / 0 skipped**，包括 certificate、PnP、first40 depth、bearing、goal-switch、causal pairing、controller handoff 和新增研究分支的关键 contracts。

这证明被选中的本地逻辑通过，不证明所有 GPU runtime bundle 无误，更不证明旧 CSV 中的物理计量正确。authority-spectrum 恰好说明了“本机测试通过、正式 runtime 拼装错误”的区别。

### 9.3 ICRA 2027 当前官方规则

2026-09-06 官方页面实时核对：contributed paper 总计 8 页，含参考文献；双匿名、双栏；额外文字补充材料也必须在这 8 页内。视频附件上限 20 MB、180 s，≥480 高、≥20 fps。论文截止 2026-09-15；视频窗口为 08-05–09-09 和 09-17–09-22，09-10–09-16 暂停视频上传。

来源：[ICRA 2027 Call for Papers / submission FAQ](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)。该站另一个 final-paper 页面残留 2025 的 6+n 规则，本轮未拿它约束 2027 初稿。没有向 PaperPlaza 上传任何内容。

未完成的外部提交检查：官方 PDF Test、作者/关键词、AI disclosure 准确性由作者最终确认；不存在“未做真机就自动 desk reject”的已确认官方规定。

## 10. 下一步只做高价值事项

### P0：投稿前必须关掉的问题

1. **路径/SPL 封账。** 对所有入稿 SPL 按原始位置重新计算；先 Table II，再 dense/sparse attribution；明确末帧与 planar/3D 约定。禁止只更新有利数字。
2. **完成既定 authority-spectrum 的 8 个失败 blocks。** 修 history/scene namespace 和 transaction source，保持其他 20 个完成结果不动。先实服务最小验证，再按原 manifest exact retry。
3. **修 Fig. 1 重叠并重新构建。** 不改标题、Abstract/Introduction，不增加防守段落。用新的 PDF，而不是仓库中旧 `main.pdf`。
4. **给论文 release 建唯一来源映射。** 稿件、源码 bundle、checkpoint、manifest、CSV、verifier、真实位移重算各自对应；canonical 与 longrange development 不混成一个实验方法。

### P1：强烈建议，但不需要盲目烧 GPU

- 更新会议 matrix/ledger 中过时的“running / pending”；纳入 latest rear-alignment 失败结论。
- 同硬件分开报告首次 recall 和 cached update；不能将亚毫秒 cached projection 当成完整系统延迟。
- 从正式 paired traces 做一份紧凑成功/失败视频，说明真实 memory recall 与控制过程。
- 检查首读动机中过强的一两个词；若作者同意，只做最小修饰，不重写整篇。

### P2：独立研究问题

- 可靠的长程 route progress 和绕行 cue；
- 有界索引/几何缓存与首次 recall latency；
- 自动 arrival、真机与五段以上完全自主序列；
- learned relocalizer 的 actionability-aware 目标。

这些有研究价值，但不是本轮“让论文更像论文”所必须添加的模块。训练失败没有证明 learning 不可能；training-free 有效也不需要靠训练来获得发表资格。

## 11. 可复核产物与边界

本轮审计产物目录：`.diagnostics/project_reaudit_20260906/`。

- `recount.py`：HPC Table I/II 原始 SR 配对复算；
- `path_recount.py`：Table II C 逐帧位置 + `end_x_m/end_z_m` 路径复算；
- `tableII_executed_path_recount.json`：80 条记录的实际位移复算摘要、计量差异与样例；
- `core-tests.xml`：172 项测试结果；
- `paper/`：当前稿件隔离构建、log/PDF/参考文献；
- `page1.png`、`page3.png`、`page4.png`、`page5.png`、`page6.png`、`page7.png`：人工图面检查；
- `hpc/rear_alignment_summary.json`：最新完整机制结果。

原始关键 run roots 均在 `/scratch/yz11502/Research/Nav-axis-uturn-results/` 下：

```text
hm3d_table1_controller_portability_20260829/
  navdp_server_namespace_repair_20260829T054615Z_51ee9a4c/formal/navdp
  formal_20260828T231109Z/formal/vint
mp3d_table1_controller_portability_20260829/
  formal_20260829T085025Z/formal/{navdp,vint}
hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/
  formal_20260830T045416Z_1f4979a7/table2_leg3_power/
    policy_authority_closure_repair_v3/formal/navdp
hm3d_table1_authority_spectrum_20260904/formal_9bb6fc2fcc16303b
hm3d_longrange_rear_alignment_20260904/consumed_a86e3d9b07234fd6
```

本轮没有承诺逐行审查数百个历史试验文件，也没有把已有 verifier 当成数学正确性的保证。最有价值的变化是：**将“主线 SR 是否成立”“SPL 是否真的测了实际运动”“补实验是否完成”“长程机制是否解决问题”分开回答，并各自找到原始依据。**
