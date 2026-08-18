# 项目完整总账：Certified Episodic Compass

更新时间：2026-08-18
文档性质：论文主线、代码边界、实验结果和负结果的统一入口。

## 1. 项目现在研究什么

准确任务是：

> 面向 lifelong navigation 的单目 sequential ImageGoal navigation 中，如何把机器人自己积累的因果视觉历史转化为可靠的 Revisit 控制信息。

它不是完整 lifelong navigation，也不是传统 SLAM loop closure：

- 传统 loop closure 通常以当前帧为 query，输出 pose-graph constraint；
- 本项目以未来 ImageGoal 为 query，判断它是否受到过去因果历史支持；
- 通过时只输出当前相机坐标中的 scale-free bearing；
- 证据不足时不输出“Novel”标签，而是逐动作精确保留 native NavDP。

### 1.1 任务术语

- **Novel**：当前 ImageGoal 没有足够可认证的历史视觉支持；并不等于语义上从未出现。
- **Revisit**：目标位置/视图受到此前 actual-online 视觉历史支持。
- **2-leg**：Novel A 后评测一个 Revisit B。
- **3-leg NNR**：Novel A、Novel B 后评测对 A 历史的 Revisit C。
- **role-free**：policy runtime 不读取 Novel/Revisit 标签；标签只用于评测分层。
- **exact fallback**：拒绝记忆时，seed、trajectory selection 和 executed trace 与 native 一致。

## 2. 冻结主方法

论文主方法为 **Certified Episodic Compass（CEC）**：

```text
actual-online causal RGB history + ImageGoal
    -> frozen DINO temporally-diverse top-8
    -> SuperPoint + LightGlue correspondences
    -> Fundamental-MAGSAC label-free geometry ranking
    -> LingBot historical monocular depth
    -> 2D--3D PnP relative pose
    -> atomic certificate
         PnP status = ok
         inliers >= 16
         query hull coverage >= 5%
         reference hull coverage >= 5%
         reprojection RMSE <= 2 px
    -> accept: discard metric scale, retain unit bearing
               project to fixed 2.5 m residual
               frozen NavDP PointGoal execution
    -> reject/error: exact native ImageGoal NavDP
```

它不显式训练 Novel/Revisit 二分类器。Certificate reject 只表示“当前历史假设不能自认证”。

### 2.1 为什么只传 bearing

LingBot 的跨段单目 translation scale 没有被 certificate 证明，因此主方法不把它当成米制距离。验证过的向量只归一化为方向，再投影到冻结的 2.5 m residual。这样把定位模块和 controller 之间的接口缩减为最小控制变量。

### 2.2 为什么还保留 DINO

直接把几百帧历史交给 candidate-free transformer 已被否决：true Revisit-C 上，DINO-addressed GCT 为 18/20，full-prefix anchor-free GCT 只有 5/20，配对 +0/-13，p=0.000244。长程内容寻址仍需显式地址簿。

## 3. 正结果总账

### 3.1 最早 geometry memory

- 20 MP3D scenes、40 个严格配对 2-leg episodes；
- native joint：4/40；
- geometry memory joint：19/40；
- 配对 +15/-0；
- exact McNemar p=6.10e-5。

这首先证明了“历史记忆本身具有闭环控制价值”。

### 3.2 Fresh160 supported Revisit

在共享 Goal-A 成功的 120 条历史上：

| 方法 | Revisit B given A |
|---|---:|
| native | 27/120 |
| old geometry | 91/120 |
| raw-DINO direct | 106/120 |
| CEC | 112/120 |

CEC 对 native/geometry 很强，但对 raw direct 只有 +9/-3，p=0.146。该集合 online max-covis 中位数 0.898，属于高支持、接近饱和 Revisit；不能用它证明 certificate 提高 Revisit ceiling。

### 3.3 Final14 fresh MP3D mixed-role

21 histories、10 scene clusters，每条包含一个 Natural Novel 和一个 standard Revisit query；runtime role visibility 为 none。

| 方法 | Novel | Revisit | 合计 |
|---|---:|---:|---:|
| native | 7/21 | 4/21 | 11/42 |
| raw fixed | 2/21 | 19/21 | 21/42 |
| geometry fixed | 9/21 | 18/21 | 27/42 |
| learned Pi3X | 8/21 | 19/21 | 27/42 |
| **CEC** | **8/21** | **20/21** | **28/42** |

CEC 对 native：

- Revisit +16/-0，p=3.05e-5；
- Novel +1/-0，p=1.0；
- 合计 +17/-0，p=1.53e-5。

CEC 对 raw fixed：

- 合计 +8/-1，风险差 +16.67 pp，p=0.0391；
- Revisit +1/-0，p=1.0；
- Novel +7/-1，p=0.0703。

这是第一次在 fresh mixed-role population 上证明 CEC 相比 always-on raw memory 有显著整体优势。增益来自减少 unsupported Novel interference，而不是继续抬高已饱和的 Revisit ceiling。

限制：natural population 冻结目标为 28 histories，实际只有 21，因此必须保留 underpowered 标签，不能事后补样把同一集合重新称作确认。

### 3.4 外部 HM3D Revisit transfer

从 outcome-disjoint HM3D val scenes 构造 36 intention-to-treat episodes；一个场景在 policy 运行前构造失败并明确保留 attrition。共享 Goal-A 成功 21/36。

| 方法 | B given A | joint |
|---|---:|---:|
| native | 7/21 | 7/36 |
| old geometry | 17/21 | 17/36 |
| raw fixed, oracle role | 18/21 | 18/36 |
| role-free CEC | 19/21 | 19/36 |

CEC 对 native 为 +12/-0，p=0.000488，增益覆盖 8/9 scene clusters。这证明 MP3D 上开发的主方法具有 HM3D 跨数据集 Revisit utility。

该协议只包含 Revisit，因此不能单独证明跨域 Novel safety；raw arm 还读取 oracle role，只是上限对照。

### 3.5 actual-online 3-leg NNR

19 条可构造 actual-online A/B prefix、8 scene clusters：

| C controller | SR_C given frozen A,B |
|---|---:|
| native | 5/19 |
| known-role direct | 14/19 |
| role-free CEC | 16/19 |
| CEC + equal stuck budget | 16/19 |
| CEC + graph rescue | 16/19 |

CEC 对 native +11/-0，p=0.0009766。Graph rescue 确实执行了 92 个历史 subgoal，但与 equal-budget CEC 完全持平，因此从主方法删除。

### 3.6 double Revisit

Fresh20：native joint 0/20，full memory 12/20，role-free CEC 17/20。它证明多个历史记忆可被保留和再次调用；但“更早记忆保留”主比较只有 12/14 vs 8/14，+6/-2，p=0.289，不能称为确认。

## 4. 机制结果

### 4.1 Novel oracle bearing

同机同进程 N=40：native Novel-A 28/40，oracle periodic yaw 与 oracle bearing+token 均为 40/40；配对 +12/-0，p=0.000488。

它证明方向是可恢复瓶颈，且 token execution 可以兑现 oracle direction；它不构成可部署 Novel 方法，因为 bearing 来自 Habitat geodesic oracle。

### 4.2 bearing 执行边界

早期 N=6 诊断得到约 ±30° 容差、45° 临界、60° 明显退化，以及 point-token 对后向角度的非线性。但 N=6 只可作为机制线索，不作为论文主统计结论。

### 4.3 Pi3X causal visual bridge

Current 与远端 anchor 无直接共视时，单独多视图求解会发生坐标断连。只使用已经真实观察过的中间 RGB 构造 b16 causal bridge 后：

- positive candidate <=30°：585/701 -> 659/701，+82/-8；
- raw-DINO top-1：130/155 -> 144/155，+16/-2；
- Pi3X proposal top-1：147/155。

这是有效的 learned mechanism，但最终 proof 没有通过部署门。

## 5. 学习路线与负结果

### 5.1 GLP / Phase-B

- Candidate ranking AUC：DINO 0.9103 -> learned 0.9535；
- 但 train 最优阈值 0.397，dev 最优 0.807，发生明显尺度漂移；
- Stage 2 为 72.7% vs 87.3%，未通过；
- 结论：学习特征适合“选哪个 anchor”，不适合直接决定“要不要接管”。

### 5.2 CDEC

- OOF top-1：learned 128/155、geometry 126/155、DINO 115/155；
- 进入真实 PnP/certificate 后，geometry actionable 122，CDEC 115；
- CDEC-only 相对 geometry +1/-8，p=0.039，显著更差；
- geometry-first、CDEC-on-reject 在 349 次 fallback 中只增加一个 actionable session。

因此不进行昂贵闭环长训。

### 5.3 Pi3X learned proof

Final14 Natural Revisit：learned 19/21，native 4/21，说明 learned arm 有真实 utility；但 CEC 为 20/21。

预注册判定：

- L1 对 native 有用：通过；
- L2 相对 CEC non-inferiority：未通过；
- L3 Novel proof safety：未通过；
- 479 个 accepted plan bearings 中有 69 个 >90°，p95 error 149.5°。

所以 Pi3X 是 promising but unqualified learned simplification，不能替代 CEC。

### 5.4 X-NavDP

相同 verified PointGoal 上：mixed 20/26、base PointGoal 20/26、official X+MPC 21/26；X 相对 mixed +2/-1，p=1.0。Controller replacement 不是当前主要瓶颈。

### 5.5 其他停止路线

- temporal top-K：18/40 vs 18/40，p=1.0；候选数量不是瓶颈；
- active-glance 最好 25/40，native 31/40；原地扫描/主动转圈停止；
- goal-blind frontier：4/5 vs 4/5；
- DINO frontier 排序早期 3/6，出现灾难候选；
- semantic-first vs geometry-first：25/28 vs 25/28，p=1.0；
- graph rescue：3-leg 16/19 vs 16/19；
- Replica：当前长程构造协议得到 0 个正式 histories，是 constructibility failure；
- GOAT first-ImageGoal：CEC 没有可执行 Revisit intervention；NavDP 与 GOAT 相机、目标图和离散控制合约也不一致，不作为主 benchmark；
- MemoNav/Gibson：公开 episode 缺 goal RGB/rotation 和 evaluator，不能产生可比较官方分数。

## 6. 当前可写与不可写的论文主张

### 可写

1. actual-online causal history 可显著恢复 frozen NavDP 的 Revisit failures；
2. scale-free bearing 是当前测试中足够的最小 controller interface；
3. always-on memory 会干扰 unsupported Novel；
4. proof-before-control 在 fresh mixed-role population 上改善 utility-interference，并显著超过 raw fixed aggregate；
5. runtime 不需要显式 role label；
6. Revisit utility 已从 MP3D 转移到 HM3D。

### 不可写

1. CEC 形式化保证安全或零 Novel 误接管；
2. CEC 在 Revisit ceiling 上显著超过 raw/geometry；
3. learned proof 已替代显式 certificate；
4. Novel direction 已被可部署地解决；
5. X-NavDP 提升了本项目整体 SR；
6. 已解决完整 lifelong navigation；
7. GOAT、MemoNav 或 Replica 已产生可比较正式分数。

## 7. 当前部署限制

- CEC 每个 query 首次 uncached median 3.40 s；
- Revisit 首次 median 10.64 s；
- p95 约 26.35 s；
- 后续 cached bearing update median 约 0.152 ms；
- stored history median 166 frames，最大 364。

主要长尾来自查询时对较晚 anchor 重放 dense causal LingBot depth。下一工程任务应是在线预计算 DINO/local features/depth，并以 0 proposal/certificate/bearing/action mismatch 做 decision-equivalence 验收。

## 8. 论文剩余工作

1. 冻结 CEC，不再搜索新架构；
2. 在全新 HM3D scenes 上做足够规模的 mixed Novel/Revisit 确认；
3. 完成 feature/depth cache 与 latency/memory scaling；
4. 生成 risk--coverage、utility--interference 和 failure taxonomy 图；
5. 条件允许时做 3 个真实环境的 role-free 真机闭环；
6. 发布 episode manifests、结果 receipts、代码 manifest 和依赖说明。

## 9. 最终论文定位

推荐标题/摘要术语：

> Causal episodic relocalization with proof-before-control for sequential ImageGoal navigation.

`lifelong navigation` 只作为动机；`goal-conditioned navigation loop closure` 可作为科学 hook；正式任务名应保持 `monocular sequential ImageGoal navigation with episodic revisits`。
