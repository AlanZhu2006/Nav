# CEC 项目全审计与最小重构：2026-09-06

本轮目标：对账代码、论文和原始实验；完成已冻结的机制检验；以实际结果决定是否改架构。
本文件是当日连续审计的最新入口。旧文档保留其当时状态，不覆盖原记录。

**本轮状态：审计与本机最小重构检验已完成。** 新HM3D 224条和本机12条记录均独立
复算通过；论文两项实质修正仍待作者确认，未擅自修改。长程正式oracle分解仍未完成，
不把本轮审计完成写成整个研究已经完成。

## 1. 最重要的判断

**项目不是“没有效果”，而是不同层次的效果曾被混在一起解释。**

- 因果视觉历史对冻结策略的 Revisit 收益，有跨数据集、跨 controller 和后续目标的支持。
- 把这项收益全部归功于更严格的 certificate，不符合完整消融。最新 HM3D 中 raw 总 SR
  比 strict CEC 高，差异不显著；严格授权也可能拒绝有用的历史方向。
- 学习路线没有被证明“不可能”。Pi3X 有真实的方向长尾，但早先把路线切向差当成定位误差，
  夸大了错误数量；本轮进一步检验的是当前状态更新的职责，不是再训练一个 selector。
- 现有证据支持跨时间复用历史，不支持通用空间长程导航。相机高度可消除公共尺度自由度，
不能消除相对旋转/形状误差，也不能把终点 bearing 变成绕墙路线。

因此，推荐保持最小主线：**一个因果 RGB 几何状态，一个可寻址的同源历史记录，
稠密深度与稀疏目标关系两个读出，一个冻结策略。** 不为了新颖性继续并列更多 expert。

## 2. 本轮做了什么，哪些是历史成果

### 本轮新增工作

1. 完整核对活动论文、主要实现、现有结果与当日旧审计，保留所有已有未提交改动。
2. 拉取并独立复算 HM3D 四臂授权消融全量 224 条记录；8 个缺失配对块已完成修复。
3. 本机实现 Pi3X 连续当前状态读出，只修改隔离诊断 wrapper；运行原 Pi3X、共享状态、CEC
   三臂，在 4 条已消费查询上检验是否兑现到闭环，不预设增益。
4. 修正一个 AST 路由测试漏载真实 transaction helper 的 fixture，补充两项单目事务测试。
5. 用项目已有 TeX 镜像在隔离目录编译当前稿，检查首页、复杂图表页和末页。
6. 读取 HPC 长程 oracle 任务的实际 accounting 和失败日志，发现旧“等待中”状态已过期。

### 沿用但已明确区分的历史工作

当日更早的 SR 原始复算、SPL 修复、Pi3X 固定观测归因均有独立记录。
本轮不会把这些内容全部重复命名为新发现，也没有重跑所有旧 benchmark。

- [最初完整审计](PROJECT_REAUDIT_20260906.md)：808 条已完成 arm-role CSV 记录的复算与版本边界。
- [SPL 实际位移修正](PAPER_SPL_CORRECTION_20260906.md)：669 行计量核验，其中 459 行可精确修正。
- [贡献与学习归因复审](CEC_CONTRIBUTION_AND_IMPROVEMENT_AUDIT_20260906.md)。
- [Pi3X 上下文归因](PI3X_TEMPORAL_CONTEXT_ATTRIBUTION_20260906.md)。

证据优先级：冻结运行代码与原始 trace/CSV > 独立复算 > summary > 状态文档 > 聊天概括。

## 3. 当前主架构，按真实数据流解释

```text
逐帧 causal RGB
  ├─ RGB archive + DINOv2 descriptor index
  │       └─ 按 ImageGoal 检索历史地址
  └─ frozen LingBot streaming state
          ├─ 当前 relative depth × 固定相机高度尺度 → NavDP depth input
          └─ 历史 depth + historical/current poses
                   ↓
           top-8 → SP/LightGlue → epipolar 排序 → 单候选 PnP witness
                   ↓
           经验证的历史目标关系 → 当前相对 unit bearing
                   ↓
           原 ImageGoal + 2.5 m directional PointGoal → frozen NavDP
```

### 三种不同的记忆不能混称

| 状态 | 存什么、负责什么 | 不是什么 |
|---|---|---|
| 显式 RGB archive/index | RGB、1024-D DINOv2-L CLS、时间地址 | 不是 LingBot 的检索 head；rosbag 只是可选录制载体 |
| LingBot streaming state | 因果几何上下文、深度与相机关系 | 不是带 GT pose 的全局地图，也不保证无限时长无漂移 |
| controller FIFO/随机状态 | 原策略短上下文、goal、采样状态 | 不能为 memory probe 重复写入或重新采样 |

Dense 分支在第 0–39 帧用 zero depth，之后使用
`clip[0.8,6.0](1.15 × h_camera / h_estimated_40) × D_relative`。
40 是冻结协议的估计窗口，不是网络能力突变的时刻；无效 receipt 仍是 zero depth。
单目指没有深度传感器输入，不是内部不计算深度或不使用已知相机高度。

Sparse 分支 top-8、四帧 temporal NMS；原始 DINO similarity 只提出候选。
SP/LightGlue、Fundamental-MAGSAC 对候选排序，部署时只测试最高候选的一次 PnP，
不是一直换候选直到通过。certificate 使用 PnP 有效、≥16 inliers、双侧 hull≥0.05、
RMSE≤2 px；已认证目标关系缓存，当前方向由连续状态更新。

正常拒绝：原 native 请求不变。共享 geometry stream 故障：停止执行。
这两者不是同一种失败，更不是把 Novel/Revisit 标签作为 router 输入。

NavDP 保留原 ImageGoal，并接收 `2.5 b`。ViNT 的既有接口不同：先受限转向，
再消费认证 anchor 图。因此 Table I 证明同一记忆证据可适配不同 controller，
不是它们接收完全相同的低层输入，也不是 controller 绝对排名。

旧 gatecurr 权重和 learned retrieval 的代码仍存在于大型运行类中；canonical CEC 的
候选与授权不由该训练 gate 决定。移除无关加载是 release 精简候选，不应未经实际
trace 等价检查就改正式运行环境。

## 4. 正式成果总账

### 4.1 Table I：跨 controller × dataset

| 数据集 / controller | histories / scenes | Novel native→CEC | Revisit native→CEC | Revisit +/− | exact p |
|---|---:|---:|---:|---:|---:|
| HM3D / NavDP | 28 / 21 | 6→6 /28 | 8→25 /28 | +18/−1 | 7.63e−5 |
| HM3D / ViNT | 28 / 21 | 3→3 /28 | 3→19 /28 | +16/−0 | 3.05e−5 |
| MP3D / NavDP | 42 / 25 | 17→17 /42 | 9→37 /42 | +29/−1 | 5.77e−8 |
| MP3D / ViNT | 42 / 25 | 9→9 /42 | 2→24 /42 | +22/−0 | 4.77e−7 |

原始 SR/配对复算见当日第一份审计。这是当前最应留在读者视线中的结果。
MP3D 一条弱支持 Novel 有授权但 outcome 不变；不能把 Novel SR 相同扩写成从不干预。
ViNT 查询共享 NavDP 形成的历史，不是 ViNT 自己完成全部 first-goal 的端到端评测。

### 4.2 完整单目 HM3D：不是 Table I 的同一批查询

28 histories /21 scenes，actual mono-A → causal history → mono query。

| arm | Novel /28 | Revisit /28 | 总计 /56 |
|---|---:|---:|---:|
| native | 8 | 9 | 17 |
| raw fixed | 4 | 24 | 28 |
| CEC | 8 | 24 | 32 |

CEC 对 native +16/−1，p=0.000274658；对 raw +6/−2，p=0.289。
全 mono composition 成立；高支持 Revisit 的额外 CEC-over-raw 优势没有建立。
CEC 四条 Revisit 失败均有持续 accept，runtime failure 为零，最后 stuck。
它们不是全由初始硬拒绝造成，也不能只凭 accept 判断定位已经正确。

### 4.3 Table II：累计历史支持第三目标

- 第一目标：131/196；第二目标 candidate trials：54/183；多个 B 可共享同一 A。
- 条件化到成功 A+B 且符合历史支持的 20 histories /13 scenes：
  Novel-C 4/20→4/20；Revisit-C 8/20→17/20，+10/−1，p=0.01171875。
- 修正后的 Revisit SPL：0.149→0.762。成功数没有变化。

这是连续第三目标的 conditional 效果，不是 17/20 的三段 joint；不能直接把两个
前缀成功率相乘。完整自主 5-leg 的统计确认目前仍没有完成。

### 4.4 Tables III–IV：Final14 机制归因

21 histories /10 scenes，各一条 Novel/Revisit；共享 metric-depth A 历史。

| 深度 / query 方法 | SR /42 |
|---|---:|
| metric native | 11 |
| zero native | 4 |
| mono native | 10 |
| metric CEC | 26 |
| mono CEC | 28 |

两种 CEC 的 Revisit 均20/21。不能由28>26宣称单目优于 metric；first-goal
mono27/40、metric30/40未满足预定义非劣效门。Table III 的旧 SPL 尚未精确修复，见第8节。

同一 mono 归因 population 的 raw / finite-PnP / strict 总计23/42、25/42、28/42，
Revisit 都20/21。CEC 对 raw +5/−0，p=.0625；对同 proposal 的 finite-PnP +4/−1，
p=.375。只能说明不同授权覆盖，不是已确认的 strict SR 优越性。

## 5. 本轮新增完整结果：HM3D 四臂授权消融

完整报告：[HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md](HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md)。

28 histories /21 scenes /224 条记录，原缺失8个配对块均完成。

| 方法 | Novel /28 | Revisit /28 | 总计 /56 | Novel 接管 |
|---|---:|---:|---:|---:|
| native | 7 | 8 | 15 | 0 |
| raw fixed | 9 | 26 | 35 | 28 |
| finite PnP witness | 4 | 26 | 30 | 24 |
| strict CEC | 7 | 25 | 32 | 0 |

CEC 对 native +19/−2，p=.00022125；对 raw +5/−8，p=.58105；对 witness +5/−3，p=.72656。
独立复算 `verified=true`；224/224 的 SR、距离、配对身份/种子、role 隐藏与单目来源检查一致。

raw 的 Novel 相对 native 是 **+7/−5**，不是28次全部有害。
唯一被 CEC 严格拒绝的 Revisit，在 raw/witness 下成功；其 reference hull 低于0.05，
没有代码错误的证据，也不能针对这一条放宽阈值。

所以“有证据再介入”带来行为保留，但不保证总 SR 最高。新结果必须进入成果说明，
不能因为方向不符合预期就只保留旧 Final14。

这是同一 Table-I HM3D population 的 retrospective ablation，不是 fresh confirmation；
新跑 native7/28与旧6/28不得混用。另一个 full-mono population 的 CEC也32/56，
但样本不同，不能因成功数相同当作同一条记录。

## 6. 本轮最小架构检验：连续状态与稀疏目标关系分工

### 6.1 修正错误归因后，问题更具体

原 Pi3X 正式计划中 `69/479 >90°` 比较的是 geodesic 首段，不是目标直线方位。
重新计算 endpoint bearing，实际为20/479；CEC为0/449。四条异常 query 的首次
方向误差均小，异常都出现在之后更新；anchor 未切换。

仅在稀疏 bridge 中联合重建当前图和目标图，会同时改动“我在哪里”和“目标在哪里”。
已有固定观测干预显示，主要不稳定项可落在 current-to-anchor，而不是 anchor-to-goal。
这不证明所有 learned 方法有此缺陷，但足以改变下一步的优化位置。

### 6.2 实际实现的改变

保留同一个 Pi3X、输入图、目标图和 learned proof。只将 Pi3X 当前相机状态替换为
连续 LingBot 当前状态在共同历史坐标中的表示，同时更新其 world points；然后重新
计算原 proof。没有沿用旧 acceptance 标签，没有降低阈值，没有给模型 GT。

设 `L` 是连续重建、`P` 是稀疏重建，以历史 h 与 h±8 的共同基线对齐：

```text
a = ||pP(h+8) − pP(h−8)|| / ||pL(h+8) − pL(h−8)||
Q = RP(h) RL(h)^T
RP_new(t) = Q RL(t)
pP_new(t) = pP(h) + a Q [pL(t) − pL(h)]
```

goal 仍来自当次 Pi3X 关系估计，所有 proof 在一致的新几何上计算。这是诊断对照，
不是把第三个大模型加入正式 CEC。公共 anchor/基线也会有误差，不是完美对齐定理。

### 6.3 固定观测重算：已完成

4 条已消费查询、3 scenes，159 个原观测时刻；Torch2.8 server-compatible 环境。

| 方法 | accept plans | endpoint median | P95 | >90° |
|---|---:|---:|---:|---:|
| 原 Pi3X | 143 | 12.54° | 97.47° | 18 |
| 连续当前状态 | 141 | 2.28° | 23.28° | 0 |

accept 集合不同；并非159个独立 episode。此前 Torch2.5 只替换读出的20/143与
本次重新计算 proof 的18/143不是同一口径，不追求跨 CUDA 的逐位复现。

### 6.4 真正闭环：已完成，未获得新SR增益

结果专页：[PI3X_SHARED_STATE_CLOSEDLOOP_RESULT_20260906.md](PI3X_SHARED_STATE_CLOSEDLOOP_RESULT_20260906.md)。
12条原始记录独立复算 `verified=true`；三臂均4/4成功，所有runtime failure为0。

| 查询 | 原Pi3X步数 | 连续状态步数 | CEC步数 |
|---|---:|---:|---:|
| 8W / 0000，Revisit | 175 | 170 | 168 |
| V2 / 0004，Revisit | 112 | 115 | 115 |
| PuK / 0001，Revisit | 310 | 163 | 164 |
| PuK / 0004，弱支持Novel | 491 | 243 | 306 |

连续状态对原Pi3X、对CEC均 `+0/−0，p=1.0`。不能拿旧环境Pi3X的3/4与本机4/4
跨运行计算救回。前三条使用原完整配对；第四条整块完成无损host-KV资源修复后配对，
不拼接此前OOM尝试。原Pi3X和连续状态两臂的位姿/规划哈希在资源修复前后完全一致，
但传输开销不能用作部署时延。

实际路程的明显改善在PuK两条：10.27→5.20 m、16.50→9.00 m；CEC分别为5.24、
9.26 m。**两个明显改善都来自同一个场景**，另外两条变化很小。两种Pi3X读出初始
anchor逐例相同且不切换，说明不是换候选造成的收益。原Pi3X自己的118个accepted
plans中有10个endpoint误差>90°；连续状态81个和CEC96个中均为0。后者是不同
闭环轨迹的描述统计，不是相同观测的独立样本比较。

本机采用冻结20260817的 metric-controller depth，目的是只改读出；它不是 full-mono
正式主表。四条是按已知异常选出的 consumed 开发查询，不能声称新 SR 泛化。

## 7. 长程：什么知道，什么仍不知道

### 已完成的结果

- 原48-history controlled survey，每桶16条，Revisit native→CEC：
  0–20m为2→4，20–30m为1→2，30–50m为2→0。
  原actual-mono长history构造仅2/0/0，才另冻结该survey；两者不能混称。
  长桶中跨层更多，所以该stress结果未隔离“距离”本身的因果效应。
- 23条 same-floor controlled causal survey，8 scenes，geodesic20.37–24.66m：
  native0/23，endpoint0/23，route tangent4/23，+4/−0，p=.125。
- 所有初始 CEC 均 accept。**route-tangent 臂**有14条后续geometry停止，其中13条
  相邻帧PnP状态问题、1条低inlier；其余9条中4成功、5 stuck。这里是新增
  route-motion estimator 的连续边失败，不能写成原LingBot主stream有14条漂移失败。
  同批endpoint臂则是22条stuck、1条预算耗尽。两种失败机制必须分别解释。
- 2条dense PnP开发诊断消除部分停止，但0/2成功；其中记录的route progress已100%，
  真正目标仍距6.63m，说明累计进度不等于实际到达。
- U-turn/rear-alignment，9 consumed histories /6scenes：3/9→4/9，+2/−1，p=1；
  转向臂4条geometry stop。没有“加U-turn解决漂移”的普遍证据。
- 短中程metric-distance对照：fixed25/28、metric24/28，+1/−2，p=1。
  这批metric误差MAE约.361m，不能用“尺度肯定完全错了”解释无增益。

### 本轮核对出的未完成归因

旧 `LONG_RANGE_ORACLE_ATTRIBUTION_STATUS_20260903.md` 仍有等待中的描述，但实际：

- array `16817050_32..47` 全部失败/批次取消；
- analysis `16817051` 失败，`16817052` 取消；
- 正式16-history四臂完整块数为0，没有完整summary/verifier；
- 至少两个memory-server日志明确 `OSError [Errno122] Disk quota exceeded`：
  `eval_64_oracleAttr033/.../ep_0004/1624.jpg` 与
  `eval_87_oracleAttr035/.../ep_0002/2335.jpg`；不将此原因扩展到所有失败。

远端根目录：
`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_oracle_attribution_20260903/formal_ca97c1aa4ed3f26a`。

此前单history gate各臂失败，只能作1条gate结果，不能推导16条oracle整体失败。
因此目前不能宣称已经证明长程完全是 NavDP 的能力上限；也不能认定完全是 LingBot 漂移。

### 需要分开的三个问题

1. **定位**：当前到目标的估计 bearing 与真实 endpoint bearing 是否一致？
2. **路线**：正确 endpoint bearing 是否与实际绕行的局部方向不同？
3. **执行/状态更新**：给可行局部方向后，策略执行与相邻视觉进度是否持续有效？

已知 camera height 解决的是共同尺度；不回答上述全部问题。长程也不是2.5m向量
让机器人硬走直线：NavDP仍作局部规划，但输入没有指定要绕哪一面墙或经过哪扇门。
若研究空间长程，下一步应完成已有oracle分解，不再先改半径或继续调certificate。

## 8. 论文与实现的状态

### 8.1 真正的目录与入口

| 对象 | 路径 | 当前职责 |
|---|---|---|
| 研究工作树 | `/home/asus/Research/Nav-graph-blind` | 实验、诊断、未发布开发 |
| 已发布主线 | `/home/asus/Research/Nav` | 已提交代码基线；本轮未动 |
| 活动论文 | `/home/asus/Research/Memnav_Paper/main.tex` | 本轮实际审稿、编译对象 |
| Method/Problem | 活动论文的`sec_lg/4_method.tex` / `3_problem.tex` | 主入口实际include的章节 |
| 会议计划 | 研究树`paper/会议实验优先级与执行清单_2026-08-27.md` | 计划依据，不是活动LaTeX |

标题、Abstract、Introduction SHA与本轮开始一致；论文原有12个tracked修改均保留。
本轮没有自动改稿、commit/push，也没有操作真机。

按原会议清单逐项对账：Table I四组SR、Table II shared-prefix逐leg结果、dense五臂SR、
sparse机制消融均已有；新HM3D四臂现在也已闭合。SPL的剩余修正不等于SR未跑。
低优先级长度项只有替代survey压力结果，不能说原actual-online设计全部完成。
正式真机结果仍未纳入，但按用户已有分工本轮不操作真机；五段完整自主统计也不冒充已完成。

### 8.2 两项需要实际处理的稿件问题

1. `sec/7_analysis.tex`仍以69/479描述>90°定位误差。最小修改是明确 endpoint统计20/479，
   不改变Pi3X的19/21 Revisit结果或原替换未通过决定。这是数字语义修正，需要作者确认。
2. `tables/mono_factorial.tex`仍显示旧单值SPL。210条日志缺最终落点，只能得到每个arm的
   确定数值上下界，不能当精确SPL。建议删本表SPL列、保留全部SR；不为次要指标立即
   重跑210条。Table II的459条则已有完整末位姿并已精确修正。

新增HM3D授权反结果尚未写入当前稿，建议作为Table IV外部扩展或紧凑并列段落。
无须改摘要主题，也不需要新增一页防守性表述。

### 8.3 编译与检查

使用已缓存 `ghcr.io/xu-cheng/texlive-small:latest`，复制到隔离的
`.diagnostics/overnight_project_audit_20260906/paper_build` 后编译，网络关闭。
容器缺`placeins.sty`，只在隔离目录复用本机此前已有的标准文件；未升级模板或依赖。

命令：`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`。
最终8页，Letter，PDF1.4；日志无未定义引用、未定义citation、overfull或underfull。
查看第1、4、5、6、7、8页，未见裁切/重叠。图版仍是现有TeX图，未替换为尚未确认的设计稿。
编译通过不等于上述两项数值语义已修复。

官方当前ICRA2027 CFP明确总8页（含references），double-column、double-anonymous：
[ICRA2027 CFP](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)。
本轮不是重新完整运行每条投稿合规检查，不据此给出已满足所有desk-rejection规则的承诺。

### 8.4 代码验证与实际改动

本轮唯一新增的 tracked 源码改动为
`MemNavData/test_learned_pi3x_online_integration.py`：原AST测试只抽取路由函数，
漏载已存在的`bind_navdp_monocular_transaction`，造成3条测试NameError。
修复fixture读取真实helper，并增加2条测试：receipt传到native且不修改调用者payload；
旧frame事务在模型调用前拒绝。不是新增策略门控，也不是伪造stub让错误通过。

最终针对13个相关文件运行回归：92 passed、14 warnings；JUnit产物为
`.diagnostics/overnight_project_audit_20260906/regression.xml`。
12项是既有Matplotlib/pyparsing弃用告警，2项是Torch nested-tensor优化不可用提示，
没有通过忽略真实警告来制造“干净日志”。此外逐层KV主机暂存新增单独的CPU/GPU一致性测试。
连续读出与资源驻留专项另有9项测试通过，产物为同目录`readout_and_residency_tests.xml`。

连续状态读出的两侧公共坐标变换也分别测试：改变Pi3X的Sim(3)只改变共同表示，
改变LingBot的Sim(3)后对齐的当前pose保持不变；零基线不能生成有定义的比例。
这验证的是该对齐运算的数学/实现性质，不证明真实相对位姿误差为零。

新研究实现全部在`.diagnostics/pi3x_shared_state_closedloop_20260906/`隔离：
shared-state readout、原proof预检、三臂驱动、独立复算、资源修复。
当前生产CEC、正式阈值、已封存raw文件、公开依赖源码均未因本轮重构而更改。

## 9. 作者侧人类审稿视角：只保留三个关键问题

本轮使用`icra-human-review`：同上下文、作者侧预审，并非独立人类评审或录用预测。
技能要求先只读、选择少量重要问题，因此本轮保留稿件并把语义修改单列。

**真实强项：**完整mono链、两套controller/数据集的paired记忆收益，以及第三目标的
累计历史效果，已经能支撑一篇聚焦的系统研究；不因training-free自动降低贡献价值。

1. **贡献归因（Method、Table IV）**：raw也有几何记忆，完整HM3D不支持strict总SR最好。
   若把主贡献说成新门控或验证器，就与证据不符。最小处理是保留Abstract的streaming-memory
   主线、披露新的四臂结果，并清楚说哪个输入被复用；不必发明learned MoE。
2. **结果解释的准确性（Analysis、Table III）**：endpoint/route混用和旧SPL影响可信度。
   最小处理是改一处统计定义、去掉不可精确恢复的次要列；不改变SR、不扩写长篇免责声明。
3. **适用范围（Table II、Limitations）**：时间上多目标记忆不等于空间上长程路线能力，
   更不等于完成5-leg lifelong。当前论文已有主要限制说明，保持短中程episodic定位即可；
   若要扩大claim，必须有相应闭环证据，不能靠方程复杂度或重命名。

最近工作的区别应精确而非夸大。公开
[AnyImageNav, Sec.3.5/5](https://arxiv.org/html/2604.05351v3)也以图像为几何查询，
采用短/长窗口、几何验证和目标位姿细化；其方法依赖记录位姿作Sim(3)对齐，讨论中明确
使用深度/odometry。我们的区别是因果单目episode-local状态与冻结策略方向读出，
而不是“首次提出检索后几何验证”。两者任务/传感器/停止条件不同，不直接横减published SR。

## 10. 最小而有依据的下一步

### P0：把已知事实收口

- 作者确认后修正Analysis的endpoint统计，处理Table III SPL；纳入完整HM3D四臂。
- 主表SR保持不动；标题、摘要、Introduction无需重新定调。
- 更新版本/结果入口，保留原summary和raw traces，不以当前代码覆盖冻结实验源码。

### P1：本轮去留决定——保留机制原型，不替换CEC

本轮只降低部分绕行，没有SR增益。保留隔离诊断原型，不替换CEC，也不立即长训。
真正值得泛化的设计原则是“连续状态负责当前，稀疏查询负责目标关系”；
下一次学习实验应该替换关系读出并保持这条分工，而不是重训同一个selector或再加gate。
这也不等于永久缓存第一次目标估计：此前四例固定观测中，弱支持Novel的动态目标关系
更准确；当前闭环保留动态目标。应减少不必要的当前状态重建，而非预设所有目标关系
都无需更新。
新的开发查询不得伪装成正式确认；原Pi3X未通过决定不因开发修复倒签通过。

### P2：若继续解决空间长程

本轮补查`myquota`：scratch容量1.37TB/5TB（27.39%），文件数4,897,908/5,000,000
（站点显示97%）。`df`全盘仍有603TB，不能据此认为用户配额宽裕；`quota -s`在本站
返回权限错误，应使用`myquota`。这不能还原9月3日精确占用，但确认当前限制主要在文件数。
快照见`.diagnostics/overnight_project_audit_20260906/hpc_capacity_snapshot.md`。

先处理用户文件数容量并复用已冻结oracle四臂设计；修复最小完整配对块、验证
实际import与产物容量，再补缺失块。不要把基础设施失败当成controller能力失败。
只有定位/路线/执行分解完成后，才值得实现新的视觉重锚定或路线进度模型。

一个帮助归因、但不代替实测的线性化表达是：令当前平面目标向量为`v`，其误差为`δv`，
则方向误差近似`δθ ≈ v_perp^T δv / ||v||²`。共同正尺度不会改变方向，但目标/当前
相对平移误差和当前朝向误差会进入`δv`；靠近目标时同样的位置误差可放大成角度误差。
这解释了为什么“已知高度”不充分，却不能单独解释所有长距离失败：长程的路线拓扑、
状态累计误差和controller能否执行局部绕行仍需各自的对照。

### 暂不做

不重复加宽候选、训练旧共享decoder、直接整段历史无检索、角度扫描、阈值sweep、
按失败样本定制graph rescue；不追加GOAT/更多controller/真机工作来掩盖当前问题。
已有负结果不证明这些方向永远无效，但需要新的、可区分旧结果的假设才能重启。

## 11. 操作与复现边界

HPC按`HPC_SHARED_SSH_OPERATIONS_20260816.md`使用现存alantorch/yz11502共享连接。
本轮只读远端；没有新增Slurm提交/取消。authority修复任务已完成，读取时队列为空。
临时localhost文件通道、转发、审计PTY已关闭，共享master保留。

本机GPU实验只使用专有18891/8891服务器。其他工作区进程即使抢占显存也不停止；
基础设施OOM日志保留，不算科学失败、不与完整三臂混合。
第四条最终采用`closedloop_resource_repair3`完整三臂，14:22完成独立验证。
本任务18891/8891服务器及调度/复算进程均已退出，其他工作区进程保留；不再留后台eval。

## 12. 最终交付与仍需作者决定的事项

- 完整总账：本文件。
- 新HM3D消融：[结果专页](HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md)。
- 最小读出重构：[闭环结果、两张实际轨迹和复现入口](PI3X_SHARED_STATE_CLOSEDLOOP_RESULT_20260906.md)。
- 最小稿件修正：[待确认的before/after](PAPER_MINIMAL_CORRECTIONS_20260906.md)。

本轮目标已经完成的是审计、完整开发对照、独立核验和基于结果的去留决定。
没有声称已经找到新的正式SR提升。下一步先纠正论文两处具体事实并纳入HM3D完整消融；
若继续研究学习或空间长程，各自依据第10节的不同未决问题推进，不混成新的“大一统”模块。
