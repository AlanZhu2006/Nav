# GLP 文献综述：机制根源、最近邻与可声称边界

日期：2026-08-08

方法：三支并行检索（贝叶斯搜索理论/POMDP、frontier 与拓扑记忆导航、扩散策略与
选择性接管），全部引文经 2026-08 网络核实（arXiv/PMLR/CVF/项目页）。标注
"venue 未核实"者在引用前需再确认。

对应设计稿：`MemNavData/GOAL_POSTERIOR_DECISION_LAYER_20260807.md`。

---

## 0. 一句话结论

GLP 的**每一个机制都有扎实文献根**（这是好事：优雅需要根），但截至 2026-08，
**没有任何已核实工作把它们组合成"在 {记忆图节点 ∪ frontier ∪ ω₀} 上维护单一
归一化后验、以后验形态取代 Novel/Revisit 分类器、为 image goal、驱动冻结扩散
策略"**。三篇必须正面处理（BEINGS、AnyGoal、GOAT），两篇提交前必须精读全文
（FTM、OVAL）。

---

## 1. 机制 → 文献根对照

| GLP 机制 | 文献根 | 引用姿态 |
|---|---|---|
| Carving（未见即减质量，`∝(1−d)`） | **Koopman 1946/1956**《Search and Screening》；**Stone 1975**《Theory of Optimal Search》；USS Scorpion（Richardson & Stone, NRLQ 1971）；**AF447**（Stone et al., Statistical Science 2014, arXiv:1405.4720）——`p′ = p(1−q)/(1−pq)` 正是经典未搜到更新 | **作为 lineage 引用，绝不声称为贡献**。80 年历史、捞过沉船的数学 |
| 检测功效上限/下限（floor） | AF447 给信标失效设 P=0.25（正因 2009 年没设够，"排除区"里漏了残骸两年）；Koopman 指数检测律天然 q<1 | 同上；我们的显式 floor 只是参数化差异 |
| ω₀（假设空间外） | SAR 实践的 probability-of-containment<1（Frost & Stone 2001） | lineage |
| voxel 版同款滤波 | **3D-MOS**（Zheng et al., IROS 2021, arXiv:2005.02878）、**GenMOS**（ICRA 2023）：octree belief + miss-likelihood 更新 | 最近的机器人版本；差异：固定度量区域、手工检测模型、category goal |
| 负证据 | Hoffmann et al., IROS 2005（定位中的 negative information，同样为检测不可靠而做软更新）；Koch, Information Fusion 2007（跟踪） | lineage |
| "定位目标"= Markov localization 对偶 | Fox/Burgard/Thrun, JAIR 1999；主动定位熵最小化（Burgard 1997） | **对偶框架无人显式声称——可作为叙事贡献**（不是新数学） |
| 目标在未观测空间的概率 | Aydemir et al., T-RO 2013；**L2M**（Georgakis, ICLR 2022, arXiv:2106.15648，逐 cell μ+σ）；**PEANUT**（ICCV 2023，dense 目标概率图）；OVAMOS 2025 | 必引先例；均为 category goal / dense grid |
| graph∪frontier 假设空间结构 | **NTS** ghost nodes（Chaplot, CVPR 2020）；**NRNS** frontier 节点距离预测（NeurIPS 2021, arXiv:2110.09470）；**FTM**（ECCV 2024，节点∪ghost frontier 供 image goal） | 结构已存在但均非概率化/非归一化 |
| 熵门 abstention | 主动定位熵最小化的被动镜像；**KnowNo**（CoRL 2023）conformal abstention | "熵作为对默认策略的信任门"这个用法未见先例 |
| 走近式验证 | **IEVE**（CVPR 2024）的 verification-by-approach；SLING 的 last-mile 切换 | 我们的连续证据累积应 subsume 其离散版本 |
| bearing 手递给固定局部策略 | **SLING**（CoRL 2022, arXiv:2211.11746，检测到目标后切换到 keypoint 相对位姿模块，ImageNav 45→55%）；NRNS 的 (distance, direction) 头；PixNav 的 goal pixel | 最近邻；均为启发式开关、无校准无 abstention |
| 经验使导航变好（SR 随目标序号上升） | **GOAT**（RSS 2024, arXiv:2311.06430）**Fig.6：真机 60%→90% SR / 0.20→0.80 SPL，无记忆消融无提升**；GOAT-Bench（CVPR 2024）分序号曲线；OneMap（ICRA 2025）SPL 随序号单调升；RECON 复访提速 20–85% | **曲线不是我们首发**——引用并差异化 |
| 扩散策略 mode collapse | **SIME**（2025, arXiv:2505.01396）：操作臂上重采样 91% 全成或全败；自动驾驶 DIVER 等；**Mazza et al. 2026**（arXiv:2605.22493）理论解释（Lipschitz 限制迫使塌缩） | **现象不能声称为新发现**——写成"在 image-goal 导航域确认并给出新诊断" |

---

## 2. 五大威胁论文与差异化

### 2.1 BEINGS（arXiv:2409.10216；ICRA 2025 待确认）

Bayesian image-goal 导航：离散 cell 上显式目标位置后验（未探索 cell 均匀初始
化持有质量）、贝叶斯搜索理论更新、VLAD 图像差异度似然对 3DGS 渲染视图、MPC
控制。**单篇最接近"image-goal 的贝叶斯位置后验含未观测空间"**。

差异化：dense 度量格 vs 结构化 {记忆节点∪frontier} 假设集；手工 VLAD 似然 vs
校准学习似然（patch/几何）；单 goal episode、无 revisit 语义、无多 goal 持久；
MPC vs 冻结扩散策略。引用前核实 venue。

### 2.2 AnyGoal（arXiv:2606.13878，2026-06 并发预印本）

逐像素 (μ,σ²) 贝叶斯价值图（VLM 分数融合），跨 2,669 个 lifelong 子任务从不
重置；frontier 按 VLM softmax + UCB 排序。**同时命中"贝叶斯 frontier 后验"和
"lifelong 持久"**。

差异化：object/text goal（image-goal 支持未核实）；度量图非拓扑图；**没有
"目标在我已见过的地方"这个假设**（无 revisit 通道）；training-free 启发式似
然。作为 concurrent work 引用。

### 2.3 GOAT（RSS 2024）+ GOAT-Bench（CVPR 2024）

系统骨架最接近：跨序贯 goal 的持久实例记忆；query-memory-first-else-frontier；
且已发表 lifelong 上升曲线。其记忆门是**硬阈值**（image goal 用 SuperGlue
匹配分 >6.0；低于阈值的早先目击不可恢复；记忆位置与 frontier 从不在同一分数
里竞争）；深度+位姿+MaskRCNN 度量语义图；FMM 点导航。

差异化：归一化后验中节点与 frontier **竞争**（软证据可恢复）vs 硬门；RGB-only
学习几何 vs 传感器度量图；冻结扩散策略 vs FMM。GOAT-Bench 缺
"episode 内实例是否已被见过"的拆分分析——**这个分析槽位是空的，可以占**。

### 2.4 FTM（ECCV 2024，Frontier-Enhanced Topological Memory）

表征双胞胎：image-goal 导航的 {已探索节点 ∪ ghost frontier 节点（在线隐式表征
预测其特征）} 记忆图。但端到端策略消费该图——无显式后验、无 revisit/novel
语义、深度 frontier、单 goal。**目前仅 poster 级核实，提交前必读全文**。

### 2.5 SLING（CoRL 2022）

"发现目标→切换到 bearing/相对位姿手递"的最近邻，插在五种探索策略下 ImageNav
45%→55%。差异化：可见性启发式开关 vs 校准后验；接管控制器 vs 喂给冻结策略；
无 abstention 保证；无扩散基座；无 mode-collapse 诊断。

### 次级必处理

- **NRNS**（NeurIPS 2021）：机制祖先——frontier 节点按预测到 goal image 的
  距离贪心排序 + 独立的 goal-in-sight 检测器（β>0.5）。我们加：已探索节点进
  同一假设空间、时间贝叶斯累积、后验集中度替代检测器阈值。
- **UniGoal**（CVPR 2025）：zero/partial/perfect 三态匹配阶梯 = 后验形态的
  3 级离散化。**IEVE**（CVPR 2024）：分级置信驱动行为的手工表亲。
- **MemoNav**（CVPR 2024）：拥有"多 goal 序贯 ImageNav + 跨 goal 持久拓扑
  记忆"协议——评测协议引用。
- **RNR-Map**（CVPR 2023）：概率化 goal 定位热图但**仅限已见空间**——正好是
  我们统一掉的分裂。
- **Mod-IIN**（ICCV 2023）：goal-blind frontier 探索 + 二值重识别是 IIN 的
  常态——最干净的反衬。
- **NoMaD**（ICRA 2024）：探索/goal-reaching 统一在策略层，但模式开关是外部
  手设的二值 attention mask——我们统一在信念层。

---

## 3. 可安全声称清单

1. **在 {记忆图节点 ∪ frontier ∪ ω₀} 上的单一归一化后验，Novel/Revisit 是
   后验形态而非分类器/模式开关，作用于 image goal**——每个近邻都是硬阈值
   （GOAT/Mod-IIN）、离散阶梯（IEVE/UniGoal/NTS 双头）、单侧概率
   （RNR-Map/IGL-Nav 只管已见；NRNS/PONI/VLFM 只管 frontier；
   L2M/PEANUT/BEINGS dense grid 无记忆结构）或不透明端到端（FTM/MemoNav）。
2. **endpoint-heading 圆统计 R 诊断** + 困难状态条件化表征（image-goal 扩散
   策略）+ **正确初始 bearing 救活同一冻结策略的配对证明** + **bearing 容差
   曲线**（±30° 足够/45° 临界/60° 崩溃）——四者均未被占。
3. **校准的选择性干预 + 对原生策略的逐位 abstention + non-inferiority 统计**
   ——组合无先例（runtime assurance 文献全部为 safety 干预且 fallback 非基座
   自身；KnowNo fallback 是人）。
4. "goal 搜索 = Markov localization 的对偶"叙事框架。
5. GOAT-Bench 缺失的"episode 内实例已见/未见"拆分分析。
6. 后验层 + 冻结端到端扩散局部策略的解耦（该谱系无先例）。

## 4. 必须对冲清单

- 不声称"发现扩散策略 mode collapse"（SIME/DIVER/Mazza 已报道）——写"在
  image-goal 导航域确认，并给出圆统计诊断与 bearing 解药"；
- 不声称 carving/未搜到更新为新数学（Koopman/Stone/3D-MOS）；
- 不声称"首个 SR 随经验上升的曲线"（GOAT Fig.6/GOAT-Bench/OneMap/RECON）——
  声称"首个由冻结端到端扩散策略 + 小校准层涌现的该曲线"；
- BEINGS/AnyGoal 作为最近先例/并发工作正面引用。

## 5. 评测采纳建议

- 采用 GOAT-Bench 的 image-goal 子集作为公开对标（5–10 序贯子任务）；
- 报 per-goal-index SR/SPL 曲线（对齐 GOAT Fig.6 的展示方式）；
- 新增 GOAT-Bench 没有的拆分：子任务按"目标实例此前是否进入过视野"分层——
  正是后验 revisit/novel 语义的直接检验；
- SLING 式对照：同一后验层插在不同探索策略/基座下的可移植性。

## 6. 提交前行动项

1. 精读全文：FTM（ECCV 2024）、OVAL（arXiv:2604.12872）；
2. 核实 venue：BEINGS（ICRA 2025?）、OVRL 系、SplatSearch、Monaci 2025、
   LagMemo/T2Nav/IPPON；
3. 跟踪 AnyGoal 是否中稿（并发→已发表会改变引用姿态）；
4. GOAT-Bench 复现管线搭建（image-goal 子集）。

## 7. 关键引文速查

```text
Koopman 1946/1956 Search and Screening; OR 4:324,4:503,5:613
Stone 1975 Theory of Optimal Search; Richardson&Stone 1971 (Scorpion)
Stone et al. 2014 AF447, Statistical Science, arXiv:1405.4720
Fox/Burgard/Thrun 1999 JAIR Markov Localization, arXiv:1106.0222
Hoffmann et al. 2005 IROS negative information; Koch 2007 Inf. Fusion
Aydemir et al. 2013 T-RO; Wandzel 2019 ICRA OO-POMDP
Zheng et al. 3D-MOS IROS'21 arXiv:2005.02878; GenMOS ICRA'23 arXiv:2303.03178
Yamauchi 1997; SemExp NeurIPS'20 arXiv:2007.00643
PONI CVPR'22 arXiv:2201.10029; L2M ICLR'22 arXiv:2106.15648
PEANUT ICCV'23 arXiv:2212.02497; VLFM ICRA'24 arXiv:2312.03275
SPTM ICLR'18 arXiv:1803.00653; NTS CVPR'20 arXiv:2005.12256
NRNS NeurIPS'21 arXiv:2110.09470; RECON CoRL'21 arXiv:2104.05859
ViKiNG RSS'22 arXiv:2202.11271; RNR-Map CVPR'23 arXiv:2303.00304
FTM ECCV'24 (Springer 978-3-031-72897-6_17)
MemoNav CVPR'24 arXiv:2402.19161; Mod-IIN ICCV'23 arXiv:2304.01192
IEVE CVPR'24 arXiv:2402.17587; UniGoal CVPR'25 arXiv:2503.10630
MultiON NeurIPS'20 arXiv:2012.03912; Marza IROS'22 arXiv:2107.06011
GOAT RSS'24 arXiv:2311.06430; GOAT-Bench CVPR'24 arXiv:2404.06609
OneMap ICRA'25 arXiv:2409.11764; BEINGS arXiv:2409.10216
AnyGoal arXiv:2606.13878 (concurrent); OVAL arXiv:2604.12872
Diffusion Policy RSS'23; NoMaD ICRA'24 arXiv:2310.07896
ViNT CoRL'23 arXiv:2306.14846; NavDP arXiv:2505.08712
SIME arXiv:2505.01396; Mazza et al. arXiv:2605.22493
SLING CoRL'22 arXiv:2211.11746; PixNav ICRA'24 arXiv:2309.10309
KnowNo CoRL'23 arXiv:2307.01928; Sim-to-Lab-to-Real AIJ'23 arXiv:2201.08355
Recovery RL RA-L'21; Silver 2018 arXiv:1812.06298 (residual policy 对照)
Igbinedion & Karaman ICRA'24 arXiv:2305.16502
```
