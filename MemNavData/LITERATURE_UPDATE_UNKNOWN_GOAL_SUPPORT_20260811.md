# 最新文献整理：Unknown-goal Memory Support

日期：2026-08-11（CST）  
范围：只整理会直接改变当前 Revisit/Novel 未知阶段架构与下一实验的工作。检索与核对以
论文主页、arXiv/CVF 等一手来源为准；2026 年工作均按 concurrent preprint 对待。

## 0. 结论先行

最近文献没有支持“再训练一个 Novel/Revisit 分类器”这条路。更一致的趋势是：

1. 将长程目标推理与局部控制分离；
2. 先用廉价的语义/检索信号缩小范围，再用几何或多视角进行自验证；
3. 把匹配可靠性建模为 query-reference 或候选集合的不确定性，而不是一个跨场景固定阈值；
4. 利用自然运动产生的连续观测累积证据，而不是部署时先原地扫描八个方向；
5. 不显式接收 Novel/Revisit 标签，由统一 belief 决定 memory 是否有支持。

这与本项目刚完成的 OOF 结果一致：相对 anchor 排序可以学，但单时刻 existence 仍因
跨场景可观测性与校准不足而过度 abstain。

## 1. 最相关的新增/近邻工作

| 工作 | 核心机制 | 对本项目的直接含义 | 新颖性边界 |
|---|---|---|---|
| [What does really matter in image goal navigation?](https://arxiv.org/abs/2507.01667)（2025，2026-02 v3） | 将 ImageNav 分成核心导航能力与由 observation-goal 比较产生的方向信息，并指出模拟器设置可能形成 shortcut | 支持把 NavDP 当冻结局部执行器、把 goal-conditioned 支持/方向作为外层变量；也要求避免把 oracle bearing 当部署方法 | “方向重要”本身不能作为新颖性，只能以严格闭环能力分解和未知阶段 memory 机制差异化 |
| [IGL-Nav](https://arxiv.org/abs/2508.00823)（2025） | 单目前馈增量 3DGS；先离散粗定位，接近后再用可微渲染细化目标位姿 | 最强近邻之一，说明 coarse-to-fine 几何定位是有效主干；若我们继续做完整 3D 重建会直接进入其赛道 | 我们不应声称“记忆中粗检索再几何细化”本身新；差异必须来自 episodic revisit、未知阶段选择性接管与冻结 diffusion policy |
| [AnyImageNav](https://arxiv.org/abs/2604.05351)（2026 concurrent） | training-free semantic relevance gate → 3D multi-view geometry → self-certifying registration loop；报告 Gibson 93.1%、HM3D 82.6% | 直接说明“语义候选 + 几何验证”的两级结构已被占；同时支持验证应是循环、多视角、可自证，而非一次 RANSAC 硬判 | 我们不能把简单的 semantic-to-geometry cascade 当贡献；可研究其没有覆盖的跨 goal episodic memory 与 residual action utility |
| [KappaPlace](https://arxiv.org/abs/2605.19435)（2026 concurrent） | 用 vMF concentration 建模 VPR 表征不确定性，并给出 query-reference match-level 可靠性；支持 frozen backbone 的 post-training extension | 当前 learned router 应预测“这个 goal-anchor 匹配有多可靠”，而不是预测 episode phase；match-level uncertainty 是最直接可借鉴模块 | “学习匹配置信度”已不新；新颖性只能来自它如何进入未知阶段 memory belief 与闭环 action advantage |
| [Through the Lens of Doubt](https://arxiv.org/abs/2510.13464)（2025） | 无训练地从候选相似度分布构造 distinctiveness、ratio spread 和组合 uncertainty | 证明 top-K 不一定用于扩大候选动作，可以只用于估计集合歧义；这不与本项目 top-K 闭环 null 冲突 | 先把这些廉价集合统计作为强基线，不能只和未经校准的 max-DINO 比 |
| [On the Estimation of Image-matching Uncertainty in VPR](https://arxiv.org/abs/2404.00546)（CVPR 2024） | 比较 retrieval、aleatoric uncertainty、geometry；SUE 利用 reference pose，且与几何验证互补 | 支持“学习证据 + 几何证据互补”，不支持把 RANSAC 蒸馏掉；geometry 应是 expert/likelihood，而非负标签生成器 | 必须报告简单 descriptor-distance/SUE 级基线，并保持几何通道 |
| [Improving VPR Navigation by Verifying Localization Estimates](https://arxiv.org/abs/2407.08162)（RA-L 2024） | integrity monitor；除单 query 拒绝外，还利用 recent-query history + odometry 外推已验证位置 | 给出最直接的时序先例：历史中最近一次可信匹配可以跨时刻维持，而不是每个 planning step 独立重判 | 单纯“时序验证”不新；我们需要证明它对 image-goal episodic memory 和冻结 NavDP 的闭环因果增益 |
| [Think before Go / HRNav](https://arxiv.org/abs/2604.17407)（ACL 2026） | 慢速高层短程规划 + 快速低层执行，并显式抑制 wandering | 支持双时标设计；当前 memory expert 应低频提供短承诺，NavDP 保持高频局部控制 | 层级 planner/executor 不新；我们不应再包装成通用层级导航 |
| [MemoNav](https://arxiv.org/abs/2402.19161)（CVPR 2024） | STM/LTM/working memory，过滤与目标无关的历史节点；在 multi-goal ImageNav 中评估 | 是“目标相关记忆选择”的直接近邻，必须正面引用 | 不能声称首个 goal-relevant memory；差异是 causal episodic support、未知阶段 abstention 与 frozen diffusion residual |
| [AnyGoal](https://arxiv.org/abs/2606.13878)（2026 concurrent） | lifelong 不重置的 Bayesian value map，按均值/方差与 UCB 排 frontier；没有外部 Novel/Revisit phase | 支持统一 belief 取代 phase router，也说明“持久 Bayesian map + frontier”已是并发方向 | 本项目不能再把单一 goal posterior 本身当核心新颖性；应聚焦图像实例 revisit 与动作效用 |

## 2. 文献与本项目数据共同排除的路线

### 2.1 再做一个单帧二分类器

旧 Phase-B 和本次 train-only nested OOF 都显示：candidate ranking 的相对信息可迁移，
绝对 existence operating point 不稳定。本轮已经使用集合特征、连续几何、风险匹配阈值，
仍在真正 Revisit 状态上因 abstain 丢失覆盖。继续换 MLP、加层数或调 `C` 不针对失败原因。

### 2.2 完全蒸馏/删除 RANSAC

VPR uncertainty 文献明确把 learned uncertainty 与 geometry 视为互补。本项目的 N=40
闭环显著结果也来自 geometry memory。合理方向是把 RANSAC 作为高精度但不完备的 expert：
pass 提供强正似然，reject/insufficient 只表示“尚未证实”，不能直接当 Novel 标签。

### 2.3 为获取方向而原地八向转圈

部署时的强制 360° glance 既昂贵，也已在 Active-glance V1--V3 中造成净损失。更合理的
观测来源是 NavDP 自然运动中的连续视角；只有在已检测到高价值但高不确定 memory hypothesis
时，才允许有限、可停止的主动验证。

### 2.4 直接复刻 3DGS/语义-frontier 大系统

IGL-Nav 与 AnyImageNav 已经把 coarse-to-fine 3D localization 做得很强；AnyGoal、HRNav
也占据了 Bayesian frontier 与层级规划。完整复刻会大幅增加系统复杂度，却稀释当前唯一
显著的可部署结果——episodic geometry memory 对冻结 NavDP 的闭环增益。

## 3. 对当前架构的收敛建议

部署时不提供 phase，系统只有一个统一接口：

```text
goal image + causal episodic memory + natural observation stream
                         |
             Memory-Support Belief
        (DINO set uncertainty + match uncertainty
         + geometry likelihood + temporal persistence)
                         |
        high-confidence anchor proposal / abstain
                         |
        residual action-advantage safety gate
                         |
              frozen NavDP / X-NavDP executor
```

- **Geometry expert**：提供高精度 anchor 与相对位姿证据；不是 phase classifier。
- **Learned support expert**：估计 `P(memory contains a useful support | evidence)`；不直接输出
  Novel/Revisit。
- **Action expert**：只有在 counterfactual rollout 表明 memory proposal 比 native 更优时才
  短时接管；否则严格回到 base policy。
- **时间**：以自然 planning stream 累积证据并做 hysteresis，避免单帧尺度漂移。

## 4. 当前可能成立、但尚未被结果授权的贡献

最有希望的论文命题不是“首个记忆导航”或“首个几何验证”，而是：

> 在不知道目标是否已见过的条件下，将 episodic image memory 表述为带不确定性的可选择
> residual support；只有当该 support 对冻结 diffusion navigator 的动作具有正优势时才接管。

它与近邻的实质差异是同时包含：unknown-goal、跨 goal causal memory、match uncertainty、
base-policy-preserving residual 和同机同进程 paired closed-loop。当前只验证了 geometry
memory 的闭环价值；learned support 与 action advantage 尚未通过，因此现在只能写成研究
假设，不能写成已完成贡献。

## 5. 下一实验由文献直接给出的约束

下一轮应是 **natural-stream temporal support shadow**，而非闭环长跑：

1. 只采集 train scenes 的正常 NavDP planning stream，不主动旋转、不读取 phase；
2. 对稳定候选跟踪 DINO score distribution、match-level uncertainty、RANSAC 连续量和
   pose/bearing agreement；
3. 用 scene-disjoint nested OOF 选择风险点；
4. 与当前 hard geometry 和本轮 single-state factorized head 同时比较；
5. 只有在三种子均实现更高 correct-anchor coverage，且 strict false activation、wrong
   anchor 均不恶化时，才进入 action expert/闭环。

这一步若仍失败，应保留 geometry router 作为最终方法，不再为“学习化”牺牲可靠性。
