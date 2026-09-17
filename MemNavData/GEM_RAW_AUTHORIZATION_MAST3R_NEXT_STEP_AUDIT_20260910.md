# GEM 下一步：固定 raw 路径的匹配验证与 MASt3R-SLAM 对照

日期：2026-09-10，北京时间。状态：**证据审计与实验设计；尚未运行新模型或导航。**

本轮范围是近期共视结果、既往 selector/learned-proof 实验和 MASt3R 可承担的角色，
不是重新审计所有真机、论文排版或 HPC 作业。本轮不修改现有方法、阈值、模型或论文；
不安装依赖、不提交作业、不 commit/push。工作区原有修改全部保留。

## 1. 决策摘要

下一步优先回答一个问题：

> **保留当前 raw 的历史候选、LingBot 目标定位及 NavDP 执行，仅增加一种新的匹配证据，
> 能否保留低共视下有用的方向，并识别不可靠的历史干预？**

主检验先使用冻结 MASt3R 的双视图匹配证据，而不是立刻移植全部 MASt3R-SLAM。
这不是否定完整 SLAM：匹配验证和统一定位后端回答不同问题，必须分开比较。
若 raw 本身的位姿/时序更新错误是主要损失，或要检验现成 SLAM 能否替代自建后端，
再运行完整 MASt3R-SLAM 的独立定位对照。

不能把下一步称为“终于补上一个从未做过的判别器”。DINO 阈值、存在性与排序分离、
多视角稳定性、learned proof 都已经做过；新的信息来源和受控比较对象才是本轮区别。

## 2. 最新结果与可支持的结论

### 2.1 修复版共视谱

28 histories、21 scenes；131 个支持查询与 28 个原 Natural Novel 对照。
支持查询通过改变目标 yaw 构造五档，历史、起点和目标位置在同一 history 内保持固定。

| 共视区间 | 查询数 | Native | Raw fixed | GEM |
|---|---:|---:|---:|---:|
| [0.1,0.3) | 23 | 7 | 22 | 13 |
| [0.3,0.5) | 27 | 11 | 26 | 20 |
| [0.5,0.7) | 28 | 7 | 28 | 25 |
| [0.7,0.9) | 28 | 5 | 28 | 25 |
| [0.9,1.0] | 25 | 8 | 23 | 22 |
| 支持查询描述性合计 | 131 | 38 | 127 | 105 |
| 原 Natural Novel | 28 | 6 | 5 | 6 |

131 不是独立 history 数，支持合计也不是自然部署的 Novel/Revisit 配比。
最低共视档 GEM 相对 raw 是 +1/-10，原始 p=0.01171875，五档 Holm p=0.05859375；
不能把该比较写成通过预注册多重比较门。其他正式统计保留原报告口径。

原 Novel 中有一条重标注共视为 0.1166767；保留原身份，不能声称 28 条全是新标注下的严格零支持。
这种情况也说明 benchmark role 和几何可定位性不应混成一个训练标签。

### 2.2 修复版 actual-mono A 后的查询

45 histories、23 scenes，每条一 Novel、一 Revisit；这些源场景此前已经使用。

| 查询 | Native | Raw fixed | GEM |
|---|---:|---:|---:|
| Novel | 15/45 | 9/45 | 15/45 |
| Revisit | 18/45 | 45/45 | 45/45 |
| 等比例查询合计 | 33/90 | 54/90 | 60/90 |

GEM 对 raw 总体 +12/-6，p=0.2378845；对 native +27/-0。
GEM 的 Novel 全部 exact native，Revisit 与 raw 都饱和。此总体的 Revisit 初始测地距离
2.014–3.865 m，最大历史共视约 0.717–0.756，不能代表低共视和长程问题已解决。

本轮重新读取本地派生 JSON，加总复核了上述人数与支持 SR；没有重新读取全部远端归档。
原完整核验、逐任务收据和 SHA 来源见 [终局报告](REPAIRED_HM3D_FINAL_RESULTS_20260910.md)。

### 2.3 拒绝归因真正查到了什么

前一轮读取了全部 24 条 raw 成功/GEM 失败的支持查询，以及全部 28 条原 Novel：

- 23/24 从未接管；1/24 已通过证书但执行失败。
- 8 条在 Fundamental 内点或覆盖预检查就停止，未计算 PnP。
- 15 条已计算 PnP 但拒绝；全部至少一侧 coverage <5%。
- 其中 8 条满足 PnP 状态、内点数和 RMSE，只卡覆盖。
- 3 条当前 top-8 内已有其他候选通过原 Fundamental 预检查，但其 PnP 尚未求解。

这只定位了程序分支，并未证明 23 个拒绝都是错误拒绝。
**Raw 与 GEM 的目标估计不同，不能由 raw 成功推断被拒 PnP 的 bearing 正确。**

另有一个必须明确的度量约定：Fundamental coverage 用原图面积，PnP coverage 用 LingBot
padded raster 面积。实际 294×518 有效区域填充到 518×518 时，padded 5% 相当于有效区域
约 8.81%。这是冻结规则的真实行为；改变分母是方法改动，不能静默修正旧结果。

把最终内点下限由 16 降到 8、其余不变，已有被拒位姿新增通过 0 条。
把最终 coverage 改成 2% 会让选定损失中的 4 个已有位姿通过，但该反事实不含所有查询，
也没有补算预检查后新进入的 PnP，不能称“救回 4 条 SR 且无 Novel 风险”。

详见 [52-query 拒绝审计](HM3D_COVISIBILITY_REJECTION_AUDIT_20260910.md)。

## 3. 以前已经做过什么：避免重复

| 已做实验 | 已有结果 | 对下一步的约束 |
|---|---|---|
| 公平 DINO 阈值 | 旧 session 判定 87.3%；固定 0.5 得到 21.7% 是不公平基线 | 不能再次把未校准 DINO 当弱基线 |
| DINO-only GLP/set model | GLP 打平 max-DINO；一个 set model 的 top-1 更差 | 新名字和更大 MLP 不提供新证据 |
| Phase-B/OOF 校准 | session 72.7%，OOF 后 84.5%，仍低于同协议 DINO 87.3% | 排序 AUC 和跨场景授权不是一个指标 |
| F2/F8 因子化 selector | 已分离存在性、候选排序，做 scene-OOF；未超过原基线 | “把排序和判别分开”本身不是新实验 |
| MRC 多视角稳定性 | 相邻重放高度相关，内部稳定性受 scene 影响很大 | 不能以反复相同匹配/重放当独立证据 |
| DINO-order + 原证书 | 28 个已消费 Revisit，25/28 对 25/28，+0/-0 | 仅改候选顺序已测；不是普遍等效结论 |
| Pi3X spatial learned proof | 同方向指标离线正确正例接受 119 对 107；闭环 Revisit 19/21 对 CEC 20/21，未获替代资格 | 不是“从没学出过信号”；也不能再次只报离线 AUC |
| 固定 top-8 多参考 PnP | 原单帧 14/41、联合 12/41 达到 0.5 m 位置标准 | 朴素合并对应未显示收益；不是所有多视图方法都无效 |
| 同 PnP、不同授权强度 | 旧 HM3D 四臂里 finite-PnP 30/56、strict CEC 32/56，raw 35/56 | 放松证书已经有对照；不能自动期待更高 SR |

这些历史实验的总体、标签、执行版本不同，只用于去重和机制约束，不与本次 SR 横减。
旧文档中的阶段性 fresh160 人数不覆盖最终 112/120 等已封存结果。

关键来源：

- [Selector 去重审计](SELECTOR_DEEP_AUDIT_AND_NEXT_STEP_20260812.md)
- [F2 原始协议](UNKNOWN_GOAL_SUPPORT_OOF_PROTOCOL_20260811.md)
- [MRC 信号归因](MRC_SIGNAL_ATTRIBUTION_AND_LITERATURE_20260812.md)
- [DINO-order 闭环结果](SEMANTIC_PROPOSAL_GATE_B_RESULT_20260815.md)
- [Pi3X 原始结果](FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md)
- [Pi3X 后续方向定义纠正](CEC_CONTRIBUTION_AND_IMPROVEMENT_AUDIT_20260906.md)
- [联合 PnP 结果](GEM_JOINT_REFERENCE_RESULT_20260907.md)
- [同 PnP 四臂授权结果](HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md)

## 4. 三个不能混淆的目标

1. **历史对应**：目标图与选中历史帧是否真正对应，证据来自哪些区域？
2. **定位可用性**：raw 输出的当前到目标 bearing 是否准确？
3. **导航收益**：使用该 bearing 是否比 native 更容易成功？

MASt3R 的局部匹配置信度主要服务第 1 层。它不会自动认证另一个模型 LingBot 的位姿，
更不会直接预测第 3 层。正确位置在墙后时，正确目标 bearing 也不是绕障碍路线。

早期若干 selector 以 covis≥0.5 为正、≤0.2 为负，中间忽略。
新共视结果表明低共视仍可能提供有用方向，不能把这些旧标签直接当成新模块的真值。
但也不能宣称所有旧学习失败均由此引起：Pi3X 已包含方向可用性监督，并有其他时序误差。

下一轮评分分开保存 role、连续 covis、实际目标直线 bearing 误差和导航结果：

\[
e_\theta=\arccos\!\left(\operatorname{clip}
(\widehat{\mathbf b}_{\rm raw}^{\mathsf T}\mathbf b_{\rm endpoint}^{\rm GT},-1,1)\right).
\]

GT 只在结果封存后的评分端读取。这里比较目标直线方向，不比较最短路首段，
不把目标相机朝向差当 bearing 误差。报告误差分布及 15/30/45/90° 诊断计数；
30° 沿用历史报告便于对账，不宣称是跨场景已确认的 controller 容差。
已经在成功半径内或相对向量退化时，角误差单独标记不适用，导航样本仍保留。

## 5. P0：先补全同状态、同假设的离线审计

### 总体

- 主开发总体：完整 159-query 共视谱，不能只挑 24 条差分失败。
- 协议鲁棒性检查：完整 90-query 新 actual-mono A 总体。
- 两批分别报告；场景/历史交集先核对，不称相互独立确认。
- 这些结果已经被看过，可用于开发和错误定位，不用作调完之后的新 held-out 确认。

### 每条导出什么

原始目标 RGB、raw 实际选择的历史帧 RGB 与索引、历史边界、DINO 原始分数、
当前状态、raw bearing、GEM 候选/拒绝原因/可用 PnP、图像和原收据身份。
初始判别样本一 query 一条，不把缓存重复调用当新样本。
后续 raw 换 anchor 或 bearing 漂移另列时序诊断，不混进初始判别精度。

对 raw、GEM 位姿做直接比较时，只使用物理状态一致的决策，或在相同记录状态上做无动作重放。
两臂轨迹分叉后，不能把各自 step 相同当作相同观测。
有些 GEM 只有有限 pose，没有完整几何证据；尚未计算的 PnP 是 missing，不是错误位姿。

优先复用已有点、匹配和位姿；缺失证据才补计算，不重新采集 Goal-A。
前一轮 52-row 派生表不含完整 raw 位姿/方向，不能假装已经足够完成本步骤。
当前本地只核实到部分 pilot RGB/日志；正式全量还需按原归档索引提取必要资产，
不以“路径存在”声称 249 套图像都已在本机。若访问 HPC，严格按现有共享 SSH 手册。

## 6. P1：固定 raw 假设，测试新的匹配证据

### 6.1 唯一要改变的量

保持 raw 实际候选、LingBot warm-reinsert 目标定位、当前位姿、坐标转换与
2.5 m 投影不变。新模块只输出是否使用这份 raw 提案，不重新排名或输出另一套 bearing。

第一阶段不是完整 MASt3R-SLAM。它是 **MASt3R 的匹配验证实验**，必须如实命名。
不用单个候选匹配结果代表完整 SLAM 的效果。

### 6.2 三个离线读出

| 读出 | 输入和用途 |
|---|---|
| DINO 分数基线 | 复用已冻结的校准来源；缺少可追溯 artifact 时先只报告分数，不凭记忆填阈值 |
| 当前 LightGlue 匹配证据 | 在同一个 raw anchor 上读出已有匹配；与原 geometry-first 选择不同的情况单列 |
| 冻结 MASt3R 匹配证据 | 同一对真实 RGB，使用官方模型与预处理，导出双向有效匹配及对应置信度 |

第一遍使用官方设置，不先扫阈值。若采用 MASt3R-SLAM 的匹配有效性定义，应明确复现
其 `valid_match` 与双向描述子置信度组合；不能把另一种 reciprocal-NN 分数当成相同量。
其典型候选分数可按官方逻辑记作：

\[
s(G,I_m)=\min\left(\frac{|V_{G\rightarrow m}|}{N_G},
\frac{|V_{m\rightarrow G}|}{N_m}\right),
\]

其中 V 包含几何匹配有效性和官方 Q-confidence 检查，N 为对应方向的采样点数。
当前官方配置 Q_conf=1.5、重定位 min_match_frac=0.3，作为一个不调参参照点。
这个 s 不是“真实共视率”，也不是“目标存在概率”；不能用 s=0.3 推导 GT covis≥0.3。
不能只用最大 Q 或最高几个匹配的均值冒充整张图的定位可信度。

复现时保存模型/代码版本、真实图像范围和点数分母，明确 padding 与坐标映射；
不要在同一实验悄悄更换原 CEC coverage 分母。

### 6.3 主要读数

- 各共视档，有多少 raw 正确方向被保留、有多少错误方向被放行。
- 原 Novel 的接管数与具体定位证据；重标注边界样本单列。
- 分数能否区分「有可靠匹配但 raw 位姿错误」与「匹配且 raw 位姿正确」。
- 在相同不可靠接管水平下的正确方向覆盖；同时展示官方默认操作点。
- 首次调用、缓存命中与新增 anchor 的耗时，峰值显存；不引用论文 15 FPS 代替实际延迟。

不能只看 AUC 或总 accuracy。低共视多保留一些样本，却放过更多错误目标，也不是已解决。
开发集曲线只回答有没有可利用信号；不能直接从 HM3D 已见结果选最佳点，称为新正式效果。

若默认点过严，但连续分数确实有区分力，最多进入一项单一、全局操作点的训练场景校准。
校准规则须在读取其测试折前固定，按 scene 划分，并对 DINO 基线同样处理。
旧 train40 中间共视样本不能因旧 ambiguous 标签直接丢弃；需补方向真值后分别报告。
一次校准后冻结，不为五个共视区间各选一个阈值。

### 6.4 如果匹配置信度不足以验证 raw 位姿

这是明确的分叉，不继续堆 score/margin/consensus。
可以检查高置信对应是否支持 **raw 已预测的目标变换**（使用可追溯的历史深度、
目标/历史内参与 raw 目标位姿作重投影读出），或直接转向完整定位后端对照。
这是待验证方向，不是本轮已经设计好的第二套证书，更不能再先列若干新阈值。
如果只能证明两图相关，却不能识别 raw 错位姿，结论是「匹配存在性信号有效」，不是
「raw bearing 已被认证」。

## 7. P2：小规模闭环，只验证一名候选

只有出现值得验证的新证据，才选定一个候选版本和固定操作点。
离线未显著不自动等于无效；离线通过也不自动升级主方法。

### 实现只需要四个关键等价性检查

1. 判别器强制接受：必须复现 raw 的请求、方向与动作。
2. 判别器强制拒绝：必须复现 native；不多写 FIFO、不多推进模型随机状态。
3. 正常判别：不更改 raw 选中的 anchor、原始向量或固定投影半径。
4. 查询图、临时定位状态不永久写入真实 causal history；证据缓存按相同输入复用。

这些检查限定实验改动，不扩展为一套新的运行时机制。raw 自己已有的候选更新规则保持不变；
若换 anchor，需要对新假设读取证据，不能沿用旧 anchor 的接受结论。

### 本机 smoke

在已消费总体中，按固定 scene/history 排序选择两个不同场景的完整 history；每个取
一条低共视、一条高共视、一条原 Novel，共 6 queries。选择不依据旧成败。
四臂：native、raw、GEM、raw+固定匹配验证，共 24 个 rollout。
它只验证接线、代价和是否出现明显系统性退化，不作为有效性或非劣效确认。

保持修复后的 RGB/单目深度、bounded pursuit、标准 Habitat try_step、真实转身后新 RGB
重规划、600 步预算和 1 m 到达判定；不恢复旧 snap/30% 重试。新臂不读取 role/GT。
标准 Habitat 碰撞几何、理想仿真状态反馈和 GT 到达仍属于仿真合约，不改写成真机能力。

通过 smoke 后再决定完整开发评测或新的冻结确认总体。所有 arm 在同机同进程配对、
顺序平衡，不用旧机器的 native 结果拼接新臂。完整共视五档按 scene 聚类；同 history 多
目标不能当独立样本。正式主比较与多重比较族在运行前确定。

报告 SR/SPL、Novel 的实际 gain/loss、不可靠接管、正确方向覆盖和延迟。
不能用“更少接管”自动代替“更高 SR”，也不能用不显著声称与 raw 非劣效。
如果要声称保留 raw 的成功率，需另行明确非劣效界值和统计规模，不能由 6-query smoke 给出。

## 8. 完整 MASt3R-SLAM 在什么位置

它是统一几何/重定位后端的强对照，不是本轮先验排除的方案，也不必等匹配验证成功才有研究价值。

第一步在已有 causal RGB 历史上独立建图，再将 goal 作为隔离的重定位查询：

```text
causal RGB -> MASt3R-SLAM history state + current pose
goal image -> retrieval + relocalization -> goal pose / unsuccessful
same map revision: current pose + goal pose -> bearing -> frozen NavDP
```

- 原版成功重定位会将 query 写入关键帧/检索库并优化地图；我们的 goal 必须在隔离副本中查询，
  不能冒充机器人实际观测。允许使用给定目标图作定位证据，但不改变真实历史成员身份。
- 当前与目标必须在同一地图坐标系、同一优化版本；不能将 MASt3R 目标位置直接减 LingBot 当前位置。
- 失败重定位不等于语义 Novel；官方配置也有匹配与置信度检查。
- 第一轮只换记忆定位后端，NavDP 的观测深度、controller 和执行器保持现有版本。
  若后续也换短程深度，单列另一项实验。
- 地图后端不能消费查询后的未来帧来改善先前定位；事后全序列全局优化不计在线能力。
- 单对匹配的成败不能推出完整 SLAM 的成败；相反，完整 SLAM 得分也不能归因于某一个匹配阈值。

如果完整后端更好且代价可接受，应考虑采用它。若其性能相当，就应承认“能在历史中定位目标”
不是我们独有的贡献，而不是额外添加模块来保住原叙事。

## 9. 资源与启动顺序

1. 先提取必要 RGB/JSON 与分层身份，避免复制整个大型运行目录；复用已完成 A。
2. 主环境和正在运行的他人 server 不动。若安装模型，使用独立环境并先做两图 CPU/GPU 接口 smoke，
   固定官方代码/权重和许可来源；本轮尚未下载或接受任何新的条款。
3. 编码后的历史图特征可按像素、预处理和模型身份缓存；联合 decoder 的 pair-conditioned 输出不能
   错当作只依赖单张图的缓存。先测实际冷/热延迟再估总时长。
4. 官方 README 的完整 SLAM 安装含 CUDA 扩展与额外依赖；匹配验证阶段不为此安装整套可视化后端。
5. 本地单位检查和上述 6-query smoke 后，才考虑 HPC。HPC 依照
   [共享 SSH 手册](HPC_SHARED_SSH_OPERATIONS_20260816.md)，不另造连接方式。
6. 本设计不指定“今晚长训 8 小时”；当前问题首先是信号和比较对象，而不是训练时长。

## 10. 结果出来后如何决策

| 实测情况 | 结论与下一动作 |
|---|---|
| raw 方向普遍正确；MASt3R 能保留低共视真匹配并少放错匹配 | 进入 raw+匹配验证的小闭环，不更换定位器 |
| MASt3R 只能区分图像对应，不能区分 raw 错误位姿 | 不宣称已完成方向认证；检查固定 raw 位姿的一致性或比较完整后端 |
| raw 在失败组的目标位姿/时序更新本身错误 | 优先统一定位/坐标更新，单纯判别器只能拒绝，不能纠正 |
| MASt3R 官方点仍大量拒绝低共视，连续分数也不分离 | 不启动 confidence MLP 长训；记录匹配证据不足，换研究问题 |
| MASt3R-SLAM 独立后端兼顾定位、拒绝和代价 | 接相同 NavDP 检验；允许替代当前几何后端 |
| 新方法仅多放行、Novel 干扰同步上升，或闭环无净收益 | 不升级主方法，不用“更优雅”替代结果 |

这份设计不保证恢复 24 条损失，也不承诺 0.1–1.0 全区间成功。
其价值是用一次清楚的实验区分「缺匹配证据」「缺位姿可靠性」「缺控制收益」，避免再次混为一谈。

## 11. 本轮核查出处

本地两个派生文件的 SHA-256（不是远端原始 summary 的 SHA）：

- `REPAIRED_HM3D_FINAL_RESULTS_20260910.json`：
  `79e7c3636a651b5109a1b629d8b0ba54916f63a602761e1e27681cba3c4f3077`。
- `HM3D_COVISIBILITY_REJECTION_AUDIT_20260910.json`：
  `6c96d6927bbe52680300bafc0401c675afb70bed8e820ee5d91df2867ff216ff`。

核对到的代码入口：

- `NavDP/baselines/memnav/policy_agent.py`：`plan()` raw 检索、warm-reinsert、目标位姿缓存，
  以及 `certified_relocalize()` 的独立几何路径。
- `InternNav/internnav/model/basemodel/memnav/lingbot_stream.py`：
  `goal_append_warm()`、`camera_pose()`、上下文恢复。
- `MemNavData/revisit_bearing_adapter.py`：raw/verified 的共同固定半径投影。
- `MemNavData/compare_pi3x_spatial_proof_to_certificate.py`：旧方向标签与 strict-negative 的区别。

公开原始来源，2026-09-10 查询：

- [MASt3R 官方仓库](https://github.com/naver/mast3r)：双视图几何与局部描述子、官方推理入口。
- [MASt3R-SLAM 官方项目](https://edexheim.github.io/mast3r-slam/)：单目跟踪、融合、回环和全局优化。
- [官方匹配验证](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/mast3r_slam/global_opt.py)：
  双向对应置信度、有效匹配比例与边验证。
- [官方配置](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/config/base.yaml)：上述默认操作点。
- [官方重定位入口](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/main.py)：重定位成功后的关键帧/数据库更新。

本地检索未发现 MASt3R-SLAM 的已运行脚本或结果；已有文献/计划中提到它不等于已做过实验。
**本轮新成果是去重审计与下一实验设计，不是新的 SR、定位精度或运行时 benchmark。**
