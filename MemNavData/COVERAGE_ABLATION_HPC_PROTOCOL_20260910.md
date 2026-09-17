# CEC 覆盖面积条件：完整共视总体的四臂消融

日期：2026-09-10。已消费 HM3D 总体上的预先固定扩展试验，不是 blind/held-out 确认。
本机16条定向试验已完成并独立复核；本协议在新增 HPC 导航结果之前固定。

## 问题与唯一改动

问题：删除图像凸包覆盖面积这个充分条件，能否恢复低共视 Revisit 的闭环收益，
同时保持高共视能力，并避免 raw memory 在无历史支持目标上的干扰？

只比较 `strict_certificate` 与 `certificate_without_coverage`：后者取消原 Fundamental
和 PnP 两处 query/reference hull coverage 硬检查。top8、temporal NMS、geometry-first
排序（仍使用覆盖统计）、SuperPoint/LightGlue、深度、PnP解算、16内点、2px RMSE、
有限位姿检查、2.5m residual、冻结 NavDP 和修复执行器全部相同。不训练、不网格搜索、
不按共视档设不同参数，不更换定位器，不把去掉面积检查称为整个方法没有阈值。

## 数据与配对

- 使用原 population.json，SHA256
  `182f3a6d2519d5b2c178b88345db4d0bb678088dd88487cfecaad9814c6fdaaf`。
- 159 queries：131支持查询、28原 Natural Novel；28 histories / 21 scenes。
- 全部159保留，不只选此前 raw 成功/CEC失败的目标，也不根据新运行结果补样。
- A来自原实际 mono NavDP online历史，不重新采A。该A使用旧执行器，query使用修复
  执行器；因此这是 query-stage 消融，不冒充新修复系统的端到端 A+query 评测。
- 每查询四臂：native / raw_fixed / cec / cec_no_coverage，共636 rollout。
  四臂位于同一array element、同节点、同进程服务，独立reset重放同一A。
  四臂顺序按task index循环Latin排序；原任务索引、目标、seed不变。
- 不复用旧HPC或本机结果作为本轮对照。新CEC和新raw/native全部重新配对执行。

## 分层定义与分析

原共视只覆盖frame≥39，本轮预先定义主分析共视为实际CEC候选域frame≥8到A结束的
最大已存共视。它只是离线分层，不改变候选可见范围或输入网络。
同时保留原q39和raw首决策候选域q，不能悄悄覆盖旧注释。

| 实际CEC历史共视 | 支持查询数 |
|---|---:|
| [0.1,0.3) | 14 |
| [0.3,0.5) | 26 |
| [0.5,0.7) | 23 |
| [0.7,0.9) | 32 |
| [0.9,1.0] | 36 |

主效应：合并q∈[0.1,0.5)的40个支持查询，新CEC对原CEC的配对SR差。
报告N、独立history/scene数、gain/loss、exact McNemar和20,000次scene-cluster bootstrap
95%CI。相邻目标共享history/scene，不把查询当独立场景。

次分析：各共视档的 SR/SPL、新CEC对raw/native、高共视91查询的损益。
五档分层p值对每个比较族做Holm校正，不从最好的一档挑结论。
原q39分组只作定义敏感性分析；排除本机导航已测history2/12的结果另报，仍不称盲测。

Novel：保留原28个控制，报告各方法SR、新旧CEC接管次数、相对native救回/破坏、
未接管时逐动作精确一致性。history22原Novel的q约0.1167，原身份不改；额外报告去掉
这一个边界例的27查询敏感性分析。不能把所有Novel accept直接判为定位误报，不能把
“未见显著损失”写成安全等价。零损失的小样本也不构成安全保证。

不汇报人为分档混合后的总SR作为自然部署SR。

## 固定执行合约与指标

复用本机已验证代码：bounded_standard、rgb_v1、source_rgb monocular depth、front-goal
真实转身；历史/当前 RGB 才进入模型，role、共视、GT位姿和GT运动收据不进入请求。
不使用可行走落点投影来修正策略动作；Habitat try_step仅承担模拟器碰撞响应。
max_steps=600、exec_horizon=8、成功距离<1m；保留原停滞提前结束条件。
GT位置/测地线用于SR/SPL评分，末步动作后落点必须记录；这不是自主视觉STOP评测。

## HPC计划

复用已验证的A100、CUDA12.8容器、两个现有conda环境、固定模型/依赖及OpenCV补充目录。
只建立新source-only immutable bundle，不改旧封存版本或共享环境，不升级依赖。

先跑同协议 task10（Novel）和task71（低共视）两个集成gate；检查加载、四臂、
授权差异收据、配对、执行器、末步SPL与归档，不以SR好坏决定是否放行其余157查询。
gate成功后自动运行其余157，gate两查询本身纳入完整159，不另算独立证据。
每元素1卡、10 CPU、96GB、1小时；最多4并发。48分钟进程上限预留归档时间，
不改变导航步数。若gate显示时限不足，只调整资源/时限，不缩短方法或预算。

提交前检查myquota字节/文件数、8个role×arm CLI、完整159输入、容器导入、源码/模型哈希。
逐帧证据写node-local，结束后逐文件哈希归档到scratch。完整缺失清单由afterany汇总输出；
缺任务不算导航失败，也不输出完整SR。异常只能重跑exact failed index并保留原attempt。

## 决策

- 低共视收益可重复且Novel/高共视未发现实质性新损害：再考虑把面积检查简化为诊断量。
- 同时新增正确接管与明显错误干预：如实报告风险—覆盖变化，不继续在本批数据调阈值。
- 方向授权增加而SR不变：转向定位/局部可通行性分析，不把“更多accept”当收益。

本轮不自动更换论文主方法，也不声称已解决所有低共视问题。
