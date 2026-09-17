# 修复后已有 Revisit：独立三臂集成验证

2026-09-09。用户已同意继续。仅验证已有合法查询的修复后闭环，
不是 mixed-role 补齐、正式确认或新的 A collection；不改原实验分母。

## 固定人口与问题

此前六个 source 中，全部具有 standard Revisit 构造记录的两个历史都纳入：

| scene / episode | actual mono-A 来源 | 已选 Revisit |
|---|---|---|
| rJhMRvNn4DS / episode_0000 | 17188986 | frame 55，geo 3.1079 m，max-covis 0.7186 |
| jgPBycuV1Jq / episode_0000 | 17196996_0 | frame 175，geo 2.3251 m，max-covis 0.7197 |

两者原 Natural Novel 均未构造成功。保留这些事实；本测试不创建假 Novel，
不放宽 2–9 m、frame 39 或共视阈值，不用旧 metric-A 代替 mono-A。
仅按原记录的 source_frame / render_attempt 重建已经选定的标准目标，
独立重算共视与几何并与旧记录比较，绝不重新搜索或依 SR 选择目标。

问题是：输入/执行修复之后，真实 mono 历史的检索、认证、方向接管与控制能否连起来？
2 histories × native / raw_fixed / CEC = 6 个查询 rollout。
第一历史顺序 native→raw_fixed→CEC；第二历史反序 CEC→raw_fixed→native。
这是小样本次序控制，不宣称实现了完整三位置平衡。

## 固定方法与执行

- 复用封存代码 c8cf8c60e7efd55f 和同一模型、依赖，新增的只是独立查询编排入口。
- runtime 不读取 Novel/Revisit 标签、GT pose、共视或目标 GT 深度。
- 所有历史帧只重放 RGB；NavDP FIFO 只重放原决策帧，不重跑 A、不抽样 A policy。
- query 为 LingBot 单目深度，source-RGB raster、RGB 色序修复。
- CEC 使用 canonical reference depth，原几何证书与固定 2.5 m bearing；不启用 graph rescue。
- bounded_standard 执行器：有界 pure pursuit + Habitat 标准 try_step 碰撞响应；
  不用 snap_point 修正策略落点。NavMesh 仍属于仿真环境碰撞实现，不宣称 GT-free physics。
- 原 PointGoal heading adapter 保持开启；一轮转向期间重放新 RGB，不重新抽样策略。
- 每条上限 600 ticks、每次计划 8 ticks；成功采用 evaluator 平面距离 <1 m，
  不是自主 STOP；记录末步落点后精确积分 SPL。

## 运行与验证

先本机 CPU 单元测试和三臂 CLI dry run，再按共享 SSH 手册提交 A100 / 1 GPU / 1 小时。
不再尝试 H100 来排除此栈上已观察到的连续运行故障；不改共享环境。
两条历史和六臂使用同一作业、同一节点、同一对私有服务，各臂 reset 后精确重放历史。
buffer / runtime 放节点临时盘；保留失败输出，不自动补 source 或启动正式批次。

验收记录：目标和历史 SHA、相同起点/首帧、重放队列、RGB-only HTTP 边界、
逐动作真实位移、单目深度收据、CEC accept 与接管次数、完整第一视角录像。
如果 CEC 不接管，另核验其请求、输入深度、候选轨迹和实际动作是否与 native 精确一致。
方法失败也完整保留；不把基础设施错误记作导航失败。

输出只报告每场景结果、N 与配对增减。N=2 不做泛化或显著性主张；
即使六条都跑通，也不等于原 mixed-role 查询 gate 通过。
