# 去面积条件：本机导航机制小测

本协议先于本轮任何导航结果固定。此前定位组件结果已经读取，故为已消费数据上的
定向机制实验，不是新场景确认或无偏总体性能估计。

## 样本与目的

取原 Fundamental 面积预检查中断、且按 CEC 实际可检索域 frame≥8 计算仍处于
[0.1,0.3) 的全部两个查询，按 history 顺序执行；不只取组件测试新增接受的正例。

| history | scene | Revisit task / q | 配套原 Novel task / q |
|---|---|---|---|
| 2 | vBMLrTe4uLA | 37 / 0.1853 | 154 / 0 |
| 12 | SiKqEZx7Ejt | 71 / 0.2647 | 10 / 0 |

选择依据是原程序停止位置及修正后的共视范围，不是新导航成败。它是此前“按固定
history 顺序”计划的具体化，但不是取全体历史最前两条。两个 Revisit 的组件结果
已知：一个去覆盖后仍拒绝，一个获得 31 内点的方向；两者都保留。

四个查询分别跑 native、raw fixed、原 strict CEC、CEC without coverage，合计
16 条 query rollout。采用四行循环 Latin 顺序，每个 arm 在每个顺序位置各一次。
一个私有服务器进程组完成全部比较，每臂重新 reset 并重放同一实际 online-A RGB。
不重新采 A，不改目标，不用 expert-A 替代。A 来源为旧实际 mono 执行，query 使用
修复后的执行器；因此不是“新修复端到端 A+query”结果。

## 不变项

- NavDP/LingBot/checkpoint、相机内参和高度先验、随机种子、起点、目标图不变。
- 私有端口与 buffer，不调用真机的 8888 / 18888。
- bounded_standard 执行、RGB 通道修复、source_rgb 深度光栅、front-goal 实际转身。
  Habitat try_step 只作模拟器碰撞响应；不把 GT 可行走落点投影结果用于选动作。
- max_steps=600，exec_horizon=8；目标位置距离<1 m 判成功，记录最终动作后位置并复算 SPL。
- CEC top-8、geometry-first、匹配、PnP、16 内点、2 px RMSE、2.5 m residual 均不变。
  唯一消融为 Fundamental 和 PnP 两处双侧覆盖面积硬条件；覆盖仍参与原候选排序。
- 保留运行时原候选域。共视同时记录 q(frame≥39)、q(CEC frame≥8)、raw 首查询域。
  不通过改可检索历史来改变样本难度，不按共视档选不同门限。
- role / covis / GT pose 只供 evaluator 分组和评分，不进入网络或执行请求。

## 验收与报告

先验证四臂接口与修复执行器，逐条保存全部成功和失败。独立复算 SR/SPL、真实位移、
首查询 RGB、目标与 A replay 一致性；检查未接管的 CEC 是否逐动作等同 native。
异常退出记为基础设施未完成，不算导航失败、不改预算掩盖问题。

汇报 N、逐查询四臂、接受/接管、方向和行动诊断；不以 N=2 宣称泛化显著增益。
如果方向恢复却没有导航恢复，应继续区分方向与局部可通行性，而不是继续调面积条件。
本轮不提交 HPC GPU、不改变默认方法、不更新论文结论。
