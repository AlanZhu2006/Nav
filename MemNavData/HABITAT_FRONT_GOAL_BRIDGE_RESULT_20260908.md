# 后向 PointGoal 域适配：完整本机结果

2026-09-08。16/16 查询完成；独立复算、源码核验和全部第一视角视频通过。

## 结论

保留 NavDP、CEC、2.5 m 与原控制预算，将已发出的后向 PointGoal 先兑现为限幅真实转向，
用转后新 RGB 重新规划，解决了本轮两条 CEC 零轨迹停滞。raw memory 使用完全相同的接口也受益。

| 记忆来源 | 不做后向适配 | 做后向适配 | 配对 |
|---|---:|---:|---:|
| raw fixed bearing | 2/4 | 4/4 | +2/−0，p=0.5 |
| CEC | 2/4 | 4/4 | +2/−0，p=0.5 |

**分母为四条已消费历史，不是 16 个独立样本。** 这是针对已知接口缺陷设计的机制验证，
不证明新场景 SR，也不证明 CEC 比 raw 更强。两个记忆来源都救回 pLe、mJX，
不能将这两行视为四个独立 gain 来报显著性。

本轮所有臂已使用正确 RGB、对齐的单目 depth、bounded pursuit 和标准 try_step。
不是直接拿旧 snap/BGR 的结果与最新开启臂横减；旧六臂结果保留在另一份报告。

## 逐例结果

表内为“是否到达 / 动作数 / 实际路程 m”，动作数包含全部原地转向。

| 历史 | raw off | raw on | CEC off | CEC on |
|---|---:|---:|---:|---:|
| gxdoq | 1 / 125 / 4.198 | 1 / 111 / 2.539 | 1 / 125 / 4.103 | 1 / 113 / 2.599 |
| pLe | 0 / 323 / 3.422 | 1 / 100 / 2.326 | 0 / 151 / 0.000 | 1 / 103 / 2.391 |
| yqst | 1 / 106 / 3.619 | 1 / 115 / 2.565 | 1 / 106 / 3.621 | 1 / 116 / 2.588 |
| mJX | 0 / 151 / 0.000 | 1 / 96 / 2.037 | 0 / 151 / 0.000 | 1 / 95 / 2.065 |

- gxdoq：保留两种记忆的成功，步数和实际平移路程下降。
- pLe：CEC 从零平移失败变为到达；raw off 有过推进但停在距目标约 1.84 m，on 到达。
- yqst：均保留成功，但 on 多花 9–10 动作。不能称每条都更快。
- mJX：两种记忆的零平移失败均恢复到达。

开启臂的 SPL 都是 1.0。它按实际平移路程与原 shortest-path distance 计算，
采用固定 1 m 到达半径，不计原地 yaw 的路程；**不等于零转向代价或最短执行时间**。
因此同时给出动作数，尤其保留 yqst 的反向效率证据。

## 转向和几何检查

每个开启臂只触发一次转向，转后均重新产生非零 XY 轨迹。

| 历史 | raw / CEC 转向动作数 | raw / CEC LingBot yaw 差 ° | raw / CEC 虚假平移估计 m |
|---|---:|---:|---:|
| gxdoq | 36 / 36 | 2.01 / 1.83 | 0.289 / 0.311 |
| pLe | 36 / 37 | 6.58 / 4.58 | 0.364 / 0.373 |
| yqst | 34 / 34 | 1.56 / 1.46 | 0.115 / 0.095 |
| mJX | 40 / 40 | 1.63 / 1.74 | 0.122 / 0.145 |

所有转向的真实仿真平移为 0。表中估计平移乘了原冻结相机高度尺度，不是真实 metric GT。
这说明纯转动仍使 LingBot 产生一定虚假平移；高度先验没有消除它。
本批转后 bearing 保持可用并到达，不能外推为任意长程或任意转动都稳定。

实际转向角约 149–178°，每 tick 不超过原 4.5° 上限；没有一步改写成目标朝向。
转向中每帧 RGB 只进入 LingBot 一次；原定 decision tick 仅重放 NavDP FIFO，不重采样。
第一次计划仍计算并留档，但转向时不执行；转完的下一张新图立即规划。
该初始丢弃样本的计算没有隐藏，本接口诊断不是最终推理延迟优化。

## 核验与边界

- 同一模型服务进程内成对运行，奇偶历史平衡顺序；各记忆来源 off/on 的首次输入/完整输出一致。
- 共享真实旧 metric-NavDP-A；不是新 full-mono-A，也不是 expert-A。
- 16 条路径、末步后坐标、SR、SPL 独立重算；241 份深度 producer/readout 张量转换独立验证。
- 运行期间 source/weights 未改变，`changed_source_files=[]`；16 个成功/失败视频全部导出。
- 原 600 tick 总预算和 150 tick 位置停滞退出保留；转身没有另加预算。
- 模型请求移除了全部六个 simulator-odometry 表单字段；LingBot 逐帧收据中未收到该类 motion receipt。
- 方向来自已经发出的 PointGoal，不来自 GT 目标、角色标签、最短路线或新增 critic 门。
- 仍是 Habitat 运动学圆形代理、标准 NavMesh 碰撞和理想低层位姿反馈，不是无 GT 物理复现。
- 到达仍由 evaluator 的 GT 平面距离 <1 m 判定，不是自主视觉 STOP，也不检查目标朝向。

原生非接管分支另按 `HABITAT_NATIVE_REQUEST_EQUIVALENCE_PROTOCOL_20260908.md` 做短测；
此处的 Revisit 成功不能单独证明 Novel 安全性。

## 最小设计与论文含义

这项修复分三层：模型输入的 RGB/depth 对齐；PointGoal 的前向输入域适配；
轨迹跟踪器的有限运动请求与环境碰撞分离。不增加第三个专家、学习门或失败搜索。

必须诚实说明：转向动作由低层适配器执行，不是 diffusion 输出。
NavDP 权重仍冻结，但不能仍写“所有动作均直接由 NavDP 产生”。
它是共有控制接口修复，不应当包装成 CEC 的独有贡献。

不能用 N=4 替换正式论文结果。正式重跑优先新版本 actual mono-A + mixed-role native/raw/CEC；
旧版离线证据不自动作废，闭环表格的版本边界见 `HABITAT_REPAIR_RERUN_SCOPE_20260908.md`。

## 原始路径

```
/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_minimal_repair_20260908/front_goal_bridge_v1/
```

其中 `summary.json`、`independent_verification.json`、`evaluation/`、`source_snapshot/`
和 `first_person/` 分别保留结果、复算、原始执行、源码与视频。
此轮没有更改生产默认、HPC、真机或论文，也没有 commit/push。
