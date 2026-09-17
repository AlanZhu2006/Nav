# 修复版 actual mono-A → mixed-role：本机端到端协议

2026-09-08。执行前固定。本轮验证完整采集/构造/评测链，不产生新版正式论文结论。

## 数据与分母

- 固定取前一轮本机清单前两条源 episode：`gxdoqLR6rwA/episode_0000`、
  `pLe4wQe7qrG/episode_0000`。使用原 source 的起点和 Goal-A 图，不使用其旧 A 动作或位姿作为策略轨迹。
- 原始 MP3D scene/source 已在本机；正式 HM3D parent manifest 已找到，但其场景资产不在本机已检查位置。
  本轮不把 MP3D 小测称为 HM3D 泛化，也不另外选更容易的 source。
- 两条 A 各重新执行一次，seed=0，最多 600 动作，原 8-tick 重规划、1 m 到达和 150-tick 停滞退出。
- A 失败或历史不足按原 materializer 记录 attrition；不会 replay expert 或旧 metric-A 来补齐。
- 从新 A 的真实 RGB 历史按现有 Final14 构造器形成 Natural Novel/standard Revisit。
  不改距离 [2,9] m、共视 Novel<0.1/Revisit[0.55,0.90]、最早 frame39、末端 margin16。
- 保留构造器现有确定性位姿网格、方向分层和相机元数据定义，隔离本轮执行版本；
  已发现 FY 元数据偏差另有审计，不在此轮暗改几何标签或 source 图。
- 所有两条 A 先完成，再构造全部查询，再开始 query 三臂。没有可构造历史也是合法结果。

## 唯一执行配置

沿用已完成四历史桥接的配置，不再根据本轮 SR 加机制：

1. NavDP 模型端正确 RGB；当前、目标、短期历史入口一致。
2. 第一 40 帧因果高度尺度；LingBot dense depth 从 pad-518 恢复至当前 RGB 栅格。
3. bounded pursuit + 一次标准 Habitat try_step；不吸附目标落点，不作 30% 重试。
4. raw/CEC 共用后方 PointGoal 的真实朝向适配，每步新 RGB，转后重新规划；A 不产生 memory PointGoal。
5. 模型请求不携带 simulator odometry；模型不读 GT 目标、最短路或 role。
6. 稀疏 CEC reference depth 为此前桥接的 canonical，未同时切换缓存/定位版本。
7. 保留原 NavDP 低分搜索、CEC 拒绝语义；没有新增失败 fallback 或重新训练。

## 三臂和归档

- 每个新历史产生 Novel/Revisit 两查询；每查询 native / raw fixed / CEC 三臂，独立 reset 并精确重放该新 A。
- 角色只在 evaluator 中决定取哪张图，不进入模型。三臂在同一私有模型服务和 GPU 上完成，按历史轮换顺序。
- 每次保存完整候选/critic、输入 RGB/depth 收据、实际动作前后状态、末步终点和计时。
- A 的 sensor-depth-consumed 必须 false；空尺度前缀按原契约 zero-depth，不是偷偷替换 metric depth。
- 独立复算 SR/SPL、对账全部三臂历史和起点、检查未接管时与 native 的实际前缀；失败视频也保存。
- 运动学代理 + 理想低层位姿 + NavMesh 碰撞的边界不变；GT 到达仍不是自主视觉 STOP。

## 收尾与正式评测

本轮成功标准是端到端链和收据正确，不按 N≤2 的 SR 选择方法或阈值。
先报告 A 成功/可构造数量，再报告所有完成的三臂查询；不把臂数当独立样本量。
生产默认、论文、真机不改；不自动 push 或提交全量 HPC。

后续正式 HM3D 应复用原 source 清单重新采集 A；不能只挑原来 28 个成功历史。
HPC 仅通过 `alantorch` 的 `yz11502` 共享连接，依赖按手册在实际解释器/container 下检查，
优先已验证的 `h100_tandon/a100_tandon`；每个配对单元先实测，再确定 1 小时是否足够。
