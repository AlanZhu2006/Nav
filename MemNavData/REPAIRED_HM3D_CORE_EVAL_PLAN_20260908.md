# 修复版 HM3D 核心评测准备：不混用旧 A 与新 query

2026-09-08。正式核心批次仍是准备阶段；同日晚本机端到端集成完成后，
两源 HPC 迁移验证 **17188986** 已正常完成：2 A / 0 query；因为没有可构造 role-pair，
尚无新版 HPC memory-query SR。固定 rank 2–5 四源追加 **17196996** 已结束：
shard 0 A=2/2 成功但 0 query；shard 1 首条 A SIGABRT 后，已由同代码 A100
**17201733_1** 完整补齐，另两条 A 均成功、verifier=true。
四源最终 A=4/4、query=0；已停止自动扩样，不改方法或构造阈值。
详见 [初始 gate](REPAIRED_HM3D_GATE_STATUS_20260908.md) 与
[四源追加状态](REPAIRED_HM3D_EXTENSION_STATUS_20260908.md)。
两源 gate 不等于本文件的正式核心配对批次已经运行或完成。

## 1. 正式要回答的问题

在修复后的统一输入/目标接口/执行版本上，因果单目历史是否仍提升 frozen NavDP 的 Revisit，
以及 CEC 相比直接 raw memory 的 Novel 干扰与 Revisit utility 是什么。

不以新 CEC 对比旧 native，不通过只重跑旧失败案例修补总 SR，也不将明确的控制接口收益
包装成 CEC 的独有贡献。

## 2. 可以复用什么

旧 HM3D source parent 已在本机核对：

```
.diagnostics/hm3d_fresh_fullmono_mixed_role_20260820/pulled_20260828/sealed_inputs/parent_manifest.json
SHA256: a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
```

它含 54 个冻结场景、49 个非空场景、**196 条已生成源 episode**，而非目标值 216。
五个当时没有生成完整 source 的场景为：`kJJyRFXVpx2`、`c3WKCnkEdha`、`F7EAMsdDASd`、
`u8ug2rtNARf`、`hyFzGGJCSYs`。不在本轮临时补新 source 填平这个缺口。

可以复用已封存 scene asset、source 初始状态和 Goal-A 图；它们定义任务，不定义新 A 行为。
不能复用旧 A 动作轨迹、旧 CEC/native 查询结果，或直接限定为旧 28 个成功历史。
本次也不再把这些已消费场景称为首次 unseen confirmation。

远端 source 根目录（提交前仍须只读检查存在和 SHA）：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6/source_generation
```

## 3. 三阶段依赖

```text
冻结源起点/Goal-A 图 + 新执行版本
              ↓
actual mono-A collection（native，不接受 memory 指引）
              ↓
保存全部 A 结局，成功且足够长的真实历史才进入几何构造
              ↓
Natural Novel / standard Revisit 构造与 population 封存
              ↓
每个 history 的 native / raw / CEC 六个 role-arm query 同进程配对
              ↓
独立 SR/SPL、Novel 接管、非接管等价、成本与全量 attrition 汇总
```

预算与查询构造阈值沿用原协议。统计选择只能读取 A 结局与构造可行性，不能读取 query arm SR。
原协议的场景前缀 `[30,36,42,48,54]`、目标 24 histories / 15 scenes 可作为复用方案；
提交前应把最终采用的源范围和选择规则写成单一新协议，不能运行后再选。

新 A 数量与可构造查询数可能不同于旧结果；这是新版本真实系统的分母，须完整报告。
训练免费不意味着可无记录地按结果调整评测分母。

## 4. 固定共有接口，不同时改变研究问题

- 正确 RGB 通道，当前/目标/context 一致。
- 因果 first40 相机高度尺度；dense depth 恢复至原 RGB 栅格，不使用 sensor depth。
- bounded pursuit、一次标准 try_step；所有臂相同。
- raw/CEC 共有已发出的后方 PointGoal 朝向适配，每步新 RGB、转后重规划、相同总预算。
- 保留原证书参数、2.5 m 残差和 native 自身低 critic 行为，不新增 ranker、route graph 或恢复门。
- 模型请求不传理想 simulator odometry；理想位姿只保留在低层执行和 evaluator。
- 稀疏 reference depth/缓存来源必须与本机通过版本一致并明确记入新协议，不能默默切换。

相机 FY 元数据的小偏差已另外量化。正式构造前应明确选择：保持原定义作执行版桥接，
或从实际相机矩阵导出新 K 并记录新构造版本；不能用“没有改阈值”掩盖投影定义也变了。

## 5. HPC 准备条件

已阅读 `HPC_SHARED_SSH_OPERATIONS_20260816.md` 与 `HPC_HARDENING_20260821.md`。
两文档存在历史冲突时，以用户要求复用 `alantorch`/`yz11502` 权威共享连接为准；
不按旧故障章节临时创建另一账户/socket 作为替代。模型评测优先已验证的
`h100_tandon/a100_tandon`，不沿用旧文档中含 public 分区的示例。

提交前必须完成：

1. 本机 actual mono-A → 构造 → 三臂 → independent verifier 的完整链；合法零历史也要正确收尾。
2. 新入口摆脱工作站绝对数据/解释器/权重路径，完整依赖闭包打包并做 relocated import 检查。
3. 远端共享连接身份核验；source/scene/parquet/goal 校验；`myquota` 同时检查字节与文件数。
4. 正式 container/绝对解释器/实际 mounts 下的所有 arm CLI 干跑、渲染与真实服务 smoke。
5. 从新 sealed population 生成精确 array 范围；一个配对单元的臂不拆到不同节点。
6. 以实测启动+A 或全套 query wall time 确定每单元时限；目标使用 1 小时，不能未经测量保证够用。
7. 输出文件数预算、端口锁与只清理本任务 PID；小代码 bundle 用共享 SSH，大资产优先 Globus。

两源集成入口已完成路径解耦、独立源码包和远端实际解释器预检，首次 HPC gate 已结束。
四源追加也通过实际远端预检并已提交；目前仍待完整 mixed-role query 结果。
正式核心批次的 population/array 入口仍需在 gate 通过后完成。
不把“代码已写”、CPU 测试通过或作业已排队记录成导航完成。

## 6. 最小报告

先报告 source/A/构造数量，再报告各 role 的 SR、SPL、配对增减和 scene-cluster 区间。
同时保留动作数、转向动作、零参考停滞、实际路程和端到端 wall time，避免 SPL 忽略原地转向成本。
Novel 接管不能仅凭标签直接判为误检；需要说明当时因果历史支持，但未接管时必须验证实际 native 等价。

只有核心结果完成后，再确定跨 controller、深度、授权和连续目标表的补跑批次。
不因为本机 N≤2 看起来好就开始把全部旧论文数字换成新的。
