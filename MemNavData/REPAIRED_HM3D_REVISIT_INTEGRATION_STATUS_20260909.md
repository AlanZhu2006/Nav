# 修复版已有 Revisit：三臂集成进度

北京时间 **2026-09-09 03:40** 再核查：作业 **17208480** 已在 **ga008 / A100**
完成，01:13:51–01:27:44，用时 13 分 53 秒；独立 verifier 为 true，六条第一视角录像齐全。
两个目标的严格还原通过；这只是 N=2 查询集成验证，不是论文确认批次。

| scene | native | raw fixed | CEC |
|---|---:|---:|---:|
| rJhMRvNn4DS | 失败，378 ticks，SPL 0 | 成功，94 ticks，SPL 1 | 成功，97 ticks，SPL 1 |
| jgPBycuV1Jq | 成功，287 ticks，SPL 0.295254 | 成功，86 ticks，SPL 1 | 成功，85 ticks，SPL 1 |

合计 native 1/2，raw 2/2，CEC 2/2；CEC 对 native +1/−0，对 raw 打平。
native 的 rJh 失败触发既有 stuck 条件，不是运行报错；两个 memory 臂都完成转身后推进。
不得据此宣称统计显著或 CEC 超过 raw。

实际还原测量：rJh 为 3.107867 m / covis 0.718611，jgP 为 2.325051 m / covis 0.719722；
与各自旧选择记录的逐项比较通过。未重新搜索目标。

## 本轮实际做了什么

先前六个 A source 中，两个已有合法 standard Revisit 的历史全部纳入，
不按导航结果筛选、不补 A、不改查询规则。将它们从失败的 Natural Novel 配对构造中
独立出来，只做 native / raw fixed memory / CEC 的 **2 histories × 3 arms** 集成验证。
原 mixed-role 仍是零条合法 pair；这六个查询绝不会填入原分母。

- 两个目标均由旧收据的 source frame / render attempt 唯一还原，不重新搜索。
- 先在 GPU 上重新计算原目标共视和几何，确认与记录一致，再启动任何 query arm。
- 复用已执行 mono-A 的每帧 RGB 和 NavDP 原决策帧，零 A policy 重采样。
- 复用封存模型/权重/依赖、canonical CEC、2.5 m bearing 和既有执行修复。
- 唯一共享执行工具改动是增加可选独立 query callback；旧入口默认行为不变。
- 新协议、构造、单查询编排与复算独立存放；没有伪造 Novel、修改全局 role-pair validator。

## 验证与运行边界

本机：新增 6 测试、原 integration 21、执行/方向/深度/权限边界 54，共 **81 项通过**；
三臂 CLI dry-run 通过，未创建输出目录。
远端真实 Habitat/container：6 测试和三臂 CLI 同样通过，再经 safe_sbatch lint/test-only 提交。
GPU 节点还会再次检查导入来源、依赖、模型 SHA、三臂 CLI 和目标重渲染。

20 KB addon 经既有共享 SSH/SCP 上传并逐项验证。模型及原大 bundle 不重新复制，
buffer/runtime 使用节点临时盘，未改共享 conda、未动机器人、论文或其他人的服务。
SCP 建立 channel 有延迟但最终成功；临时准备的本地隧道未用于传输，已关闭。

## 应如何读结果

本轮必须同时保存成功与失败轨迹、末步落点、精确 SPL、深度/转向收据和完整第一视角视频。
若 CEC 未接管，检查其候选、动作和深度与 native 是否一致。
若发生回头、停滞或碰撞，按真实动作归因，不把 runtime 报错计作 SR 失败。

只有 **N=2**；用于确定修复后链路是否可工作，不支持泛化、显著性或论文主表替换。
本轮结束不自动扩样、不自动提交正式批次。

## 入口与证据

- 协议：[REPAIRED_HM3D_REVISIT_INTEGRATION_PROTOCOL_20260909.md](REPAIRED_HM3D_REVISIT_INTEGRATION_PROTOCOL_20260909.md)
- 编排：[repaired_revisit_integration.py](repaired_revisit_integration.py)
- 提交绑定：[REPAIRED_HM3D_REVISIT_INTEGRATION_SUBMISSION_20260909.json](REPAIRED_HM3D_REVISIT_INTEGRATION_SUBMISSION_20260909.json)

运行根目录：
`/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_revisit_integration_20260909/run_b90fce8509fd376e`

封存原代码 `c8cf8c60e7efd55f`；新增 addon `b90fce8509fd376e`。
结果文件在 `integration/summary.json`，独立复算在 `integration/independent_verification.json`，
录像在 `integration/first_person/`。以上文件已生成并读取。

后续根据用户新要求另行构造 0.1–1.0 共视分层，不擅自扩大此 N=2 的分母：
[共视分层冻结协议](HM3D_COVISIBILITY_REPAIRED_PROTOCOL_20260909.md)。
