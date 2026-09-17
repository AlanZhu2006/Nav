# 修复版 HM3D：构造与运行时问题复核

2026-09-09（北京时间）。本轮只补失败分片、做本机 CPU 构造审计，
不改模型、导航参数或已冻结的查询条件；不涉及论文或真机修改。

**00:49 最终核查**：A100 补跑已 COMPLETED / 0:0，总耗时 4:59；
两条 A 均成功，独立 verifier=true、两段完整视频导出。四源追加最终 A=4/4、query=0。
三个短历史没有合规的 ≥2 m Revisit 源帧，另一个历史未通过固定 front Novel 构造。
当前不再有运行作业，没有自动扩样，完整 mixed-role query gate 仍未通过。

## 1. 原四源追加的实际终态

| 分片 | Slurm 终态 | A / query | 解释 |
|---|---|---|---|
| 17196996_0 | COMPLETED，gh003，6:45 | A 2/2 成功，query 0 | 独立 verifier=true，全量视频已导出；没有完整 role-pair |
| 17196996_1 | FAILED，gh008，19:58，ExitCode 1:0 | 第一 A 中断，第二 A 未开始 | evaluator SIGABRT；不能将两条计成导航失败 |

零查询不等于 CEC SR=0。也不能用两条完成 A 的 2/2 来报告四源整体 A SR。
初始两源 17188986 的独立结果仍保留，不用本次输出覆盖。

## 2. 构造问题已在本机复现，不是推测

下载已有 A trace、实际 execution.navmesh 与原 construction summary。
只加载 Habitat PathFinder（CPU），不加载模型、不渲染、不申请 GPU。
重新调用封存构造函数的源帧选择和几何筛选；原始候选编号、距离及所有拒绝计数一致。

### 2.1 nrA1tAA17Yp：成功 A 太短，合规源帧已靠近终点

- A 初始测地线 3.135 m，实际走 2.380 m，75 步到达。
- 冻结的源帧范围从第 39 帧开始，排除尾部 16 帧，每 8 帧采样。
- 实际检查帧为 39、47、55；到 A 终点的测地线分别为 **1.104、0.986、0.763 m**。
- 均不满足 Revisit 源帧距终点 **2–9 m** 的条件。
- 整条历史只有帧 0–3 距终点达到 2 m，但它们在允许源帧范围之外。

因此没有进入目标重渲染或证书阶段，不能称为 DINO、几何 matcher 或 CEC 失败。
“A 成功”与“成功历史足以构造本协议”确实不同；本例没有必要再尝试更多位姿候选。

### 2.2 jgPBycuV1Jq：有效 Novel 空间几乎全在冻结方向层之外

- A 实际走 9.797 m、306 步成功。
- 可构造 standard Revisit：距 A 终点 2.325 m、历史最大共视 0.7197。
- Natural Novel 固定为 front；原 5,000 个提议全部未通过。

| 互斥拒绝原因 | 数量 |
|---|---:|
| 重复位置 | 7 |
| 不可导航 | 1 |
| 楼层不符 | 2,201 |
| 净空不足 | 1,791 |
| 不可达 | 280 |
| 距离不在 2–9 m | 161 |
| 方向不在 front | 558 |
| 最终共视拒绝 | 1 |
| 合计 | 5,000 |

本机完整复现该计数；在方向筛选前剩下的 **559** 个空间有效提议中，
**558 个属于 rear，1 个属于 front，0 个属于 side**。
因此只有 1 个进入实际视觉共视检查；它被原运行判为有历史支持，不能构成 Novel。
不是“5000 个有效 Novel 都被 CEC 拒绝”，也不能据此宣称场景没有任何合法 Novel。

审计脚本没有重新计算共视：为了完整重放一个已失败搜索的几何随机流，
仅在本机诊断进程将最后视觉阶段替换为必拒绝标记；这个标记不是测量值、不是新实验。
输出明确 `visual_support_recomputed=false`。几何计数和源帧选择均与原收据逐项比较。
不改原构造文件，也没有临时把 front 改为 rear 来填补查询分母。

## 3. 运行时已知什么、未知什么

原 shard 1 留下 56 个完整动作、7 次规划、56 个 memory_step 完成记录。
连续 memory HTTP 完成时间间隔中位数 19 秒，但这不是服务端 forward 的分项计时。
NavDP 稳态规划日志约 0.16–0.24 秒，不能说 NavDP 每次生成轨迹需要 19 秒。

evaluator 最后为 SIGABRT；Slurm 不是 TIMEOUT 或 OUT_OF_MEMORY，MaxRSS 约 16 GB，
低于申请的 96 GB。原日志没有 native 堆栈，不能确定触发 abort 的具体函数。
渲染深度转 uint16 出现非有限值警告，但已保存的 **7/7 NavDP 深度产物均为有限数**；
不能将这条警告直接等同于“单目深度输入 NaN 导致崩溃”。

历史原始 Slurm 记账复核：17089916 九项、17089989_16 均在 A100 完成，
17089991 后处理完成。这支持对相似运行故障采用同代码 A100 补跑，
而不是立即更改模型或 CUDA 同步开关；它不是本次故障的因果证明。

我们此前复用了通用 H100/A100 分区列表，没有将这条已验证的运行规避记录
落实为本次失败重试的明确 A100 限定。导入、CLI 和短渲染预检不能覆盖这种连续运行故障。

## 4. 已执行的最小补跑

**17201733_1**：北京时间 00:35:24 提交，唯一索引 1，原两条 source，
原 sealed bundle `c8cf8c60e7efd55f`，A100 / 1 GPU / 10 CPU / 96 GB / 1 小时。
不重跑已完成 shard 0。添加 `PYTHONFAULTHANDLER=1` 记录异常堆栈，不改 CUDA 同步。
原失败树 23 个文件、3,324,063 bytes 完整保留；补跑在独立输出目录。

冻结说明：[A100 重试协议](REPAIRED_HM3D_A100_RETRY_PROTOCOL_20260909.md)。
状态和提交绑定见 [四源最新状态](REPAIRED_HM3D_EXTENSION_STATUS_20260908.md)。

### 4.1 补跑最终结果（已拉回原始 JSON 复核）

00:43:49 开始，00:48:48 完成，ga015 / A100 / ExitCode 0:0。

| A source | success | steps | 实际路程 | 含审计的单条耗时 | 合规源帧距终点最大值 |
|---|---:|---:|---:|---:|---:|
| BFRyYbPCCPE | 1 | 65 | 2.441 m | 28.10 s | 0.976 m |
| X7gTkoDHViv | 1 | 83 | 2.818 m | 26.46 s | 1.348 m |

这两条均独立核验完成，2 视频已导出；源帧筛选已在本机 CPU 精确重放。
没有修改 frame39、2–9 m 或其他条件，因此都没有进入 Revisit 渲染，query 仍为 0。
原 H100 的未完成 A 没有并入成功率分母或用 partial 拼接。

148 个 memory HTTP 完成记录的相邻时间差中位数为 0 秒（日志只有整秒精度），
最大 8 秒，包含两条 A 之间 reset/切换；已不再出现持续 19 秒间隔。
这是运行规避在本次完整两条 A 上奏效，不是对 GPU/驱动根因的普遍证明。

## 5. 本机测试与可复核位置

- CPU 真实轨迹审计：两条 source-frame 列表、jgP 的 5000 次几何拒绝计数完全复现。
- 补跑后另外两条 A 也完成本机 CPU 源帧重放；四源全部覆盖。
- 新审计单元测试 3 项通过；现有 integration 21 项回归通过。
- Habitat 环境没有 pytest，新测试使用标准库 unittest；没有安装或混入另一解释器的包。
- 只新增离线审计与提交工具；没有改变正在运行的封存源码。

```
.diagnostics/repaired_hm3d_extension_20260908/audit_20260909/
  original_minimal_evidence.tar.gz
  extracted/shard_0/integration/
  extracted/shard_1/integration/
  constructibility_cpu.json
  a100_final_minimal_evidence.tar.gz
  a100_final/
  a100_constructibility_cpu.json
```

下载归档 SHA256：`f3eeac141922d4821b5930fa89da2d87c058d98c268e76f62bf1dd44c802dd1c`。
补跑最终证据归档 SHA256：`0953c5745adce6eb8264403266ed66176439b28ef9e836acabadb967dcb2b7d4`。

复算命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=MemNavData \
  /home/asus/miniconda3/envs/habitat/bin/python \
  MemNavData/audit_repaired_hm3d_constructibility.py \
  .diagnostics/repaired_hm3d_extension_20260908/audit_20260909/extracted/shard_0/integration \
  --out NEW_NONEXISTENT_RESULT.json
```

## 6. 下一步边界

唯一失败分片现已补齐，查询仍为零，已停止自动扩样。
不新增 source、不改变原 mixed-role population。
后续应明确区分“已有合法 Revisit 的运行链验证”与“带固定方向层的 mixed-role 统计评测”。
前者可单独设计一个小验证，不能计入原 mixed-role 分母；后者若要调整 source 资格或
方向构造，必须另外冻结协议，不可追改当前结果。本文不启动这些后续实验。
