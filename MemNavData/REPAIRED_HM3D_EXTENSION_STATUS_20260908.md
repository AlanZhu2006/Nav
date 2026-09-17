# 修复版 HM3D：四源追加与下一步

2026-09-08（北京时间）。目标是完成修复版的完整 mixed-role 三臂链，尚非正式论文扩样。

**9 月 9 日 00:49 最终核查**：原 shard 0 正常完成；原 shard 1 的 SIGABRT
已经以同代码 **17201733_1 / A100** 完整补跑，00:48:48 在 ga015 正常结束，耗时 4:59。
四条追加 A 全部成功，两分片 independent verifier 均为 true，但仍 **0 个 role-pair / 0 query**。
三个历史在合规源帧范围内距终点不足 2 m；另一个不能构造固定 front Novel。
本机 CPU 已对四条源帧和有关构造几何计数精确复算。当前无在跑任务、不自动扩样。详见
[构造与运行时审计](REPAIRED_HM3D_CONSTRUCTION_RUNTIME_AUDIT_20260909.md)。

## 1. 为什么不继续调方法

初始 17188986 已正常完成：2 A、0 query。成功 A 有合法 Revisit，但固定 Natural Novel
构造失败；不是 CEC 证书拒绝，也不是三臂 SR 已经测成 0。
全部 A 动作、深度与视频已经保留，详情见 [初始 gate 最终结果](REPAIRED_HM3D_GATE_STATUS_20260908.md)。

本机已拉回原始 JSON：`.diagnostics/repaired_hm3d_gate_20260908/final_receipts/`。
独立再核对：5000 次 Novel 提议的互斥拒绝项合计 5000；`verified=true`、A=2、query=0。
不覆盖初始 gate，不删失败 A，不把单独可构造的 Revisit 放进原 mixed-role 分母。

## 2. 本轮明确限定为四条新源任务

父清单 rank 2–5，各自 `episode_0000`，与初始 rank 0/1 不重叠：

| 分片 | 场景 | seed | Novel 方向层 |
|---|---|---:|---|
| 0 | nrA1tAA17Yp | 2026082400 | side |
| 0 | jgPBycuV1Jq | 2026082500 | front |
| 1 | BFRyYbPCCPE | 2026082600 | rear |
| 1 | X7gTkoDHViv | 2026082700 | side |

这四条在查询前一起冻结；不看旧 A/query 结局挑选，不因早期 SR 好坏缩减或追加。
各分片先完成两条 actual mono-A 和全部构造，再执行每个合法历史的六个 role-arm。
最多 4 A + 24 query；实际分母由真实 A 与构造损耗决定，不承诺四条一定能产生足够 query。

## 3. 改了什么、没改什么

只添加固定 source 分片与资源入口。分片必须保留父场景编号，避免每项从 0/1 编号
导致方向分层和构造随机流改变；arm 顺序也按父编号轮转。独立 verifier 核对该编号。
初始两源 rank 0/1 的规则不变。

对初始 HPC immutable bundle 逐文件比较，已有文件仅有六项变化：source 选择器、
打包器、supervisor 的父编号传递、运行壳、单元测试、verifier 的编号核验。
另新增追加协议和 SBATCH。NavDP/CEC/输入修复/执行器/构造算法文件均保持一致。
没有修改模型、权重、证书阈值、2.5 m、深度源、构造 K、共视/距离/方向要求或动作预算。

## 4. 预检与运行包

- 本机回归：167 passed、12 条既有 Pyparsing 弃用警告。
- relocated bundle：20 passed、1 deselected；排除项依赖本机专属 source 资产，
  远端 source 另按父 manifest 校验，不意味着跳过数据验证。
- Habitat Python 3.9 语法检查：984 个 MemNavData Python 文件通过。
- 原先已经在 HPC 实际跑通的 Singularity、MemNav/Habitat 解释器、OpenCV 独立依赖、
  node-local 临时目录与端口锁继续复用；不重新安装共享环境。
- 最新配额：scratch 1.38 TB / 5 TB，4,905,361 / 5,000,000 files（98%）。
  仍把模型逐帧 buffer 留在节点临时盘，只保留科学输出；不删除其他任务数据。

```
source bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f

SOURCE_BUNDLE.sha256:
c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08

archive SHA256:
2801051c52755a94f761d7db3e9302dea3c15ae33572a9594c631f6ca905de02

remote CPU preflight:
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_extension_20260908/preflight_c8cf8c60e7efd55f
```

## 5. 提交状态

远端三组实际解释器导入、七种 CLI、ffmpeg 与四条源数据校验均通过；
`safe_sbatch --lint-fatal --test-only` 通过后，于北京时间 **2026-09-08 23:59:25**
提交 array **17196996**（0–1%2）。每项 1 GPU、10 CPU、96 GB、1 小时，
h100_tandon/a100_tandon；无自动下游任务，不启动正式全量批次。

23:59:35 核对：两项均 **PENDING / QOSGrpGRES**，尚未分配 GPU，实际开始时间 Unknown。
这是提交时快照，不代表之后仍在排队；test-only 的预测时间不是实际启动承诺。
这份提交快照当时尚无追加 A/query 结果；后续状态见本页顶部和下方第 7 节。机器可读收据：
[REPAIRED_HM3D_EXTENSION_SUBMISSION_20260908.json](REPAIRED_HM3D_EXTENSION_SUBMISSION_20260908.json)。

运行根目录约定：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_extension_20260908/extension_c8cf8c60e7efd55f
  shard_0/integration/
  shard_1/integration/
```

Slurm 日志：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cec_repaired_ext_17196996_0.out
/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cec_repaired_ext_17196996_0.err
/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cec_repaired_ext_17196996_1.out
/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cec_repaired_ext_17196996_1.err
```

## 6. 完成后如何判断

先核对两个分片所有 A、构造损耗和 query 是否齐全，再看 independent verifier 和全量视频。
至少一个合法历史完整执行 native/raw/CEC × Novel/Revisit，才表示 HPC 查询链实际跑通。
之后报告每个角色的 SR、配对增减、实际路程/动作及耗时；空分片必须同时报告。

即使有增益，N≤4 也不替换论文主表；正式补跑范围取决于完整接口验证，而不是挑最好结果。
若仍零历史，先报告具体构造原因再重新设计，不自动无限追加源任务。

冻结协议：[REPAIRED_HM3D_GATE_EXTENSION_PROTOCOL_20260908.md](REPAIRED_HM3D_GATE_EXTENSION_PROTOCOL_20260908.md)。

## 7. 9 月 9 日复核与仅失败分片补跑

原 shard 0：COMPLETED / gh003 / 6:45，A=2/2 成功，query=0，verifier=true。
nrA 的合规源帧距终点最多 1.104 m，不足 2 m；jgP 能构造 Revisit，
但空间有效 Novel 提议 558 个 rear、1 个 front，而其冻结层为 front，唯一剩余提议
被原共视检查拒绝。本机不渲染的 CPU 复算逐项匹配原源帧和几何计数。

原 shard 1：FAILED / gh008 / 19:58，首条 A 56 步后 evaluator SIGABRT；
没有终局收据，不计导航失败。原失败 23 文件完整保留。

新任务 **17201733_1** 只补原 rank 4/5，两源、相同 bundle、相同配置与构造，
限定 A100；只增加 faulthandler 堆栈记录。1 GPU / 10 CPU / 96 GB / 1 小时，
没有自动扩样或正式下游任务。

该补跑已经 COMPLETED：BFRyYbPCCPE 65 步、28.10 s 成功；X7gTkoDHViv 83 步、
26.46 s 成功；独立复算和两段完整视频均完成。两条的源帧到终点最大距离分别
0.976 / 1.348 m，不满足冻结 ≥2 m；没有 query，不能报告 CEC SR。

- [补跑协议](REPAIRED_HM3D_A100_RETRY_PROTOCOL_20260909.md)
- [补跑提交收据](REPAIRED_HM3D_A100_RETRY_SUBMISSION_20260909.json)
- [最终四源结果绑定](REPAIRED_HM3D_EXTENSION_RESULT_20260909.json)

```
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_extension_20260908/a100_retry1_20260909/shard_1/integration
```
