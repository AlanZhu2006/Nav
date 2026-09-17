# Table II：共同在线发题、各自连续执行进度

2026-09-11。用户已确认采用逐段共同发题，不再要求出发前固定完整三图。

## 当前完成了什么

- **共同采样器已实现。**每个候选分别检查全部存活臂的真实历史；Novel 对双方均须前向，
  Revisit 对双方分别具备合规的历史视角来源。发送给两臂的是同一张图、同一物理目标。
- **连续协调器已实现。**每臂只初始化一次，保留自己的物理末态、FIFO、LingBot 状态、
  first40 尺度和图像历史。段间暂停不产生新动作或重复观测，不重放参考轨迹。
- **失败保留。**该臂结束后后段未尝试，另一臂继续；不借用成功历史、不补换原分母。
  构造为空与导航失败分开记录。结束的 episode 可以释放服务，存活臂不能清 state。
- 运行时仍为原修复版 mono NavDP/GEM、strict certificate、2.5 m residual，600 ticks/leg。
  没有增加 Novel/Revisit 分类器、改面积条件或改 controller。

## 本机检查与实际结果

相关回归测试 **253 passed**；portable `--source-file` 入口的两臂 CLI dry-run 通过。

1. 原真实 A 历史上，共同 Novel 在 4–6 m、6–9 m 有合规菜单，共同 Revisit 在 2–4 m、
   4–6 m 有合规菜单。这里是候选供给，不是导航成功数。
2. 对不同长度的实际 A（152 帧）和 A+B（244 帧）做构造器交换标签测试：
   两次各 37 个视觉候选检查、1 个最终距离档，结果完全一致。这里只验证采样对称性，
   两份不同阶段历史不是一个新的同阶段导航比较。
3. 双臂持续服务首测：四个服务均启动，native A 成功。保持其状态后，GEM A 在深度头
   发生显存不足，配对中止；**没有完整双臂 SR**。已保存所有 partial 和错误日志。

本机 allocator 收据显示，native A 之后释放了约 9.46 GB 的闲置块，但 14.34 GB 的
live allocation 必须保留。OOM 时另一臂已占约 16.37 GiB，另有不属于本任务的常驻进程。
没有关闭其他工作区，也没有通过清 KV、重新播放 A 或缩减图像预算绕过限制。

先前单臂开发例中的 GEM NRR、NRN 三段成功仍然有效，但不能冒充本次共同目标配对结果。

## HPC 技术首例

已提交原 source index 0、NRR 一个技术首例：A100 80 GB、1 GPU、12 CPU、128 GB RAM、
1 小时；没有提交正式数组，也不按结果换来源。原共享 `alantorch` / `yz11502` 会话正常。

**23:39 CST 实时核验：作业 `17374795` 已在 `a100_tandon / ga016` 运行。**
23:38:30 CST 提交，23:38:42 CST 获分配。源码与单一来源的哈希、原容器中的
Habitat/MemNav/NavDP 导入、七组已有 CLI 组合和新共同任务模块全部预检通过。
`sbatch --test-only` 通过后只实际提交了一次。运行初期未见 stderr 错误，尚无完整导航 SR。

| 项目 | 路径或身份 |
|---|---|
| 源码 receipt SHA | `150243d7f415a4230ea3a1268d79dc9fef2216a7d6341e958a596cef34cee991` |
| source JSON SHA | `6c0e3cc23edf0e953068fb3589ff52edb7f3d20685d5844639f6b68f0d9e8f59` |
| 实际作业 | `17374795`，`a100_tandon`，`ga016`，`01:00:00` |
| 远端源码 | `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table2_common_150243d7f415a423` |
| 远端输出 | `/scratch/yz11502/Research/Nav-axis-uturn-results/table2_continuous_common_gate_20260911_v1` |
| 提交收据 | 本机 `MemNavData/TABLE2_COMMON_GATE_SUBMISSION_20260911.json`；远端输出下 `submission.json` |
| 调度日志 | `/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/table2_common_gate_17374795.out`（同名 `.err`） |
| 本机 partial | `.diagnostics/table2_continuous_common_local_20260911_v1/` |
| 构造对称性 | `.diagnostics/table2_common_unequal_history_symmetry_20260911_v1/verification.json` |

首例验收内容：共同目标图一致、no-takeover A 的跨臂轨迹一致、各臂历史与尺度连续、
失败后的未尝试状态、实际末步路径/SPL、完整无损归档。验收不要求导航成功。
首例通过后再确定正式总体的距离分配、四种序列配额和构造损耗统计；当前尚未完成这些。

本次不是以共享原生前缀代替连续多目标：两臂各自从空历史出发，逐段保留自身状态。
在线发题器只从仍存活臂的已发生历史构造共同目标；某臂导航失败后不重新加入，
后续仅由存活臂继续。最终需分别报告 `N → N_A → N_AB → N_ABC` 与构造终止数量。
三段 Novel 初始前向与原距离范围保持统一，但不据此承诺三段条件 SR 相同。

无论文修改、无 commit/push、无机器人操作。旧结果、旧协议与本机失败归档均保留。
