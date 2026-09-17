# Table II 连续三目标：最新冻结与提交

更新：2026-09-12（北京时间）。**首批 12 项已全部完成并核验通过，剩余 85 项
按同一冻结版本提交，已启动 2 项、83 项等待并发空位。全体 97 项已经进入评测链，
没有重跑或替换首批。**

## 1. 本轮实际完成什么

从原先已声明的 18 个 MP3D 场景、144 个来源中，保留全部 97 个有合法前向 Novel-A
的来源；47 个构造失败保留原因，不按旧导航成绩筛选，也没有修改采样预算。

| 项目 | 数量 |
|---|---:|
| 合法来源／场景 | 97／18 |
| NNN／NNR／NRN／NRR | 25／24／24／24 |
| A 距离 2–4／4–6／6–9 m | 33／32／32 |
| 首批已完成来源／场景 | 12／12 |
| 首批每种序列 | 3 |
| 后续已提交来源／场景 | 85／18 |
| 暂未提交 | 0 |

每个来源执行 native/GEM 两条自身连续链。97 个来源对应最多 194 条三目标链、582 次
leg 导航；首批最多 24 条链、72 次 leg 导航。前段失败或后继任务无法构造时，实际次数
会少于上限；这不是事后补齐固定数量的成功 A/B/C。

## 2. 与旧 Table II 最大的区别

旧任务是共享前缀／逐阶段配对。新任务保持各方法自己的实际物理状态和因果记忆，
从 A 连续接到 B、C，整个 episode 只 reset 一次，不重放前两段。

- 两臂存活时，下一张目标图相同，但必须分别满足双方实际历史的构造条件。
- Novel A/B/C 均限制初始方向 ±60°；Revisit 允许回头。
- B/C 优先使用 A 的距离档；若该档无合规候选，按提前固定顺序选其他合法距离档。
  会记录实际距离及档位变化，不保证三个阶段 SR 或难度相等。
- 一臂失败仍保留在起始分母，另一臂可以继续；没有合规共同目标则单列构造终止。
- 这是**共同在线发题的连续实验**，不能描述成预先固定三张目标图的标准 benchmark。

主结果检查完成阶梯 `N → N_A → N_AB → N_ABC`。对未发出的任务保留未知及上下界，
不把构造损耗、环境异常填成导航失败。

## 3. Controller 和 GEM 是否变化

没有变化。正式包基于已通过 A100 双活测试的 `150243d7f415a423` 运行时，1794 个原文件
保持相同字节。只修改连续任务调度、总体入口、记录及 verifier，新增提交与汇总脚本。

保持：修复版 bounded-standard 执行与真实朝向对齐、mono height/first40、horizon 8、
每 leg 600 ticks、1 m 成功判据、2.5 m residual、strict certificate，**面积条件仍保留**。
没有新增模型训练、NavMesh 控制权限、角色标签输入、expert 记忆或 runtime 参数调整。

## 4. 提交前验证

- 工作区测试：259 passed。
- 封存代码复测：259 passed；其中旧两来源单测显式绑定已有本机 fixture，正式任务不读取它。
- 全部 12 项正式 CLI／24 个 arm 命令通过，无模型加载。
- HPC 同一 Singularity、原有两个 conda 解释器的 Habitat/MemNav/NavDP 导入和正式 CLI
  预检通过，exit 0；没有重装或更新共享依赖。
- 登录节点第一次预检因 `/tmp` 2 GB 已满导致 llvmlite callback `MemoryError`；按 HPC
  手册 §12 改用任务专属 scratch 预检目录后通过，失败日志保留。这不是导航失败。
- Slurm `--test-only` 通过。小源码包使用原共享 master 的 localhost-only 反向转发传送，
  SHA 一致，临时 HTTP 和转发已关闭。

## 5. HPC 当前任务

| 任务 | Job ID | 配置 | 提交后状态 |
|---|---|---|---|
| 首批连续配对 | `17378467_[0-11%2]` | A100；1 GPU、12 CPU、128 GB；每项 1 h；并发 2 | 12/12 COMPLETED，exit 0 |
| 首批完整性与完成阶梯汇总 | `17378468` | cpu_short；15 min；afterany 依赖 | COMPLETED，exit 0 |
| 后续连续配对 | `17390278_[12-96%2]` | 同一 A100 配置、1 h、并发 2 | 2 RUNNING，83 PENDING |
| 完整 97 来源汇总 | `17390279` | cpu_short；15 min；afterany:17390278 | PENDING / Dependency |

后续索引 12、13 已在 ga025 启动，其余等待 `JobArrayTaskLimit`，尚无后续完整导航结果。
首批运行时间为 3:46–10:11；每项 1 h 上限保持不变。

远端结果根目录：

`/scratch/yz11502/Research/Nav-axis-uturn-results/table2_continuous_formal_20260912_v1`

首批汇总输出：`first12_summary_17378468.json`。每项在 `tasks/000` 至 `tasks/011`，
完整原始证据封装为 `artifacts.tar.gz`，并保留 `archive_receipt.json`、独立核验和 summary。
GPU 上仍用节点本地临时盘，结束后逐成员校验并归档。

后续输出为 `tasks/012` 至 `tasks/096`；完整汇总为 `all97_summary_17390279.json`。
新 CPU launcher 只将原冻结汇总器的 `--count` 改为 97，不改任何推理代码或统计实现。

## 6. 后续放行

首批两臂状态连续性、单次 reset、无前缀重放、尺度收据、终点及 SPL、归档均已核验。
源码包全文 SHA 复查通过，剩余 85 的输出目录不存在，没有重复提交来源。
用户确认后于 04:14 放行其余冻结索引 `12–96`。
**没有设置成功率通过门，也没有因某个来源失败而替换来源。**

如果发现需要修改 runtime 或协议，保留原首批并记录版本差异，不能静默混入同一结果。
本轮未修改论文、真机代码，也没有执行 Git commit/push。

协议：`TABLE2_CONTINUOUS_POPULATION_PROTOCOL_20260912.md`。
机器可读提交与哈希：`TABLE2_CONTINUOUS_FORMAL_SUBMISSION_20260912.json`。
本机封存总体：`.diagnostics/table2_continuous_formal_20260912_v1/bundle/continuous_population/population.json`。

## 7. 首批完整导航结果与尚未覆盖项

| 实际执行阶段 | Native | GEM |
|---|---:|---:|
| Novel-A | 6/12 | 6/12 |
| Novel-B | 1/2 | 1/2 |
| Revisit-B | 0/2 | 2/2 |
| Revisit-C | 0/1 | 3/3 |
| Novel-C | 未执行到 | 未执行到 |

C 的两臂条件分母不同，因为 GEM 在两个 Revisit-B 上成功后能继续 C，Native 已经终止。
不能将 `0/1 → 3/3` 直接描述为同分母 C 配对提升。
实际观察到 GEM 完整完成 3 条链、Native 为 0；另外 2 个来源因无合规 B 而构造终止，
不是已观察到的导航失败。对全部 12 的三段完成率，Native 的已知上下界为 `0–2/12`，
GEM 为 `3–5/12`；不能将未知项直接填零。

索引 4：A 成功后没有合规前向 Novel-B。
索引 6：A 成功但仅观察 28 帧，不能满足原 Revisit-B 历史帧资格。
这些来源保留在首批，不补样、不改资格条件；其余 85 继续使用原规则。

实测共包含 24 次 A、8 次 B、4 次 C rollout。首批证明了已执行 NNR/NRR 的连续状态
闭环，但尚未覆盖实际 Novel-C；后续需如实报告各序列的执行覆盖与构造损耗，不预测结果。

后续提交收据：`TABLE2_CONTINUOUS_REMAINDER_SUBMISSION_20260912.json`。
本次仅新增全量 CPU 汇总 launcher（24 项相关测试、Shell/Slurm 预检通过），没有改论文、
真机、controller、GEM 阈值或源总体，也没有 commit/push。
