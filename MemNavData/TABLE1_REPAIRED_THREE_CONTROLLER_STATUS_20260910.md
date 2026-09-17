# Table I 修复补跑 + NoMaD：执行记录

更新于2026-09-10 13:11，北京时间。**本机12/12完成并独立验证，HPC已实际提交。**
按 `TABLE1_REPAIRED_THREE_CONTROLLER_PROTOCOL_20260910.md` 执行。
本记录中的小测不能替换论文表格，正式新SR尚未产生。

## 设计

| 数据集 | 复用的实际在线 A 历史 | 场景 | 每个 controller 的 Novel / Revisit 查询 | 三 controller × 两臂 |
|---|---:|---:|---:|---:|
| HM3D | 28 | 21 | 28 / 28 | 336 rollouts |
| MP3D | 42 | 25 | 42 / 42 | 504 rollouts |

NavDP、ViNT、NoMaD 各自比较 native 与 GEM。总计 210 个四-rollout 配对任务、840 rollouts。
600 ticks，实际位移 SPL，原始目标 1 m 平面距离计分，角色不进入策略。
本次修复 query 执行/输入接口，不重新采 A；不是三个 controller 各自自主完成 A 的完整系统比较。

NavDP 在 GEM 接受时使用原始 ImageGoal + 认证方向 PointGoal；ViNT/NoMaD 在接受时
使用认证历史 anchor 图，认证方向仅用于公共物理朝向适配。拒绝后仍是同一个 controller，
ViNT/NoMaD 不接收 depth/PointGoal、不退回 NavDP。GEM 使用 strict 原条件，
不吸收正在运行的去 coverage 消融修改，不调阈值。

## 接入 NoMaD 前发现的实际问题

1. 旧 NoMaD `step_imagegoal` 的 mask=1 会屏蔽目标。新入口使用官方导航语义的 mask=0。
   同一真实权重 CPU 检查中，换目标图：mask=1 的条件特征最大变化为 0，mask=0 为 6.75455。
2. 原 RGB server 的 BGR 交换与 PIL/RGB 模型输入不符。新 server 对当前图、目标图、回放统一用 RGB。
3. 旧两个 baseline 包装器附加 `predicted_distance > 7` 轨迹清零。
   新入口记录 distance，但使用模型原始动作；没有搜寻替代阈值。
4. NoMaD 每次采样真正消费 paired seed；固定 8 samples、10 diffusion steps、sample0。
5. 两个 RGB 模型首帧补齐短期 FIFO；其回放收据和检查应反映真实 padding，不能套用 NavDP 的队列长度规则。

因此后续新旧表格差值不能单独归因于执行器修复或 GEM；必须报告这些 controller 接口修正。
这是使用发布权重的局部 ImageGoal policy 对照，不是官方完整 topomap 导航系统复现。

## 本机验证

根目录：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/table1_repaired_20260910_Bu3m1J
```

- `nomad_cpu_v2/verification.json`、`vint_cpu_v2/verification.json`：真实权重、HTTP RGB、reset/replay、seed及首帧padding检查均通过。
- `local_plan.json`：开跑前固定同一已消费 history，三 controller、12 rollouts。
- `local_nomad_v1`：回放检查缺少实际队列长度收据，在导航前停止；修复收据与 padded-context 检查后完整重启。
- `local_nomad_v2`：四个rollout全部完成并通过独立 verifier。
  Novel：native/GEM 均0/1且所有候选与物理轨迹精确一致；Revisit：native 0/1、GEM 1/1。
  GEM Revisit 142 ticks，实际路径3.10747m，59个物理转身ticks，按原始目标评分。
  这是N=1 history的接口验证，不证明NoMaD/GEM的泛化SR。
- `local_vint_v1`：尝试与 NoMaD 并行时在 LingBot 历史回放阶段显存不足，未产生有效导航结果。
  不能记作 ViNT SR=0。其私有进程已经退出；后续串行完整重跑，不改变模型或预算。
- `local_vint_v2` 与 `local_navdp_v1` 串行完成，各4个rollout，独立verifier全部通过。
  两个controller的Novel均0/1对0/1且完整精确回退，Revisit均native 0/1、GEM 1/1。
  GEM成功分别为175/113 ticks、实际路径3.25867/2.59860m。按该任务SPL定义三者成功臂均为1.0。
  这些来自同一个history，不能当作三个独立场景的验证或方法排序。
- `local_gate.json`：三controller共12个rollout、源码与实际HPC bundle一致，`verified=true`。
- 本轮契约测试65 passed，相关既有回归测试18 passed；shell syntax 与 `git diff --check` 通过。

显存经验：单个 LingBot 流的常态约15 GiB不代表峰值。本轮 GEM 查询峰值约23 GiB；
并行第二个流加上已有真机服务会超过本卡容量。HPC 每 GPU 只运行一个 controller-history cell。

## HPC 准备

使用现有共享 `alantorch` SSH，身份 `yz11502`，未新建认证流程或切换账户。
容器、环境、port flock、A100/account/QOS/time-limit 按已有 HPC 手册。

准备 bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table1_repaired_7b6bffa77891ef4f
SOURCE_BUNDLE.sha256 的 SHA256:
7b6bffa77891ef4f041508dc7dc31b40fb56b7e1885334a9e180a1f06d76e099
```

准备输出根：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/run_7b6bffa77891ef4f
```

正式提交须先有三 controller 完整本机验证和 exact-container preflight。
远端exact-container预检已通过：70 histories，15,549个实际输入文件、24组CLI，
ViNT/NoMaD实际权重CPU检查均成功；最长A历史564帧，均值218.47帧。
完整收据为输出根的 `preflight/verification.json`，本机副本 `hpc_preflight_v1.json`。
Slurm GPU/CPU模板 `--test-only` 均通过；试提交打印的编号不是真正评测作业号。
GPU gate 索引固定为 0,28,56,84,126,168，覆盖三 controller × 两 dataset；
其余204 cells依赖六项执行/验证成功，不以导航成功或GEM增益作为放行条件。
每项 A100 / 1 GPU / 10 CPU / 96 GB / 1小时，最多4并发；全部逐帧材料 node-local 写入，
结束压缩并逐成员hash回读后持久化。失败项不得混入已完成分母。

## 代码入口

- `image_controller_policy.py` / `image_controller_repaired_server.py`：真实 RGB policy 与私有 HTTP 服务。
- `image_controller_goal_adapter.py`：原目标/认证历史图接口，复用现有 GEM 判定和朝向执行。
- `table1_repaired_eval.py`：冻结总体、同服务配对运行。
- `verify_table1_repaired.py`：独立 SR/SPL、实际动作、目标输入和 exact native 检查。
- `preflight_table1_repaired.py`：全部原始历史/图像/场景、24组 CLI 和远端实际权重 CPU 检查。
- `seal_table1_local_gate.py`：将本机三项完整检查与实际 HPC 源码绑定。
- `submit_table1_repaired_hpc.sh`：六项集成、204项正式数组和最终汇总。
- `summarize_table1_repaired.py`：六组 role-separated SR/SPL、配对增损、McNemar、scene bootstrap、Holm。

本轮未改论文、默认 GEM、真机服务、已有 coverage 作业，也未 commit/push。

## 实际提交（不是 test-only 编号）

2026-09-10 13:10:11，北京时间，提交脚本成功，Slurm确认以下作业及依赖：

| 作业 | ID | 范围 | 依赖 |
|---|---|---|---|
| 首批完整集成 | 17306819 | 6 cells / 24 rollouts，三controller×两dataset | 无 |
| 其余正式配对 | 17306820 | 204 cells / 816 rollouts | afterok:17306819 |
| 完整统计汇总 | 17306821 | 六组SR/SPL、损益及统计 | afterany:17306819:17306820 |

首批六项包含在210项总体内，不重复计数。13:11的队列状态：首批PENDING/QOSGrpGRES，
后两项PENDING/Dependency。尚未分配GPU，不能写成环境启动失败、评测完成或已有新SR。
GPU作业明确为a100_tandon / gpu48 / torch_pr_769_tandon_advanced / 1 GPU / 1小时，
最多4并发；汇总为cpu_short / 30分钟。正式plan SHA256：

```text
0964f52738e22f2d16b099196605ef2e9681f57afce20fbdb99b417d8b535622
```

远端 `submission.json` 在上述输出根内；本机副本为结果根的 `hpc_submission.json`。
全部结束后读 `evaluation/paired_summary.json`；单项的 `archive_receipt.json`、
`independent_verification.json` 与完整压缩原始证据在 `evaluation/task_NNN/`。
缺项或基础设施失败会列入missing清单，不把缺项计作SR=0或从分母静默删除。

本机本轮私有服务均已退出；既有真机18888/8888服务保留。
