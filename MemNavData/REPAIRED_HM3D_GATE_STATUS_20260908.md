# 修复版 HM3D 集成验证：迁移、预检与提交收据

2026-09-08；本轮继续本机 actual mono-A 集成之后的 HPC 迁移。
这是两源运行栈验证，不是论文正式扩样；不操作真机，不修改论文，不 commit/push。

**最新结局：17188986 已正常结束（7 分 56 秒），2 A / 0 query。**
两条 A 的复算与视频完成，但没有合法 mixed-role 历史，不能宣称完整查询 gate 通过。
最终损耗见第 7 节；第 5 节保留提交当时的排队记录。

## 1. 要验证什么

从原冻结 HM3D 池固定取前两条 source，重新执行单目 native A，
再从它们真正看到的历史构造 Natural Novel / standard Revisit 查询。
每个合法历史由 mono native、mono raw memory、mono CEC 共用起点、目标、历史和模型进程。
失败 A 不替换；最多 2 个历史、2 条 A + 12 条 query rollout。

固定 source：`rJhMRvNn4DS/episode_0000`、`6D36GQHuP8H/episode_0000`；
seed：2026082200、2026082300。选择未读取旧 A 或旧 query 结局。
parent SHA：`a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5`。

和本机已通过版相同：RGB 修正、单目深度逆 padding、bounded pursuit、标准 try_step、
raw/CEC 共有后方目标朝向适配、canonical reference depth、600 tick、8-tick horizon、2.5 m residual。
构造 K 暂时保留原定义，FY 的约 1.16% 差异没有暗中改动；正式构造版本需另行冻结。

不把门控拒绝等同于 Novel 分类。角色只供 evaluator 构造和统计，不供模型读取。
依然采用理想底层状态、标准 NavMesh 碰撞和 evaluator GT 平面距离 <1 m 成功；
没有验证自主视觉 STOP、目标朝向或四足全身碰撞安全。

## 2. 本轮已完成的准备

- 本机 CPU 回归：165 passed，12 条已有 Pyparsing 弃用警告；没有隐藏 warning。
  收据：`.diagnostics/repaired_hm3d_gate_20260908_preflight_v5.xml`。
- sealed bundle 的可迁移测试另行通过：18 passed、1 deselected；唯一排除项读取工作站
  专属 MP3D source，HPC source 已由实际父 manifest 单独绑定并完成数据校验。
- 完整代码闭包：1,663 个文件、15,747,731 bytes，压缩约 3.3 MB。
  不上传权重、场景、论文、Git 凭据或整个工作区。
- 本机现有 dirty worktree 保留。生产 evaluator 唯一额外调整是把 `mkdir(out)` 放到
  `contract_dry_run` 返回之后；正式导航路径的算法未因这项调整变化。
- 原有两解释器、Singularity 和固定 checkpoint 路径复用，不新建或升级共享 conda。
- 共享连接确认是 `alantorch` / `yz11502`；已有 master 可用，PTY shell 与 SCP 均实际成功。
- scratch 约 1.38 TB / 5 TB，但文件数约 489.9 万 / 500 万。
  模型逐帧 buffer 和工作目录改放本作业 node-local 临时目录；科学收据、深度审计和视频持久保存。
  源码保存一次 immutable bundle 引用，不按每个 arm 重复复制。

## 3. 启动前抓住的环境问题

### 登录节点 `/tmp` 已满

Habitat Python 3.9 导入 quaternion/numba/llvmlite 抛出 `MemoryError`。
最小 `ctypes.CFUNCTYPE` 回调也失败；当时可用内存约 83 GiB，而 `/tmp` 2 GB、100% 占用。
仅给子进程指定本任务 `TMPDIR`/`LIBFFI_TMPDIR` 后，同一解释器导入通过。
没有清理他人临时文件，没有修改共享依赖，更不是显存不足。

### 新图像审计需要 OpenCV

旧 Habitat 环境没有 `cv2`；新栅格审计和视频入口需要它。
只复制既有 MemNav 的 OpenCV-headless 4.9.0 `cv2` 与其二进制库到任务依赖目录，
在 Habitat Python 3.9 / NumPy 1.26.4 中实际验证 abi3 导入和 resize。
不把 Python 3.10 的整个 site-packages 混入 Habitat。

依赖目录：

```
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_hab_cv2_20260908
SOURCE_DEPENDENCY.sha256: 43c80ec16d3363f516843f2897233f80a28ea18647f386ff7964e0a5a954567d
```

两项问题均在 GPU 提交前发现。失败预检日志完整保留，没有伪装成导航失败或成功。
已加入 [HPC 手册](HPC_SHARED_SSH_OPERATIONS_20260816.md)第 12–13 节。

## 4. 最终运行包

```
source bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_ee865620b4540675

SOURCE_BUNDLE.sha256:
ee865620b4540675e1c9f875361692207f5be33f7ecd373d955041f379c2d9e1

archive SHA256:
29f8f610bae6b5400f64cf30ac0b74c252f17a9426956d95c0d039913197f79b

actual-interpreter preflight:
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_gate_20260908/preflight_ee865620b4540675
```

较早 v3/v4 bundle 和失败预检仅为迁移记录，不是多次 GPU 试跑或结果选优。
最终包在远端再次运行 Habitat 七组 CLI、MemNav/NavDP import provenance、FFmpeg 和源数据检查。
调度资源固定 h100_tandon/a100_tandon、1 GPU、10 CPU、96 GB、1 小时；不自动触发正式扩样。

## 5. 提交当时的状态（历史记录）

最终包三组实际解释器预检、Habitat 七组完整 CLI、FFmpeg、两条源数据以及原基础依赖
校验均通过。`safe_sbatch --lint-fatal --test-only` 通过后才执行一次实际提交。

- Job：**17188986**，`cec_repaired_gate`。
- 提交时间：2026-09-08 14:16:38 UTC / 北京时间 22:16:38。
- 实际资源：a100_tandon/h100_tandon、1 GPU、10 CPU、96 GB、1 小时；account
  `torch_pr_769_tandon_advanced`、QOS `gpu48`。
- 22:17 后核查：**PENDING，Reason=QOSGrpGRES，StartTime=Unknown / N/A**。
  当前未分配 GPU，未运行 A/query；不是环境失败、不是低利用率取消。
  test-only 的临时估计不是正式启动承诺，不据此报预计完成时间。
- 没有自动下游正式批次。正式 HM3D 修复版 SR 尚未产生。

收据和输出：

```
remote submission:
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_gate_20260908/submission_ee865620b4540675/submission.jobs

run (created only after node prechecks):
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_gate_20260908/gate_ee865620b4540675

Slurm logs:
/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cec_repaired_gate_17188986.out
/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cec_repaired_gate_17188986.err

local submission:
MemNavData/REPAIRED_HM3D_GATE_SUBMISSION_20260908.json
```

恢复工作时先 `squeue -j 17188986` / `sacct -j 17188986`，不因 SSH 命令无回显重复提交。
启动后先查看 `preflight/*.log`、`integration/progress.json`、模型日志，再看 A/构造与 query；
若失败，保留原输出并修复确切错误，不覆盖旧试运行、不补选有利 source。

不能以服务 ready、单元测试通过或合法零历史宣布完整 query gate 通过。
完成需同时有完整 A/构造损耗、实际 query 数、独立逐动作复算、深度记录和全部成功/失败视频。

## 6. 与旧任务的区别

此前 Final14 Table-III SPL 补跑 `17089916`、`17089989_16` 与汇总 `17089991`
已从 sacct 确认 COMPLETED/0:0；它们仍属旧执行版本。
不能将其新保存的末步坐标/精确 SPL 当成此次修复版的导航确认。

本轮不是 PT1 expert 轨迹重放：只复用原 source 起点与 Goal-A 图，A 行为必须由新单目策略重新执行。
协议详见 [两源 gate](REPAIRED_HM3D_GATE_PROTOCOL_20260908.md)；
后续正式批次准备见 [核心评测计划](REPAIRED_HM3D_CORE_EVAL_PLAN_20260908.md)。

## 7. 最终结果：正常收尾，但尚未跑到 memory query

Slurm：COMPLETED / 0:0，ga043，7 分 56 秒。
北京时间 22:23:58 开始、22:31:54 结束；没有环境崩溃或超时取消。

| source | A success | steps | actual path | final distance |
|---|---:|---:|---:|---:|
| rJhMRvNn4DS | 1 | 128 | 4.771 m | 0.986 m |
| 6D36GQHuP8H | 0 | 492 | 11.074 m | 9.117 m |

独立 verifier：`verified=true`，2 条 A、620 个动作、78 次深度数组检查；
两段完整成功/失败视频（129/493 帧）导出，首帧 RGB 与原 rollout 一致。
`query_arm_count=0`、`pairs=[]`；这不是 CEC 的零成功率，而是没有运行 CEC 查询。

第一条成功 A 已有 standard Revisit（geo=3.1079 m、max-covis=0.71861）和 hard Revisit，
但没有满足原合约的 Natural Novel，故不保留成对历史。第二条因 A 失败不进入构造。
固定 5,000 次 Novel 提议按实际筛选顺序分解：

- 重复位置：10；不可导航：10；楼层不符：2,529；净空不足：1,934；
- 距离不在 2–9 m：348；不可达：1；方向不在既定 front 层：59；
- 剩余 109 个均未通过 max-covis < 0.10；最终接受 0。

上述互斥叶项合计 5,000；汇总字段 `floor_or_clearance_rejects` 等不能再叠加计算。
没有存储失败候选的完整共视曲线，不能声称每个候选的超阈幅度，也不能证明场景里绝对不存在合法 Novel。

代码与记录核对：楼层检查比较候选地面 y 与 trace endpoint 地面 y；渲染时再加 0.5 m
相机高度，历史 materialization 逐帧 RGB 哈希一致。没有发现相机高度混入楼层判定的证据。
不把可构造性不足归咎于 CEC，因为该阶段没有查询模型；不据此放松原构造阈值。

下一步固定父清单 rank 2–5 的四条 source，两个分片，全部计账；详见
[追加协议](REPAIRED_HM3D_GATE_EXTENSION_PROTOCOL_20260908.md)。这是另一次明确记录的
运行栈追加验证，不覆盖或改名此次零查询结果。
