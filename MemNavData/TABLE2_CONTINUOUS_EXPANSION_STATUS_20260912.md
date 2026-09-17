# Table II 固定扩样：提交与衔接状态

2026-09-12 12:06（北京时间更新）。用户授权设计进一步扩样并提交 HPC。

## 已提交的链

| 阶段 | Job ID | 资源／依赖 | 提交后状态 |
|---|---|---|---|
| 全部新来源的 A 目标菜单 | 17432300_[0-17%4] | A100；每项 32 来源；1 GPU、6 CPU、24 GB、1 h | 2 RUNNING（ga033/ga012），16 PENDING（QOSGrpGRES） |
| 冻结合法总体并自动提交导航 | 17432301 | cpu_short；2 CPU、16 GB、30 min；afterok:17432300 | PENDING，Dependency |
| 真正连续导航 | 尚未生成 job ID | 上一步成功后自动提交；A100、1 GPU、12 CPU、128 GB、1 h、并发 4 | 未开始 |
| 完整汇总 | 尚未生成 job ID | 自动依赖导航数组 afterany；读取逐任务独立 verifier | 未开始 |

A 菜单构造只生成目标图和离线资格信息，不是 NavDP 的 A rollout，也不是 expert 记忆。
两个已启动分片正在持续构造目标；日志没有出现新依赖报错。
后续每个合法来源仍真实从空历史执行 A→B→C；两方法各自持有连续历史。
因此当前没有新 SR，不把 576 个声明来源写成 576 个已经有效／成功的 episode。

## 扩样设计

- 18 个原场景 × 32 个新 seed／初始朝向 = 576 个新声明案例。
- 物理起始位置仍均匀复用原八个载体；不是 576 个新物理位置，不是新场景泛化确认。
- 新 episode_0008–0039 和 seed 2026120000 起，与旧 144/97 任务身份不重叠。
- 不筛旧成功 A，不重放 A/B，不读取 expert 轨迹作为运行时记忆。
- Novel A/B/C 前向 ±60°、2–9 m 和完整历史无支持条件保持；Revisit 允许后向。
- 唯一构造修订：Novel 每距离档候选预算 12→48，三个阶段一致；Revisit 预算与资格不变。
- 控制器、权重、mono height/first40、2.5 m residual、每 leg 600 ticks、horizon 8 不变。
- 沿用旧连续实验 **含 5% hull coverage 的 strict GEM**；不是主文无面积版本复测。

目的是增加各方法各类 C 的实际发题数，覆盖目标为至少 10，但不保证达到，不将其当充分统计功效。
只运行固定总体，不按中途 SR 无限加样；即使样本不够也必须保留所有失败和构造未知。
因候选预算已修订，本轮单独报告，不与旧 97 静默合并。

## 已完成预检

- 封装内 218 项单测通过；原 1805 个文件不变，唯一原代码差异为离线构造常量 12→48。
- 三个新来源执行本机真实 Habitat 目标构造：两例生成三个合法距离档，一例无合法 A；均正确记录。
- 一个已用历史的 C 共同发题接口检查通过，生成 2–4/4–6 m 合规目标；只作接口诊断，不进入正式数据。
- 远端原 Singularity 与原 Habitat/MemNav/NavDP 解释器导入及 CLI 全部通过；未安装／更新 conda。
- 18 个真实 GLB 哈希通过；源码包传输 SHA 一致；Slurm test-only 和实际提交参数回读通过。
- 仍按手册使用 yz11502 的共享 alantorch。SCP 新通道超时，但 PTY 正常；用 localhost-only 反向转发
  完成 3.91 MB 源码传输，临时 HTTP 和转发已关闭，未重登、替换或关闭共享 master。
- 登录节点 /tmp 已满：预检使用本任务 scratch 临时目录；GPU 任务仍使用 node-local 临时目录。
- scratch 配额实查约 1.45 TB/5 TB、3557792/5000000 文件；未清理旧科学证据。

## 后续如何查

远端运行根目录：

`/scratch/yz11502/Research/Nav-axis-uturn-results/table2_continuous_expansion_20260912_v2`

- `submission.json`：初始两作业及范围。
- `construction_receipts/scene_00.json` 至 `scene_17.json`：完整 A 菜单分片。
- `population_freeze.json`：真实合法 A 数、序列／距离档分配、新总体和运行包 SHA。
- `eval_submission.json`：后续导航数组编号。
- `summary_submission.json`：后续完整汇总编号。
- `tasks/NNN/`：逐任务归档、收据和独立 verifier。
- `expansion_summary_JOBID.json`：按新 population 实际 count 汇总，含 C 前序 B 类型拆分和是否达到 10 查询。

若构造作业出错，冻结／导航不会被自动放行；查看原始失败记录并 exact retry，不重新提交整个总体。
如果冻结成功但摘要衔接失败，先查已写入的 eval_submission.json，避免重复提交导航。
QOSGrpGRES 是项目 GPU 配额等待，不是 SSH、环境或样本失败。

旧剩余 85 项累计 38287 GPU 秒，平均约 7.5 分钟，最慢 14 分钟。
新预算可能增加时间；总耗时依赖合法 A 数及排队。每项 1 h 不是总计 1 h。

完整冻结协议：`TABLE2_CONTINUOUS_EXPANSION_PROTOCOL_20260912.md`。
机器可读收据：`TABLE2_CONTINUOUS_EXPANSION_SUBMISSION_20260912.json`。
本轮未修改论文、真机、既有结果，未执行 Git commit/push。
