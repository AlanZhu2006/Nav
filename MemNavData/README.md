# MemNavData：GEM 研究与评测入口

这里既保留早期多目标数据生成/训练代码，也包含当前 **GEM**（沿用 CEC 方向接口命名）
的评测、消融、HPC 提交和审计记录。不要把目录内所有实现都当作论文主方法。

## 建议阅读顺序

1. [最新项目总账](STATUS_20260917_GIT_SYNC.md)：目前架构、结果、工作区与存储。
2. [仓库与代码地图](../docs/REPOSITORY_GUIDE.md)：主线入口、数据/权重边界、运行环境。
3. [实验索引](../docs/EXPERIMENT_INDEX.md)：论文表格与正式结果、探索性结果分开。
4. [记忆模块](../NavDP/baselines/memnav/gem/README.md)：RGB history、LingBot state、NavDP FIFO 的区别。
5. [共享 SSH/HPC 手册](HPC_SHARED_SSH_OPERATIONS_20260816.md)：提交前先读，复用已认证通道。

## 常用文件类别

| 文件类别 | 职责 |
|---|---|
| `*_contract.py`、`*_runtime.py` | 接口语义与运行时读出；是否属于主线见代码地图 |
| `eval_*.py`、`run_*.py` | 闭环执行与固定实验编排 |
| `audit_*.py`、`independent_verify_*.py`、`summarize_*.py` | 输入、原始轨迹、统计及完整性核验 |
| `*_PROTOCOL_*.md`、`*_protocol_*.json` | 冻结实验问题、样本、臂与通过条件 |
| `*_RESULT_*.md` | 对应实验结果；不能仅凭文件名判断是否已经完成 |
| `*_SUBMISSION_*.json`、`slurm_*.sbatch`、`submit_*.sh` | 提交收据与 HPC 编排；Job ID 不等于成功结果 |
| `test_*.py` | 本机/相应依赖环境中的回归测试 |

旧脚本路径保持不变，因为冻结 manifest、source bundle 和历史结果引用这些路径。
本次通过索引整理主次，不移动或删除历史证据。

## 数据生成与早期训练

原 README 已逐字保存在 [早期数据生成说明](README_DATA_GENERATOR_LEGACY.md)。
它描述 expert 2-leg/3-leg 轨迹，包含历史配置示例，不是当前 CEC 运行指南。
当前主评测使用 actual-online history；长度诊断使用 controlled causal survey；
学习 probe 复用 train40/PT1 expert 数据。三种历史来源不能互换。

当前 bearing 截止口径见 [53 条状态](GEM_BEARING_CONTINUATION_STATUS_20260917.md)。
