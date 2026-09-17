# 实验与证据索引（2026-09-17）

论文以独立 paper 仓库的 `main.tex` 和表格为准。本页指向执行、核验及历史记录；
Git 同步不重跑实验，也不更改论文分母。

| 实验 | 代码与记录入口 | 当前口径 |
| --- | --- | --- |
| Table I：共享历史、三控制器 | [协议](../MemNavData/TABLE1_REPAIRED_THREE_CONTROLLER_PROTOCOL_20260910.md)、[核验](../MemNavData/verify_table1_repaired.py) | HM3D 28、MP3D 42 个历史；三控制器分别配对 |
| Table II：连续目标与共享历史 | [状态](../MemNavData/TABLE2_CONTINUOUS_FORMAL_STATUS_20260912.md)、[汇总](../MemNavData/summarize_table2_continuous.py)、[共享配对核验](../MemNavData/verify_table2_common_pair.py) | 连续链、共同 Revisit-C、共享双阶段历史分别解释 |
| 几何验证与低支持查询 | [覆盖率消融](../MemNavData/LOW_COVISIBILITY_COVERAGE_ABLATION_RESULT_20260910.md)、[审稿补充](../MemNavData/GEM_REVIEWER_P0_RESULT_20260915.md) | 区分提示使用率、成败和几何接受率 |
| 转向与持续引导 | [当前 cutoff](../MemNavData/GEM_BEARING_CONTINUATION_STATUS_20260917.md)、[执行](../MemNavData/gem_bearing_attribution.py)、[核验](../MemNavData/verify_gem_bearing_attribution.py) | 53 条配对；Base / 无显式转向 / 初始对齐 / Full GEM 为 16 / 24 / 40 / 51 次成功 |
| 低近期重叠子集 | [协议](../MemNavData/GEM_REVIEWER_P0_PROTOCOL_20260915.md) | 原 13 条定义保留，53 条并非都属于该子集 |
| 档案与 KV 存储 | [模块说明](../NavDP/baselines/memnav/gem/README.md)、[资源汇总](../MemNavData/reduce_gem_memory_resources.py)、[无损存储](../NavDP/baselines/memnav/gem/LOSSLESS.md) | 历史档案与 KV 编码各自按相同写入协议比较 |
| Jetson—RTX 延迟 | [真机接入记录](https://github.com/AlanZhu2006/MemNav-RealWorld/blob/main/REALWORLD_MEMORY_STORAGE_SYNC_20260916_CN.md) | 冻结 JPEG 静止回放，与导航成功统计分别记录 |

53 条 bearing 由 42 条 HPC 和 11 条本机执行组成，每个查询的各臂使用同一 GPU。
另有一条本地恢复记录已核验并保留，未加入作者选定的论文 cutoff。当前没有继续补齐
70 条的任务。旧固定转向臂的实验记录保留，不重新加入稿件表格。

分母、种子、source SHA 和统计规则见原协议与收据。被取消的检索范围、位置读出
补充实验仍未完成；代码提交不代表产生了新实验结果。

## 历史与探索

- [2026-09-07 实验索引](EXPERIMENT_INDEX_20260907.md)：当时的主表、长程和学习支线。
- [架构研究记录](../MemNavData/GEM_ARCHITECTURE_AND_PROGRESS_20260914.md)。
- [长程关系记忆设计](../MemNavData/GEM_LONG_RANGE_RELATIONAL_MEMORY_DESIGN_20260915.md)。
- [历史数组与场景恢复](LOCAL_STORAGE.md)。

旧日期报告保留原数字与状态；旧报告的 pending/running 不代表当前作业状态。
