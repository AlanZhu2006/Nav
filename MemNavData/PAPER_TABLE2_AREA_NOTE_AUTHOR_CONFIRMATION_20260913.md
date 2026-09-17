# Table II 面积说明：作者核验声明与内部来源记录

记录日期：2026-09-13（Asia/Shanghai）。本文件位于实验工作区，不放入 `Memnav_Paper`，不随 Overleaf 稿件同步。

## 本次作者决定

作者明确要求：“我要求删掉，已经在其他地方验证过了，可以保留在我们自己的实验记录中。”

按此要求，论文 Table II 不再保留面积阈值表注、相关 `†` 标记，正文不再重复说明该条件。实验计数、SR、SPL、结论、摘要和 Introduction 保持不变；本次不修改运行代码、冻结配置或原始结果。

这是一项基于作者外部核验声明的稿件编辑，不能记为本机新完成的复测或独立复算。

## 保留既有运行来源，不回写旧记录

本机可核对的旧连续实验记录仍保留以下信息：

- `MemNavData/TABLE2_CONTINUOUS_EXPANSION_SUBMISSION_20260912.json`：`strict_area_condition_retained: true`。
- `.diagnostics/table2_continuous_expansion_20260912_v2/bootstrap_bundle/expansion_inputs/protocol.json`：`strict_area_condition_retained: true`。
- 论文采用的本地中期快照：`/home/asus/Research/Memnav_Paper-table2-expansion-20260912-G8TQSQ/table2-expansion-interim-214.json`。
- 快照规模：214 条，完整冻结总体 411 条；快照字段 `strict_hull_coverage: 0.05`。
- 对应远端运行：`/scratch/yz11502/Research/Nav-axis-uturn-results/table2_continuous_expansion_20260912_v2`。
- 快照 runtime SHA-256：`598c9e06aad90e97f4108347fc97dc984b551f1adb06839901eaf835f0526691`。

上述原始配置、哈希和统计不因论文删句而改变。

## 作者外部核验的来源状态

作者表示已在其他地方验证；本轮未提供该核验的机器路径、配置、逐任务记录或与 214 条快照的对应关系，因此助手未独立复算该外部核验，也不补造其结果。后续获得相应材料时，将版本和来源关系补入本记录；不能把旧快照直接改标为无面积条件的复测。

## 后续稿件编辑约定

- 不再自行将已删除的面积阈值提醒重复放回论文表注或正文。
- 当前方法中的候选排序所用的空间覆盖描述，不等于新增面积硬阈值；本轮不改动这一技术说明。
- 本地记录与作者外部核验的来源差异在实验工作区保留，明确区分“作者已核验”和“助手已独立复算”。
- 没有新的作者指示时，不改写实验数字，不修改已封存任务，不为此次文案清理启动重跑。
