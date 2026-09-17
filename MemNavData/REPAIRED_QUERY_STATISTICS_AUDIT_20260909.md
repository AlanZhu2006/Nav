# 两项修复版查询评测：统计入口交叉检查

2026-09-09。检查对象是已提交的两个终局汇总入口，不改变方法、样本、统计协议或任务依赖。
统计公式交叉检查使用合成数据；另对真实归档做只读输入检查，没有用部分SR作选择。

## 实际使用的代码

两个HPC包中的`summarize_covisibility_repaired.py`与本机测试对象逐字节一致：

- 共视汇总：`covis_eval_f284c23d7a98b0c2`，作业`17240716`。
- 新实际mono-A汇总：`repaired_role_summary_0790a21e7f856d83`，作业`17263231`。
- 统计模块SHA：`850aa2d455fe02b8906d6a24b78c0f5fb66e757c52a28d1d0879a55143e45405`。
- 本机原Habitat Python运行检查，未安装新依赖；结果为`FROZEN_STATISTICS_AUDIT_PASSED`。

## 独立检查结果

| 检查 | 参照 | 结果 |
|---|---|---|
| exact McNemar | SciPy双侧`binomtest`，不一致对数0–159的全部gain/loss组合 | 12,880项一致 |
| 五档Holm校正 | 三组手算向量，含乱序、并列、0及截断为1 | 全部一致 |
| scene-cluster bootstrap | 每次按抽中的scene实际展开其全部query，再逐条计算风险差 | 两个baseline各20,000次，与原汇总一致 |
| SR/SPL分母 | 不等scene大小的8条合成query，失败也留在均值分母中 | 计数及均值一致 |

这里没有增加一个新的统计方案。bootstrap仍为固定seed 20260909、20,000次，
重采样单位为scene，效应量为query加权的配对风险差，不是各scene成功率的等权平均。
合成样例特意使用3/1/4个query的scene，防止二者被误认为同一个估计量。

## 输出解释边界

- McNemar是query配对检验；不能将它称为“已作scene聚类修正的p值”。
  同时提供的scene-cluster区间处理了按scene整组重采样的问题。
- 共视主分析的五档分别对GEM-vs-native与GEM-vs-raw作Holm校正，
  并不等于整篇论文所有比较共用一次多重检验校正。
- 共视完整五档共同history子集是另一份对齐诊断，不能把其query数再加到主分析分母。
- 新full-mono输出Novel、Revisit、等比例合计三个口径，同时保留源A与构造损耗。
  条件query SR不能与A成功率相乘冒充整批源任务的joint SR。
- 两个任务均须有完整三臂与归档验证才写终局统计；缺失/失败任务只产生清单，
  不按零成功补齐，不以部分结果生成正式表格。

目前运行仍在进行，以上是统计实现检查，不是新的导航结果或已经完成的总体复算。

## 实际归档读入预检

随后在HPC原容器中直接调用已提交full-mono汇总包的`source_accounting`与`read_query`，
没有修改源码，也没有写入正式`paired_summary.json`。

- 完整112条A、88条到达、45条合法history/23 scenes的实际来源绑定通过。
- 固定query index 0（Revisit）和1（Novel）的真实三臂归档通过汇总reader，
  包括population、源码、目标、三臂记录及原始archive SHA，而不只是合成fixture。
- 输出为`REAL_SUMMARY_INPUT_PREFLIGHT_PASSED`；没有计算部分总体SR。
- 环境为既有CUDA Ubuntu SIF与Habitat Python；临时路径为
  `/scratch/yz11502/Research/Nav-axis-uturn-results/query_summary_real_inputs_20260909_gBmn5v`。

这关闭了“统计函数正确，但真实来源/归档格式读不进最终汇总”的首项预检风险；
仍须等全部90项完成后，由已提交CPU任务逐项核验，不能用两项检查替代完整终局。

共视CPU汇总也已在其实际使用的host MemNav Python下完成`--help`启动检查，
所需模块可导入；没有改为新环境、安装依赖或提前写终局结果。该检查不替代
159项完整归档的终局读入和统计。
