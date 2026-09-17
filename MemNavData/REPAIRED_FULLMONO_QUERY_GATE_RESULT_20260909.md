# 修复版 full-mono：首个固定三臂任务通过

2026-09-09。`17262832_0`完成，`0:0`，总用时6分39秒。
固定population index 0：W7k2QWzBrFY / episode_0001 / Revisit，source index 61。
它来自封存前的hash顺序，不是从运行成功样本中挑选。该项保留进入完整90查询总体。

## 实际结果（N=1，运行检查，不作总体效应结论）

| 方法 | 到达 | 动作数 | 实际位移路径/m | 精确SPL |
|---|---:|---:|---:|---:|
| mono-native | 1 | 383 | 11.0983 | 0.206914 |
| mono-raw memory | 1 | 69 | 1.3137 | 1.000000 |
| mono-GEM | 1 | 70 | 1.2992 | 1.000000 |

三臂都到达，本项没有SR差异。运行/归档通过是展开剩余任务的条件；不是因为GEM成功才展开。
GEM接受并接管6次规划；没有runtime failure或geometry-stream stop。
raw与GEM各发生一次约152.5°物理转身，转身期间实际平移为0，新RGB持续进入LingBot，
完成后重新规划。该例可核实后方目标接口实际被执行，不证明所有纯旋转都无漂移。

## 独立运行与原始归档复查

- 三臂的实际新A来源、目标、seed、起点、图像与深度栅格通过逐动作verifier；
  SPL包含最终动作后的位移，未采用旧的动作前路径估计。
- 原始archive的全部成员逐文件回读通过；另直接读取该114,447,957字节归档，
  重算archive SHA并检查实际HTTP边界与逐帧几何输出。
- 三臂各有101帧实际A几何重放；每臂102条重放阶段memnav HTTP记录包含reset。
- 实际发送字段均不包含六个执行里程计字段；这项检查覆盖重放和查询，不仅检查当前导航。
- native/raw/GEM的几何帧分别484/170/171，`motion_receipt_recorded`全部为空。
- 48/6/6条实际规划深度收据均声明未消费metric-depth sensor；原独立verifier进一步复算了
  深度数组与源RGB栅格转换，不只依赖这个声明。
- 重放图像SHA一致，diffusion重采样为0；未将query role或共视作为模型请求输入。

归档SHA：`b433c62f5ceddf253a40387ae40270ca3a338e7b2a2fc8ce6673ff2774b996ac`。
独立verification JSON SHA：`8282457f13dc7308e29667262f535ba38331720ca1025350eeedf97caa4b0307`。
本机收据：`.diagnostics/repaired_fullmono_design_20260909/prefix30_progress/query_gate_task_000/`。

## 后续已提交

- `17263230`：同人口、同执行版本的index 1–89，A100/1h，最多4并发。
- `17263231`：CPU终局汇总，依赖首项和剩余数组结束，检查全部90项后输出配对SR/SPL。
- 不重跑index 0，不以新结果修改45 histories/23 scenes的总体。
- 另一个共视分层`17240715`和汇总`17240716`保持原冻结总体与入口。

[实际提交收据](REPAIRED_FULLMONO_QUERY_EVAL_SUBMISSION_20260909.json)。
总体结果仍未完成；论文TeX、摘要、引言及表格数字未改。
