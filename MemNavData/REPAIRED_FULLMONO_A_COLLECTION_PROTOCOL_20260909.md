# 修复版完整单目：实际 A 采集协议

2026-09-09。此文件在新正式 A 采集前建立；不是 query population 封存，也不是新 SR。
它服务于新实际 mono-A → causal history → role-free native/raw/GEM 的最小主实验。

## 输入与总体

- 源 parent SHA：`a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5`。
- 沿用原 54-scene 顺序；其中 49 scenes 非空，共 196 条任务。空 scene 不补选。
- 每条任务保留原初始状态与 Goal-A 图像。目标图为源任务 `switch_idx - 1` 帧，
  不把载体中的另一张 goal 图误作 A；源图和资产在运行前绑定哈希。
- 源任务来自既有生成资产，不把 expert 轨迹用作记忆。记忆只来自这次策略实际执行的 A。
- 不按旧 A 成败、距离或 query 结果选任务；这批场景已被使用，不能称新场景独立确认。
- 保留原种子 `2026082200 + 100 * scene_rank + episode_rank`。

## 运行合约

复用已封存的 `repaired_fullmono_c8cf8c60e7efd55f`：实际 RGB、源图像坐标的单目深度、
bounded pursuit、Habitat 标准碰撞、无旧式落点 snap 与 30% 重试。
模型权重、相机高度尺度收据及服务参数不变；A 仅调用 native，无 memory 接管。
同 scene 四条 A 共用一个私有服务进程对，但每条按既有 evaluator 重置状态并使用自己的 seed。
不在此阶段构造或运行 Novel/Revisit 查询。

- 每个 A 最多 600 动作，执行 horizon 8，成功阈值 1 m；沿用原定义，不按结果改变。
- 模型只使用图像与单目预测；Habitat 仍提供模拟碰撞及 evaluator 的状态/成功判定。
  这不是自主视觉 STOP，也不验证真实机体碰撞。
- 全量保存逐动作、终点、RGB、深度与服务收据；复用独立 verifier 重算 SR/SPL 与执行动作。
- 每条 A 输出独立日志，名称包含 scene 与 episode，防止四任务复用旧单任务日志名。
- node-local 保存密集帧，结束后压缩归档并逐文件哈希回读；小型进度/最终收据保留在 scratch。
  不删除失败证据，不将缺日志或超时记成导航失败。

## 预检与提交顺序

1. 本机单元测试、bash 语法检查；远端原容器/解释器执行源绑定与四条 A 的 CLI dry-run。
   登录预检不启动模型、不创建 Habitat renderer；使用任务 scratch 临时目录，避开满的 `/tmp`。
2. 两条已消费修复历史的实际图像构造验证通过后，运行原源顺序的第一个非空 scene，四条 A 完整入口验证。
   这不是以 A 成功率为通过门；需四条任务正常终止、独立验证和归档通过。导航失败亦合法保留。
3. 通过后推进初始 scene 前缀 30，其非空 scene 的原始 rank 是 array index；
   已完整完成的首 scene 不重复提交，也不覆盖原始收据。
4. A 与构造同时完成后，只依据构造可行性使用预定义前缀 `[30,36,42,48,54]`，
   选择第一个达到 24 histories / 15 scenes 的完整前缀；全量仍不足则明确报告不足。
   前缀内全部合法历史保留，不只截取前 24 个。query 成绩不参与此决策。
5. 先独立封存全部 query 资产和三臂合约，再启动 query 评测；此采集入口不会自动触发下游。

## 调度

按共享 SSH 手册使用 `alantorch` / `yz11502`，A100、1 GPU、10 CPU、96 GB、1 小时。
scene 数组由源计划导出，初始并发不超过 4；不为绕过 QOS 改用已有不稳定记录的 H100 栈。
先 `safe_sbatch --lint-fatal --test-only`，检查实际环境后再提交一次真实任务。
当前文件、源计划和入口准备完成不代表已提交；实际 job ID 另写提交收据。

## 实现

- `plan_repaired_fullmono_population.py`：任务与前缀计划，不读取旧结果。
- `collect_repaired_fullmono_a.py`：一 scene 全部 A；逐动作/终点独立核验。
- `run_repaired_fullmono_a_collection.sh`：原运行栈、私有端口与无损归档。
- `slurm_repaired_fullmono_a_collection.sbatch`：A100 一小时执行入口。
- `REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_PROTOCOL_20260909.md`：新方向可构造性协议。

全程保留旧实验；本批不是其 exact retry。论文数字需待新配对结果完成后另行决定。
