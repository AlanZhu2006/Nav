# DINO 历史图替换 ImageGoal：本机运行状态

更新时间：2026-09-10 12:57，北京时间。**16次导航已全部完成，独立验证通过。**
固定设计见 `DINO_IMAGEGOAL_LOCAL_PROTOCOL_20260910.md`，未根据中途结果修改。

## 1. 本轮到底在测试什么

比较四种记忆接口：原始 ImageGoal、DINO 检索历史图替换 ImageGoal、
原始 ImageGoal + raw 固定方向、原始 ImageGoal + strict GEM 认证方向。
新臂只替换真正送入 NavDP 的 goal JPEG，保留当前 RGB/单目深度、seed 和执行器。
它不使用 PointGoal 或几何证书，不读取 query role；评分仍针对原始目标。

使用已经消费的两个场景、四个查询（2 Novel + 2 低共视 Revisit），
四臂同机同服务重新运行，共16次导航。旧 actual-mono A 重放，新执行链运行 query。
只是小样本接口/机制对照，不是 RANa 全方法复现、外部新场景确认或完整新A评测。
image-only 与 GEM 使用 frame≥8 的候选域，raw 保留旧域；并非只改一个变量的纯 bearing 消融。

## 2. 已通过的检查

- 四臂实际 CLI 预检通过；私有 MemNav/NavDP 服务 21910/21911 已启动。
- 检索是首次 query append 的替换，不额外写入 FIFO；选中历史图固定到 query 结束。
- 历史 JPEG 与实际回放收据逐字节匹配；日志记录每次真正送入 NavDP 的 goal SHA。
- 禁止本臂读取 learned gate/目标位姿来选 anchor；禁止将评测 role/GT 位置加入策略请求。
- 本轮四个相关测试文件共 **39 passed**，`git diff --check` 通过。
- 独立验证器已检查全部16条的动作、深度、精确 SPL、历史配对和目标替换证据。

测试命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python -m pytest -q \
  MemNavData/test_dino_imagegoal_substitution.py \
  MemNavData/test_low_covisibility_navigation_local.py \
  MemNavData/test_repaired_fullmono_local.py \
  MemNavData/test_vint_rgb_input_bridge.py
```

## 3. 11:56 的中途快照

已完成6/16 rollout，正在运行 query1 的 GEM。
首个 Revisit 查询的 native/raw/GEM 成功、DINO-imagegoal 失败；
query1 的 DINO-imagegoal 和 raw fixed 也已结束。**不据此报告总体 SR、显著性或方法优劣。**
不按中途成功率停止、追加有利样本或更换检索范围。

HPC 既有低共视四臂数组在11:54已核验11/159项；已完成作业 ExitCode 均0:0。
11:56 的队列中 task11/12 正在两张 A100 上运行，`17297866` 等待最终汇总依赖。
其余由 `17297865` 继续，排队原因含 `QOSGrpGRES`，
不是 SSH/环境错误。本轮没有重复提交、改变并发或读取中途 SR 来改规则。

## 4. 路径与运行入口

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/dino_imagegoal_pilot_20260910_78edUj/navigation_v1
```

- `manifest.json`：开跑前固定的输入、源码、设计与四臂顺序。
- `progress.json`：当前 query/arm 与完成数量。
- `summary.json`：逐条已完成结果；`completed=false` 时不当作最终报告。
- `evaluation/qNN/{native,dino_imagegoal,raw_fixed,cec}/`：完整执行证据。
- 新臂内的 `retrieval_selection.json` / `imagegoal_substitution.jsonl`：选图及真实输入收据。
- `independent_verification.json`：完整结束后自动生成，`verified=true`。
- `logs/`：各臂、私有服务及验证日志。

实际启动命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python -u -m MemNavData.run_dino_imagegoal_local run \
  --inputs /home/asus/Research/Nav-graph-blind/.diagnostics/low_covisibility_no_coverage_20260910_Raarrp/navigation_inputs_v1 \
  --out /home/asus/Research/Nav-graph-blind/.diagnostics/dino_imagegoal_pilot_20260910_78edUj/navigation_v1 \
  --mem-port 21910 --nav-port 21911
```

这是已在运行的命令，不要重复启动到相同目录。完整结束后私有模型服务自动退出。
真实机器人已有的18888/8888服务未停止、未修改；没有修改论文或默认方法。
Table I 准备状态另见 `TABLE1_REPAIRED_QUERY_PREPARATION_20260910.md`。

## 5. 最终结果（小样本接口诊断）

| 方法 | Novel | 低共视 Revisit | 合计 |
|---|---:|---:|---:|
| native | 1/2 | 1/2 | 2/4 |
| DINO-imagegoal | 0/2 | 0/2 | 0/4 |
| raw fixed | 0/2 | 2/2 | 2/4 |
| strict GEM | 1/2 | 1/2 | 2/4 |

两个场景、四个查询，不构成方法泛化优劣判断。它至少说明：在这四条固定小测中，
仅把DINO历史JPEG替换成NavDP的ImageGoal没有兑现方向残差的收益。
不能据此否定RANa完整方法，也不能外推ViNT/NoMaD的认证anchor接口；两者controller和证据机制不同。
私有21910/21911服务已自动退出，未关闭真机服务。
