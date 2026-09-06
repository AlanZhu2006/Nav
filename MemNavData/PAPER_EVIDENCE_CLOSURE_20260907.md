# 2026-09-07：三项论文审计问题的实施与验证

后续更新：作者要求保留 SPL，Table III 已恢复为 **SR + SPL bounds**；没有恢复
不准确的旧 SPL 单值。最新 PDF 与验证见
[PAPER_SPL_RETAINED_20260907.md](PAPER_SPL_RETAINED_20260907.md)。以下保留本日
初次删列阶段的实施记录及旧 PDF 哈希，不再代表当前 SPL 呈现方式。

## 1. 初次处理结论（SPL 呈现随后更新）

作者要求“先解决这几件事”后，已完成三项活动稿件修正及总账同步。
活动论文目录是 **`/home/asus/Research/Memnav_Paper`**，编译入口是 **`main.tex`**；
不是研究仓库内的旧 `paper/` 副本。

- Table III：移除不准确的旧 SPL 列，全部 SR 保留。
- Analysis：将错误混称的 `69/479` 修正为相对真实目标方向的 `20/479`。
- Results：加入完整 HM3D 四臂授权消融，包括 raw 总 SR 高于 CEC 的非显著结果。
- 会议矩阵、证据总账与 README 已同步，长度诊断不再显示为“尚未运行”。

标题、Abstract、Introduction、公式、引用和原有主表 SR 均未改变。
修改后 PDF 仍为 **8 页**，最终编译日志无 warning、未定义引用或排版溢出。
本轮按 `icra-human-review` 的证据优先、最小修订原则实施，没有扩写重复的防守性段落。

这表示本轮三项具体稿件问题已关闭，不表示全部研究风险已经消除。

## 2. Table III：先追查终点，再删除旧 SPL

额外追查通过文档指定的共享 SSH，以只读方式访问原 Final14 产物。检查对象来自
`paper_executed_spl_correction_20260906/correction.json` 的原始源文件清单，不另选样本。

| 检查 | 结果 |
|---|---:|
| 原 plans JSON / SHA 相符 | 210/210 |
| 原 CSV / SHA 相符 | 105/105 |
| rollout 最后一条 step 等于 `steps - 1` | 210/210 |
| memory trace 截止于同一最后 step | 210/210 |
| plans 内额外终点坐标字段 | 未找到 |

另检查首个 history 的 `summary.json`、最后一条 planning receipt、两套轨迹及
evaluator / MemNav / NavDP 日志。规划记录只到最近一次重规划，轨迹 SHA 不是坐标；
抽查的服务器日志没有补存末步坐标。

结论限于本轮已检查的产物，并不声称穷尽其他机器上所有可能存在的备份。
已保存的是完整前缀，不是整条轨迹丢失；最终目标距离标量也不能唯一恢复最终位置。
因此采用已授权的最小处理：**不再报告该表旧 SPL 单值，不插补终点，不重跑导航。**

修改：`/home/asus/Research/Memnav_Paper/tables/mono_factorial.tex:10`。

保留的五行 SR：

| 深度 / 历史 | Novel | Revisit | Overall |
|---|---:|---:|---:|
| Metric / disabled | 5/21 | 6/21 | 11/42 |
| Zero / disabled | 3/21 | 1/21 | 4/42 |
| Mono / disabled | 6/21 | 4/21 | 10/42 |
| Metric / CEC | 6/21 | 20/21 | 26/42 |
| Mono / CEC | 8/21 | 20/21 | 28/42 |

Table II 的 459 条轨迹有完整终点，继续使用已修正的实际位移 SPL：
A `0.618`、B `0.200`、Novel-C 两臂 `0.176`、Revisit-C `0.149 → 0.762`。
本轮没有再次修改 Table II。

原上下界及计分定义见 [PAPER_SPL_CORRECTION_20260906.md](PAPER_SPL_CORRECTION_20260906.md)。
原 sealed CSV、summary、verifier 与 additive correction 均保留，未被覆盖。

## 3. 两项正文修正

### 3.1 定位误差的比较方向

位置：`/home/asus/Research/Memnav_Paper/sec/7_analysis.tex:42`。

Before:

> A learned pose estimator retains 19/21 Revisit successes but releases 69/479 accepted bearings with angular error above $90^\circ$.

After:

> A learned pose estimator retains 19/21 Revisit successes, but 20/479 accepted bearings deviate by more than $90^\circ$ from the goal's relative direction.

`69/479` 是相对测地线路线首段的方向差；`20/479` 才是相对真实目标直线方向的误差。
这次是统计定义纠正，不是新的闭环结果。`19/21`、分母 `479` 及旧模型未获替换资格的
决定均未改变，也不将逐计划记录当作独立 episode。

依据：[PI3X_TEMPORAL_CONTEXT_ATTRIBUTION_20260906.md](PI3X_TEMPORAL_CONTEXT_ATTRIBUTION_20260906.md)，
[OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md](OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md) §6.1。

### 3.2 HM3D 完整四臂结果

位置：`/home/asus/Research/Memnav_Paper/sec/6_results.tex:74`。
采用一段平行结果，不增加第五张表，也不改原 Table I / IV 的数值。

| 本轮 replay | Novel / 28 | Revisit / 28 | Overall / 56 |
|---|---:|---:|---:|
| Native | 7 | 8 | 15 |
| Raw fixed | 9 | 26 | 35 |
| Valid PnP | 4 | 26 | 30 |
| CEC | 7 | 25 | 32 |

正文说明 CEC 的 28 条 Novel 执行不变；raw 对 native 在 Novel 上 `+7/−5`。
CEC 相对 raw 和 valid-PnP 的总体差异分别为 `p=0.581`、`p=0.727`，均不显著。

这是同一 Table-I HM3D population 的 **retrospective replay**，不是新增 fresh confirmation。
它与原 Table-I 的 `14/56 → 31/56` 分开，也与另一完整单目 population 的
`17/56 → 32/56` 分开。不跨轮拼接 native，不删掉 raw 的真实 gain。

完整配对、置信区间及冻结依据见
[HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md](HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md)。

## 4. 总账与会议清单同步

论文仓库内已更新：

- `EVIDENCE_LEDGER.md`：Table II 精确 SPL、Table III 删列决定、Pi3X 角度定义、
  HM3D 四臂完整结果、长度实验当前状态。
- `CONFERENCE_EXPERIMENT_MATRIX.md`：保持会议四张主表映射，明确区分“实验完成”
  与“得到正结果”；保留旧执行快照并标明已被当前状态取代。
- `README.md`：纠正跨 controller 主表编号为 Table I；Table III 是深度消融，
  Table IV 是授权机制，不再误写成 Table IV 的两个子面板。

研究仓库内三份旧审计文档也加上实施状态，不再悬挂“等待作者授权”。

长度诊断的远端 sealed summary/verifier 已重新只读核验：48 histories、每桶 16，
96 queries、192 arm-role rows，`verified=true`。Revisit native→CEC 分别是
`2/16→4/16`、`1/16→2/16`、`2/16→0/16`。
其历史来源是 `controlled_causal_rgb_geodesic_survey`，不是实际 NavDP-A rollout。
本轮只同步状态，没有将这个诊断重命名后塞进端到端主表，也未重新采纳其旧 SPL。

远端目录：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e/table3_result
```

- summary SHA：`9cb4e883906e9509017b7bc7b10f8fdd93f4404b38ac7d5847ec92dcc3bbda47`。
- verifier SHA：`f26f8928c87bfde17451856a4f654cc6c03648cd2dc3f0340d40d73ebc135e7e`。

## 5. 构建、排版和不变量验证

先保存带原有未提交改动的工作树快照，再在独立目录编译，未用 Git HEAD 覆盖当前版本。
缓存 TeX Live 容器 `495075f44d71` 断网运行，构建命令：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

仅在独立构建目录复用已有 `placeins.sty`，没有安装/更新模板、包或依赖。

| 检查 | 修改前 | 修改后 |
|---|---:|---:|
| 编译成功 | 是 | 是 |
| PDF 页数 | 8 | 8 |
| 最终 LaTeX warnings / errors | 0 / 0 | 0 / 0 |
| Undefined refs / citations | 0 / 0 | 0 / 0 |
| 重复 label | 0 | 0 |
| Overfull / underfull boxes | 0 / 0 | 0 / 0 |

最终 PDF 是 US Letter、PDF 1.4，字体均嵌入。渲染检查第 1、5、6、7、8 页，
覆盖首页、四张表、修改后的 Analysis、正文末页和 references；未见新增裁切、重叠或空白页。
没有移动 float、修改字号/行距/模板，也没有添加负间距。表格编号仍为 I–IV。

不变量核对结果：

- `main.tex`（含标题）、`main_lg.tex`、Abstract、Introduction：逐字节不变。
- Table I / II / IV：逐字节不变；Table III 所有 SR：逐项不变。
- 全部 citation、label/ref 命令及数学环境：保持不变。
- 数字变化仅限授权删除的五个旧 SPL、`69→20` 的定义纠正及新增 HM3D replay 数据。
- `.bib`、class/style、preamble、方法公式、架构图：不变。
- `git diff --check` 通过。

保护段落 SHA：

```text
Abstract:     3b389cd59ff8aec60d7fd0afbd11d7b01ddbd41ab32ccbe5abe37f877f4c9e2e
Introduction: e0de4f23ae6122c3dbbc8f4de420ef79113667dd0b78707715a3f77cab5d00d2
```

本轮备份与验证目录：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/paper_evidence_closure_20260906_zLy4Dq/
├── before/                     修改前论文工作树快照
├── before_research/            修改前三份研究审计文档
├── before_build/main.pdf       修改前 8 页 PDF
├── after_build/main.pdf        修改后 8 页 PDF
├── rendered/                   逐页检查图
├── verification.json           构建与不变量核对结果
└── terminal_position_followup.json
```

## 6. 本轮没有扩大到哪些工作

- 没有重跑 Final14，没有提交新 HPC 作业，没有恢复已暂停的 learned 分支。
- 没有启动或修改真机任务。
- 没有 commit 或 push；Overleaf 尚不会自动出现这些本地修改。
- 没有宣称解决长程失败。16-history 四臂 oracle 归因仍无完整配对结果。
- 同协议外部记忆方法、自主 STOP/到达后保持及外部负责的正式真机结果，仍是独立
  待办；它们不是本轮三项修改已经补出的证据。

下一步若继续，应先检查这份 PDF，再由作者决定投稿前是否投入长程归因或其他研究补充。
旧 CEC 论文不需要等待新增 learned 分支成功，才能使用已经完成的实验。
