# 2026-09-06 全项目审计后的实际修复

这是 `PROJECT_REAUDIT_20260906.md` 的执行续篇。本轮按
`icra-human-review` 的原则只处理已经证实的问题，不扩写防守性论述、不改变方法主张，
不改 title、abstract、Introduction，不涉及真机，也没有 commit/push。

## 1. 已完成：SPL 计分修正

确认旧 evaluator 在 navmesh snap 后仍累加指令步长，不能直接视为真实执行路径。
本轮没有重跑 GPU，只对冻结原始逐动作位置重新积分。

- Table II：459 条记录均有终点，可精确修正。A/B SPL 为 `0.618/0.200`；
  Novel-C 为 `0.176/0.176`；Revisit-C 为 native `0.149`、CEC `0.762`。
- 所有样本、SR、W/L 和显著性不变；没有覆盖任何旧 CSV/summary/verifier。
- Table III：210 条旧记录缺少最终落点，精确 SPL 不可恢复。SR 全部有效。
  已算出明确上下界，尚未用估计值替换论文数字。
- 向作者提出的待确认项：推荐移除 Table III 的 SPL 列、保留全部 SR；
  也可明确展示数值上下界。当前不能把这列旧数字当成已修复的投稿结果。

完整 old/new 数值、脚本、原始来源与 SHA：
`PAPER_SPL_CORRECTION_20260906.md`。

## 2. 已提交：HM3D authority-spectrum 的 8 个 exact repair

原 array `16929030` 已完成 20 个 history，失败 8 个。

| 原失败 | 归因 | 最小修复 |
|---|---|---|
| index 2 | 静止时相同 JPEG 在新 transaction 下命中旧深度缓存 | 按新 token 重新取对应深度，不复用旧记录 |
| indices 3,6,7,10,13,18,21 | 同场景不同 history 的 runtime 目录冲突 | 目录加入 history index |

没有改变候选、阈值、半径、控制器或样本。每个失败 history 重跑完整四臂，
不拼接部分臂；已完成的 20 个目录通过绑定 completion SHA 的来源链接保留。
原 index 2 的 partial 完全保留在旧 run root。

### 新任务

| Job | 内容 | 最近确认状态 |
|---|---|---|
| `17010301`，index 2 | 原缓存报错 history 的完整四臂门测 | PENDING，`QOSGrpGRES` |
| `17010302`，7 个 indices | 其余缺失 history，最大并发 4 | PENDING，依赖前项成功 |
| `17010303` | 28-history 全量 summary + independent verifier | PENDING，`afterany` 后检查完整性 |

GPU 配置：`h100_tandon,a100_tandon`，1 GPU / 72 GiB RAM / 每元素 1 小时。
没有申请 public H200；遵守共享 SSH 与 Slurm 手册。
当前是组 GPU 配额排队，`StartTime=Unknown`，不是 server 崩溃或依赖报错。
此时没有新的正式 SR。

新 run root：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_authority_spectrum_repair_20260906
```

新 task bundle：`hm3d_authority_exact_repair_c7098eeafd55ea95`。
新 server overlay：`hm3d_authority_stationary_cache_repair_77985a53be98bb27`。
原 runtime closure `51ee9a4ca063c7f1` 保持不变；未把 worktree 其他改动混入。

完整提交收据：`HM3D_AUTHORITY_EXACT_REPAIR_SUBMISSION_20260906.json`。
补跑协议：`HM3D_AUTHORITY_EXACT_REPAIR_PROTOCOL_20260906.md`。
这仍是 retrospective authority ablation，不是新的 fresh confirmation。

## 3. 已完成：活动论文的最小同步

论文根目录是 `/home/asus/Research/Memnav_Paper`，不是研究目录内的旧 `paper/`。

- `tables/continual_meeting.tex`：Table II 精确 SPL 更正。
- `sec/6_results.tex`：对应的两个 Revisit SPL 数字同步。
- `sec/5_experiments.tex`：明确本项目采用实际平面路径计数。
- `figures/cec_architecture.tex`：限制模块文本宽度、增加顶部标题区净空，
  修复节点与副标题重叠；没有增删方法步骤、箭头或图注结论。
- `EVIDENCE_LEDGER.md`、`CONFERENCE_EXPERIMENT_MATRIX.md`：登记已修正和未决项。

没有移动图表 float、压缩行距、修改模板或扩大技术论断。
保留了开始前全部用户修改。

### 编译和不变量

- 本地隔离副本按项目 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  构建，使用已缓存 TeXLive 容器，禁用容器网络。
- 8 页、US Letter；最终日志无 compilation error、undefined refs/citations、
  overfull/underfull box 或 LaTeX warning。
- 已渲染检查架构图所在第 4 页和主表/正文所在第 5 页。
- citation keys、labels、refs 集合一致；`main.bib` 和方法公式文件未改。
- main/title、abstract、introduction SHA 与修改前逐字节一致。
- 审阅 PDF：`.diagnostics/project_repair_20260906/paper_build/main.pdf`。
  **因 Table III 的 SPL 列待作者决定，这不是最终可投稿版。**

## 4. 回归验证

- 本机完整相关测试：**22 passed**。
- 12 条 Matplotlib/Pyparsing deprecation warnings 来自既有环境，未隐藏。
- HPC 实际解释器、实际新 overlay：**7 passed**，并检查了 import 的绝对来源。
- HPC Habitat 解释器：四种 route/adapter 的 `contract_dry_run` 均通过。
- Slurm GPU/array/CPU analysis 的 `--test-only` 和模板 lint 均通过。
- 两个 Git worktree 的 `diff --check` 均通过。

## 5. 还剩什么

1. 作者确认 Table III 的 SPL 呈现方式；这不能靠编译通过来替代。
2. 等已提交的 8 个配对块完成，再读取全量 summary/verifier。
3. 根据全量结果更新 authority ablation 的实质结论；不能预言 CEC 会超过 raw/witness。

本轮没有解决长程 Revisit 的算法瓶颈，也没有把未完成实验标成已完成。
