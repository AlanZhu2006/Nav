# 2026-09-07：保留 SPL，并准备精确计分补跑

作者先要求“不要删除 SPL，还是保留”，随后要求提交 HPC 补跑。

当前活动论文 `/home/asus/Research/Memnav_Paper/tables/mono_factorial.tex` 已恢复 SPL 列，
暂用真实路径积分的上下界；Table II 的精确 SPL 不变。没有恢复已知不准确的旧单值。
待补跑完整验证后，应使用新轮次自身的 SR 与精确 SPL，不能拼接旧 SR 和新 SPL。

| Table III 行 | 当前 SPL bounds |
|---|---:|
| Metric / native | [0.118, 0.120] |
| Zero / native | [0.060, 0.061] |
| Mono / native | [0.121, 0.123] |
| Metric / CEC | [0.443, 0.454] |
| Mono / CEC | [0.492, 0.504] |

这些数值由 `correction.json` 全部 210 行的下界、上界重新取均值核对，向外取三位小数。
区间只反映最后动作实际位移在 0–0.0976 m 内的记录缺失，不是置信区间。
表头明确写 `SPL bounds`，caption 有一条简短说明；未使用区间中点估计。

本次仅修改一个 LaTeX 文件。全部 SR、其余正文、标题、Abstract、Introduction、
公式和引用不变。编译成功，仍为 8 页；最终日志无 warning / overfull / underfull。
第 6 页表格已渲染检查，没有裁切或溢出。

备份、PDF 与验证：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/paper_spl_retained_20260907_NE23hi/
├── before/
├── before_research/
├── build/main.pdf
├── table_page-6.png
└── verification.json
```

PDF SHA：`5ba99ea05fd4033d497d0dff5b9a8cabb108d99f39f41c248dab8b7a9fff711c`。
README、会议矩阵、总账均已改为保留 SPL bounds；之前删列报告保留为历史记录。
本次没有 commit 或 push。
