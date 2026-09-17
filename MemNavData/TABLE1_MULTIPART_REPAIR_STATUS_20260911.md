# Table I 补跑：确定根因后的传输修复

更新时间：2026-09-11 08:05，北京时间。本文不替代完整 SR/SPL 汇总。

## 当前进度

- 原正式数组：210 项已有归档，其中 **208 项完成并通过单项 verifier，2 项失败**。
- 失败项：110（MP3D/NavDP/gYvKGZ5eRqb/episode_0003）和
  158（MP3D/ViNT/sKLMLpTHeUy/episode_0005）。
- 上次原样补跑 `17336842` 换到 ga028/ga021 后仍失败，原输出保留。
- 本次修复补跑：`17348071_[110,158]`；完成后的全表汇总：`17348072`。
- 07:59 提交并回读：A100，1 GPU、10 CPU、96 GB、1 h，最多同时两项；
  当前 `PENDING / QOSGrpGRES`。汇总使用 CPU、8 GB、30 min，等待两项完整成功。

这不是只补最后一个失败臂：每项重跑 Novel-native、Novel-GEM、Revisit-native、
Revisit-GEM 四臂，保留原配对顺序。最终合并其余 208 项，再由原汇总器核验 210/210。

## 根因已经通过真实归档复现

不是模型方向错误，不是参数或 CUDA 随机性，也不是“严格校验太多”。
HTTP multipart 解析器在一个特定 64 KiB 分块边界上，错误地把分隔符中的 CR
写入文件内容，导致 JPEG 末尾多一个 `0d` 字节。

两条实际请求重构后长度恰好均为 **65,539 bytes**。在实际 HPC Python/容器、
Werkzeug 3.1.8 中，修复前收到的 SHA 与原失败记录逐字相同；修复后均恢复为
原始文件的 SHA。JPEG 解码像素在这两个案例中不受末尾 CR 影响。

| 单元 | 出错字段 | 应有文件长度 | 修复前内容 | 修复后 |
|---|---|---:|---|---|
| 110 | 当前 RGB | 65,190 bytes | 原 JPEG + CR | 精确原字节 |
| 158 | ImageGoal | 49,039 bytes | 原 JPEG + CR | 精确原字节 |

110 的原输入在 CPU 诊断里由已存 `ff d9 0d` 文件去掉已知额外一个 CR 重构；
158 使用原冻结目标文件与保存的当帧 RGB。生产代码**不做任何裁剪或 strip**。

修复仅把 partial-boundary 搜索误选到 LF 的边界向前移动一字节，连同 CR 一起保留。
三个私有 server 入口安装该修复，controller、GEM、模型、输入 JPEG 与校验逻辑不变。
原冻结 bundle 中其他 1,754 个文件逐字节保留，详见 `multipart_repair_manifest.json`；
没有把当前脏工作区整体打进去。

## 验证

- 本机完整相关回归：**233 passed**；12 个已有 matplotlib/pyparsing 弃用 warning。
- 覆盖：精确故障复现、64 KiB 周边长度、双文件目标上传、真实数据尾部 CR/LF
  不被裁掉、修复重复安装、controller 输入、Table I 配对、Table II 历史隔离与执行器。
- HPC 真实归档两例：`verified=true`；不跑 GPU，不重新导航。
- HPC parser pytest：退出码 0；ViNT 实际 Python 安装修复并导入：退出码 0。
- 原 SHA、帧、深度 transaction、目标消费与独立结果 verifier 全部保留。

完整 HPC 复现收据已下载到：
`.diagnostics/table1_multipart_20260911_p6Hepr/hpc_multipart_reproduction.json`。
本地与远端 SHA 同为：
`faf7a8a28851bd360d7440ac4e57191374d665462ea38e783318579957b7cb3e`。

## 封存与结果路径

修复源码：
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table1_multipart_9fcb3ea1d6449d31`

runtime receipt SHA：
`9fcb3ea1d6449d311093a1be0a2e281262afea317afd86f0d0a5fb76ceff8874`

原计划 SHA（不变）：
`0964f52738e22f2d16b099196605ef2e9681f57afce20fbdb99b417d8b535622`

新结果根：
`/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/multipart_retry_20260911_9fcb3ea1d6449d31`

其中：

- `preflight/multipart_reproduction.json`：两例真实输入的修复前后核验；
- `preflight/parser_pytest.log`、`preflight/vint_parser.log`：实际环境预检；
- `submission.json`：实际任务 ID；
- `recovery_manifest.json`：保留旧输出和精确替换范围；
- `evaluation/task_110`、`evaluation/task_158`：本次完整四臂输出；
- `combined_evaluation/paired_summary.json`：210 项齐全后才生成正式全表统计。

没有修改论文、模型参数或机器人；没有 commit/push。修复通过 CPU 验证不等于两项
导航已经完成，最终状态仍需读取新任务的归档与 independent verifier。
