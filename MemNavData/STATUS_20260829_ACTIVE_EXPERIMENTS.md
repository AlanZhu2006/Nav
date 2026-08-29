# 2026-08-29 活跃实验与修复总账

本文件只记录调度状态、结构完整性和冻结后的基础设施修复，不记录任何未封存的
SR、SPL、距离或逐臂结果。证据读取顺序仍为：独立 raw-file verifier 通过，最终
seal 写入，然后才读取汇总。

## 1. HM3D Table-1 controller portability

目标是在同一冻结的 28-history、21-scene mixed Novel/Revisit population 上，分别
估计 CEC 相对 NavDP native 和 ViNT native 的成对闭环效应。Novel/Revisit role
不会作为运行时输入；NavDP 与 ViNT 只做各自内部的 paired comparison，二者绝对
SR 不作为 controller superiority 结论。

### ViNT 分支

- 原正式数组：`16526731_[0-27]`；
- 唯一不完整 cell：index `18`，Habitat 在 pair audit 生成前 `SIGABRT`；
- 只重跑 index `18`：`16540176`；
- replacement aggregate / verifier：`16540207` / `16540208`；
- partial archive manifest：
  `1cf562864f59b11120798761defe7bcd610b67939843d56214a226fc38ef7946`。

修复没有改方法代码、checkpoint、query、seed、arm 顺序、600-step budget 或成功
半径；失败 cell 的 partial directory 保留为只读证据，不进入估计量。

### NavDP 分支

- 原正式数组：`16528369`；
- 三个非空 ranks `36,38,41` 在静止重复帧处遇到同一 transaction-cache 异常；
- 原因是 byte-identical JPEG 对应了新的合法 causal token，而旧 server cache
  仅按 JPEG digest 命中后把新 token 误判为冲突；
- 仓库中已有的回归修复被做成最小 server overlay，未知 token 仍 fail-closed；
- 只重跑 `16541366_[36,38,41%1]`；
- replacement aggregate / verifier：`16541367` / `16541368`；
- partial archive manifest：
  `52276bdc08e6158f7642b3ce70735d10c2504633048cb4887da9c8bed25fb5b1`。

原数组的 ranks `48-53` 是预声明的空 ranks，已正常完成且不重跑。最终 joint seal
为 `16541369`，同时依赖 NavDP verifier `16541368` 与 ViNT verifier `16540208`。

## 2. HM3D full-mono lifelong / multileg

上游 factual-C integrity barrier 已覆盖冻结的 22 histories / 15 scenes；其后封存
得到 17 条可进入 B2 的 accepted histories。两者是不同阶段的分母，不能混写。

旧 resume launcher `16521679` 在提交任何 B2 GPU job 前拒绝了
`gh005.hpc.nyu.edu` 这类 FQDN。修复只做严格规范化：接受 `ghNNN` / `gaNNN`
或加精确后缀 `.hpc.nyu.edu` 的等价形式，传给 Slurm 时恢复为短主机名；其他
hostname 仍拒绝。

- immutable amendment bundle：
  `hm3d_lifelong_node_affinity_repair_ddd01842308dfa37`；
- repaired resume launcher：`16540396`，已完成；
- B2 true-stack smoke：`16540468`；
- smoke 后的 node-affine formal launcher：`16540469`；
- study 始终标记为 underpowered mechanism experiment，不升级为 powered
  confirmation。

## 3. 当前科学边界

这轮工作没有新增架构、阈值或数据选择，也没有根据 partial outcome 做调参。它
解决的是三个会让正式表格不可审计的问题：单 cell 非科学崩溃、合法重复图像的
transaction cache 冲突、以及 receipt FQDN 与 Slurm short hostname 的表示差异。
只有在两个 Table-1 verifier 与 joint seal 全部完成后，才读取 controller
portability 结果；只有 lifelong 的 formal B2 aggregate 与 independent verifier
完成后，才读取 multileg 结果。
