# HM3D Table-2 Leg-3 finalizer copy-mode repair（2026-08-29）

Pure-import repair 已消除 simulator dependency，但 replacement finalizer
`16546413` 在写入临时 population 时以 `PermissionError` fail closed。22 个 source
fragments 在 construction 完成后按协议递归设为只读；Python `copytree` 复制了该 mode，
导致临时副本中的 `role_pairs.json` 也不可写。失败留下一个明确的
`population.tmp.*` staging 目录，没有生成 canonical `population/`，verifier
`16546417` 因依赖取消。

本修复只改变 finalizer 的临时文件处理：

- 重写路径绑定前，仅给临时 sidecar 增加 owner-write；
- source fragments 保持 byte-identical、只读；
- 异常清理会先恢复私有 staging tree 的 owner-write，再删除该 staging tree；
- 精确删除上一轮遗留的单个 `population.tmp.*`，并在新提交前要求临时目录数量为 0；
- 再次只运行 finalizer 与 independent verifier。

没有 query policy rollout，没有读取构造 gate 统计，没有改变 population、阈值、seed、
方向分层或 power gate。

## 最终结构结果

- replacement finalizer / verifier：`16546714 / 16546717`，均 `COMPLETED 0:0`；
- verifier SHA-256：
  `75da29bdd6fccaf73843694f0d9820f75b528fd2d5d23c5a7c8f19ca79f433e8`；
- `verified=true`、`construction_only=true`、`policy_outcomes_read=false`；
- 8 histories / 6 scene clusters / 16 queries；
- front / side / rear：`4 / 0 / 4`；
- forbidden old B/C identity overlap：0；
- `formal_policy_evaluation_authorized=false`。

22 条 source prefix 中，13 条无法在完整 A+B history 后构造新的
`covis<0.10` Novel-C，1 条无法构造标准 Revisit-C，最终只保留 8 条。该结果没有达到
冻结的 `16 histories / 10 scenes / each stratum >=3` power gate，因此不得提交
Table-2 controller rollout，也不得事后放宽阈值。
