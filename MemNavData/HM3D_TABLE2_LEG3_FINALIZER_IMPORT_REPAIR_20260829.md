# HM3D Table-2 Leg-3 finalizer import repair（2026-08-29）

## 事故边界

Construction smoke 与 22 个 formal construction cells 均已 `COMPLETED 0:0`。
原 finalizer `16545223` 在 14 秒、写入 population 之前失败；其 verifier
`16545224` 因依赖取消。错误是：

```text
finalize_hm3d_table2_leg3_mixed_role.py
  -> build_final14_role_pair_scene.py
  -> generate_twoleg.py
  -> import quaternion
ModuleNotFoundError: No module named 'quaternion'
```

Finalizer 只为取得一个 JSON-only `role_contract()`，却导入了完整 Habitat query
builder。换成 Habitat Python 后还会继续导入与 finalization 无关的构造模块，因此根因
是 analysis / simulator dependency boundary 错误，不是应通过不断补依赖解决的问题。

该失败发生在任何 Leg-3 policy rollout 之前；没有 population、没有 verifier、没有
query outcome，也没有 partial SR 可读。

## 唯一修复

1. 把 `role_contract()` 及其两个冻结常量移动到纯 Python
   `final14_role_pair_contract.py`；
2. Habitat builder 与 CPU finalizer 从同一纯定义导入；
3. 保留 22 个只读 fragment，不重跑 query construction；
4. 用新 immutable analysis bundle 只重跑 finalizer 与 independent verifier。

Novel/Revisit 阈值、方向分层、source population、query seed、fragment bytes、power
gate 与 future controller 均不改变。Replacement finalizer 在提交前必须证明：

- 原 construction task bundle hash 完整；
- 22/22 completion receipts 存在；
- population 与 verification 尚不存在；
- 新 protocol 与冻结 protocol byte-identical；
- `memnav` CPU Python 可独立 import finalizer/verifier，且不导入 Habitat builder；
- 独立 verifier 仍从 sealed raw files 重新计算全部 gate。

只有 replacement verifier 写出 `verified=true` 且
`formal_policy_evaluation_authorized=true` 后，才允许提交 Table-2 controller rollout。

第一轮 replacement finalizer / verifier 为 `16546413 / 16546417`。Pure import 已
通过，但 finalizer 随后暴露只读 fragment mode 被复制到私有 staging sidecar 的第二个
基础设施问题；`16546413` fail closed，`16546417` dependency-cancelled。该轮仍未生成
canonical population 或 policy outcome，后续精确修复见
`HM3D_TABLE2_LEG3_FINALIZER_COPY_REPAIR_20260829.md`。
