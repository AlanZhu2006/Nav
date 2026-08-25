# Final14 support-spectrum 独立派生复盘（2026-08-23）

## 结论

不需要为 support spectrum 重新运行一次长闭环。Final14 在同一批 `21`
条 causal online-A histories、`10` 个 scene clusters 上，已经前瞻冻结并完整运行了：

- unsupported Novel：`max online-A covis < 0.10`；
- weak/hard-support Revisit：`0.25 <= max covis < 0.55`；
- strong/standard-support Revisit：`0.55 <= max covis <= 0.90`。

Natural 与 hard-support manifests 的 `21` 个 `(scene, episode)` 身份及顺序逐项
一致。以下结果是对已独立验证 Final14 闭环的派生分析，不是新 rollout、不是新阈值、
也不是 fresh-scene Full-Mono 证据。

## 三档结果

| historical support | 实际 covis 范围 | CEC 授权 | Native SR | Raw fixed SR | CEC SR | CEC vs native | CEC vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unsupported | `0.000--0.093` | `2/21` | `7/21` | `2/21` | `8/21` | `+1/-0`, `p=1` | `+7/-1`, `p=.0703` |
| Weak | `0.399--0.406` | `19/21` | `3/21` | `19/21` | `19/21` | `+16/-0`, `p=3.05e-5` | `+0/-0`, `p=1` |
| Strong | `0.615--0.721` | `21/21` | `4/21` | `19/21` | `20/21` | `+16/-0`, `p=3.05e-5` | `+1/-0`, `p=1` |

SPL 同样来自冻结 summary：

| support | Native | Raw fixed | CEC |
|---|---:|---:|---:|
| Unsupported | `.1897` | `.0928` | `.2471` |
| Weak | `.0396` | `.6462` | `.6091` |
| Strong | `.0536` | `.6090` | `.6329` |

CEC 授权覆盖随支持增强呈 `2/21 -> 19/21 -> 21/21`。在这批控制人口中，
三档 CEC 相对 native 的 paired losses 均为 `0`。Weak/Strong 上 raw 与 CEC
已经接近饱和；CEC 的可识别价值主要位于 unsupported 端，避免 always-on memory
把原生探索重定向到错误历史。

## 科学口径

可以写：

> 在同一冻结历史人口的 controlled support sweep 中，CEC 的授权覆盖随
> online-history support 从 `9.5%` 上升到 `90.5%` 和 `100%`；它在两个
> supported 档保留 raw-memory utility，并在 unsupported 档恢复 raw memory
> 造成的多数闭环损失。

必须同时写：

- 这是 Final14 的 post-hoc derived analysis；三个 query population 和全部闭环
  arm 在读结果前已经冻结，但“support-spectrum”这一汇总视角不是预注册 primary；
- controller depth 为 metric，不能升级成 full-monocular support-spectrum；
- unsupported 上 CEC 对 raw 的 `+7/-1` 仍为 `p=.0703`，不能单独称显著；
- weak 样本被构造在约 `0.40` 附近，不代表完整连续 covis 曲线；
- certificate 是经验授权边界，不是形式安全保证。

## 复算与来源绑定

复算命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python \
  MemNavData/derive_final14_support_spectrum.py \
  --root .diagnostics/learned_relocalizer_20260817/\
final14_attempt7_formal_result_20260818
```

源 SHA-256：

- `paper_role_pair_summary.json`：
  `ab704752abcf624aebd9a598c80659995a8b443d6e7fb0e7944554b8ae320f07`；
- `paper_role_pair_independent_verification.json`：
  `268a104a64a0d9a040010646abc046f1d29323b17705ca4781ea9f10073d5318`；
- natural manifest：
  `7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a`；
- hard-support manifest：
  `5e261051b77c05a8bb91b0589b8032b5a77529f1ca4e479ce33e4ec4f2b47e5c`。

Independent verifier 中 `verified=true`，且其绑定的 summary SHA 与上述 summary
逐字节一致。

## 对下一步的影响

原计划的“重新构造三档并跑一次相同三臂”被取消，因为它会重复 Final14 已完成的
controlled evidence。若要花新的 HPC，唯一有新增信息的版本应同时满足：

1. fresh HM3D scenes；
2. actual mono Goal-A history；
3. full-mono query control；
4. 新增 weak-support query；
5. 三臂严格配对且不读取 support label。

在论文完成消融、常规导航指标和共驻延迟审计前，不提交这项昂贵扩展。
