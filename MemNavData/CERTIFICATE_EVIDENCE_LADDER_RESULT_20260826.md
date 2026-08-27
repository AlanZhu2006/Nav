# CEC certificate evidence ladder result

日期：2026-08-26（Asia/Shanghai）
状态：完成；两项分析均由独立实现从原始收据复算通过。
边界：这是授权机制消融，不是新增的闭环 controller arm，也不允许据此在已消费的
held-out 集上调整阈值。

## 1. 问题

CEC 当前依次使用：

```text
DINO top-8
  -> SuperPoint/LightGlue + Fundamental support
  -> LingBot depth + PnP pose
  -> PnP inliers / query hull / reference hull / reprojection RMSE
  -> authorized scale-free bearing or exact native fallback
```

此前 raw-DINO、old geometry 与完整 CEC 同时改变了多个因素，尚不能直接回答：二维
correspondence precheck 是否已经足够，还是完整 PnP quality certificate 对开放集
abstention 确有必要。

## 2. Train40 完整 waterfall

输入为已经冻结并消费的 40 train scenes、480 sessions：

- geometry-selected PnP endpoints：480 rows；
- 同一 session universe 的 static top-8 LightGlue/Fundamental：3,840 rows；
- 480/480 selected endpoints 均按 `session_id + candidate_frame + path` 精确连接；
- actionability 仍沿用历史审计定义：PnP global position error `<=0.75 m`；
- role、co-visibility 与 GT actionability 只用于事后审计，不进入任何决策。

| 累积证据 | 接受 | TP | FP | FN | precision | recall | strict-no-match 接受 |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry-ranked candidate | 480 | 153 | 327 | 0 | 31.87% | 100.00% | 282/282 |
| Fundamental precheck | 288 | 147 | 141 | 6 | 51.04% | 96.08% | 115/282 |
| precheck + PnP bearing available | 253 | 147 | 106 | 6 | 58.10% | 96.08% | 86/282 |
| + PnP inliers | 178 | 139 | 39 | 14 | 78.09% | 90.85% | 21/282 |
| + query hull coverage | 155 | 136 | 19 | 17 | 87.74% | 88.89% | 8/282 |
| + reference hull coverage | 132 | 123 | 9 | 30 | 93.18% | 80.39% | 2/282 |
| + reprojection RMSE（完整 CEC） | 131 | 122 | 9 | 31 | 93.13% | 79.74% | 2/282 |

完整终点的 `122/9/31/318` 与 2026-08-14 的独立 certificate challenge 完全一致。

核心结论不是“所有阈值同等重要”。数据明确显示：

- 二维 precheck 保留高 recall，但 precision 只有 51.0%，不能作为开放集接管授权；
- 要输出一个可用 PnP bearing 并不能解决问题，precision 仍只有 58.1%；
- 最大的净化来自 PnP inlier 与双侧空间支持；
- 在这批数据上，RMSE `2 px` 最后一关只移除 1 个 TP、没有额外移除 FP，因此不能声称
  每个 gate 都获得了独立正向闭环验证。

### 冻结 operating point 的 train-only 敏感性

这是 operating point 冻结后的 one-factor-at-a-time 描述，不用于重新选阈值：

| 变化 | operating points | 结果摘要 |
|---|---|---|
| min inliers | 8/12/16/24/32 | precision 91.7%→94.5%，recall 79.7% 基本不变；32 时 79.1% |
| symmetric hull | .02/.05/.10 | precision 83.4%/93.1%/99.0%，recall 88.9%/79.7%/65.4% |
| max RMSE | 1/2/3/4 px | 1 px recall 仅16.3%；2--4 px 结果近乎不变 |

这说明 16-inlier 选择处于宽平台；5% hull 是明确的 precision--coverage 折中点；2 px
避免了 1 px 的过度拒绝，并位于 2--4 px 的稳定区间。

## 3. Fresh HM3D held-out receipt ladder

随后只读取已经完成的 fresh Full-Mono HM3D 56-query `mono_cec` plan receipts，不读取
任何导航 outcome。每条 query 恰有一个 uncached certificate decision，后续 cache
decision 全部一致。

| role | N | Fundamental precheck | precheck + PnP bearing | 完整 certificate |
|---|---:|---:|---:|---:|
| Novel | 28 | 15 | **12** | **0** |
| Revisit | 28 | 28 | **28** | **28** |

因此，在这批 fresh external scenes 上，完整 certificate 相对 precheck-only：

- 保留全部 28/28 Revisit authorization；
- 拒绝 12/28 个本来会被 precheck + valid PnP bearing 接管的 Novel query；
- role 标签只用于事后分组，运行时从未读取。

这项结果是授权分离证据，不是第四条闭环轨迹。它不能报告 precheck-only SR；但它已经
直接证明，若删掉完整 certificate，CEC 的 exact-fallback 合约会在大量 fresh Novel
query 上失效。

## 4. 实验决策

当前不提交一个 56-query 的 precheck-only 长闭环臂，原因不是回避 SR，而是该假设已在
与运行完全同源的收据上失败了更前置的授权必要条件：Train40 有 106 个
precheck+pose 假授权，fresh HM3D 又出现 12/28 Novel 授权，而完整 CEC 在不降低该批
Revisit coverage 的条件下全部拒绝。只有在审稿要求“错误授权造成多少闭环损失”时，
才值得把它升级为一条预注册的诊断臂；它不应成为新的方法选择实验。

论文中允许的表述是：

> Two-view correspondence support is an effective proposal filter but an
> insufficient authorization signal. The full pose-quality certificate
> preserves supported Revisit coverage while suppressing unsupported Novel
> takeovers.

禁止表述为 formal safety guarantee，也不能声称 precheck-only 的闭环 SR 已经测量。

## 5. 代码与收据

本机代码：

- `MemNavData/analyze_certificate_evidence_waterfall.py`；
- `MemNavData/independent_verify_certificate_evidence_waterfall.py`；
- `MemNavData/audit_closed_loop_certificate_evidence_ladder.py`；
- `MemNavData/independent_verify_closed_loop_certificate_evidence_ladder.py`。

本机 train-only outputs：

- `.diagnostics/train40_certificate_reuse_20260814/certificate_evidence_waterfall.json`；
  SHA256 `cd3067969c73d52f58a64f6188c32eaa0094436cbf9b48ea7d28100b87bbce32`；
- `.diagnostics/train40_certificate_reuse_20260814/certificate_evidence_waterfall_independent_verification.json`；
  SHA256 `1dd8f9aae122250a429caa2d0179269fcce95e09f9e06a9496d0e8b11e477d20`，
  `verified=true`。

远端 fresh HM3D outputs：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_fresh_fullmono_mixed_role_20260820/
  formal_20260820T143609Z_e6dd44c6/
    certificate_evidence_ladder_posthoc.json
    certificate_evidence_ladder_posthoc_independent_verification.json
```

- audit SHA256 `788a417a5aede7ab87afe8455038fbf35bb6ae166c48d5b9644fde3c168c6eb7`；
- verifier SHA256 `80b8a8bc92ee004ae3261d8bf8cabbf8d5b12cda68c2a1d4cba7300cd0f88442`；
- independent verifier：`verified=true`。

回归测试：相关 5 tests passed；联合旧 certificate recount 共 10 tests passed。
