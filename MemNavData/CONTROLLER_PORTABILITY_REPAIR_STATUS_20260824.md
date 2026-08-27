# Controller-Portability Formal Repair Status（2026-08-24）

> 本文只记录 2026-08-22 controller-portability 正式矩阵的完整性审计与
> 2026-08-24 repair。它不把 partial outcome 当正式结果，也不替代冻结协议
> `CONTROLLER_PORTABILITY_PROTOCOL_20260821.md`。

## 1. 正式问题与应有分母

五个 frozen downstream controller：

- GNM；
- iPlanner；
- NoMaD；
- ViNT；
- ViPlanner。

每个 controller 在同一个 sealed factual-B support population 上运行三臂：

- `all_prior`：B2 可使用 A+B 的全部先前历史；
- `initial_leg_only`：B2 只能使用初始 A 历史；
- `forced_reject_native`：运行相同 CEC probe/certificate，但逐动作禁止接管，
  统一回退 shared mono NavDP。

冻结 population 实际为 **18 episodes**，不是早期口头记录的 19。原 array 的
index 18 均按协议 `SKIP outside_supported_population`。因此完整矩阵是：

```text
5 controllers × 3 arms × 18 episodes = 270 result triples
```

每个 result triple 必须同时包含 `metric.csv`、`<episode>_plans.json` 和
`summary.json`。

## 2. 第一轮真实完成度

对远端原始文件逐项审计后：

- 完整 triples：245/270；
- 不完整 triples：0；
- 完全缺失或因严格审计中止：25/270。

| Controller | 完整 triples | 三臂 paired-complete episodes |
| --- | ---: | ---: |
| GNM | 50/54 | 14/18 |
| iPlanner | 47/54 | 13/18 |
| NoMaD | 46/54 | 12/18 |
| ViNT | 52/54 | 16/18 |
| ViPlanner | 50/54 | 14/18 |

25 个缺失单元的原因在读取任何 repair outcome 前冻结为：

| 原因 | 数量 | 含义 |
| --- | ---: | --- |
| arm 未产生目录 | 19 | 第一轮 Slurm/依赖执行未覆盖，不是导航失败 |
| `query C reopened a session within one goal` | 4 | 同一 episode 的 forced arm 合约异常 |
| `shared trace rendered RGB mismatch` | 2 | NoMaD 同一 episode 两臂在共置 H100 上严格重放失败 |

机器可读冻结集合：
`lifelong_nnr_controller_repair_manifest_20260824.json`，SHA-256
`b4ce4d2bf2ead920ad2e42059fea79fb25fc79c8b20b3bcae4275eff2be34f98`。

## 3. Session 异常的确定根因与修复

四次异常全部是：

```text
fzynW3qQPVF / episode_0001 / forced_reject_native / query C
```

并分别出现在 GNM、iPlanner、NoMaD、ViPlanner；ViNT 对应单元通过。异常不是
MemNav 长程记忆重新初始化，也不是统计器误判。根因在
`cec_controller_portability_hub.py::reset_short_context`：

1. query C 内碰撞恢复调用 `navigator_reset_env`，只应清理 bounded controller
   FIFO；
2. 旧实现同时把 hub 的 `_goal_sha256` 清空；
3. 下一动作仍是同一个 Goal-C，但 hub 因哈希为空而把它标成新 goal；
4. MemNav 正确保留原 session，因此 hub 的 `expected_start=true` 与 MemNav 的
   `started=false` 相互矛盾，严格 evaluator 中止。

修复只删除短期 reset 对当前 goal identity 的清理：

- full episode/router reset 仍会清 goal identity；
- collision/FIFO recovery 保留 active goal session；
- proof anchor cache 仍清空并重新认证；
- 长期 causal history 不变；
- 原 evaluator 的严格 session 审计完全保留，没有降级为 warning，也没有把异常
  强行计成成功或失败。

本地完整提交门：**113 tests passed**；Python compile、shell syntax、source-bundle
import selftest 和 NavDP runtime import 全部通过。

## 4. RGB mismatch 的处理边界

两次 mismatch 均为 `dhjEzFoUFzH/episode_0005` 的 NoMaD
`all_prior`/`initial_leg_only`。它们在同一物理 H100 GPU UUID 上并发时失败；同一
sealed trace 在其他 controller/GPU 组合上能通过。

Repair 没有放宽 JPEG SHA-256 或 RGB 重放容差。正式做法是：

- 只用 `a100_tandon`；
- array concurrency 固定为 1；
- 每个单元 time limit 固定为 1 小时；
- 重放仍要求逐帧 JPEG hash 完全一致。

因此 repair 若再次 mismatch 会继续失败，不会通过容差掩盖。

## 5. 不可变 repair 与合并协议

已有 245 个完整结果保持只读，不覆盖、不重跑。25 个 repair 写入独立根目录。
聚合器只有在以下条件全部成立时才可从 repair root 读取：

1. 原位置缺少完整 triple；
2. `(controller, population index, scene, episode, scope)` 精确存在于冻结 manifest；
3. benchmark hash 与 factual online-B trace hash 仍匹配 sealed population；
4. repair triple 三文件齐全，runtime role 不可见；
5. post-audit 看到的 repair 集合与 25 个 manifest entries 完全相等，无多、无少。

最终任务会对五个 controller 分别运行原 aggregate 与独立 raw-file verifier，并要求：

- 每个 controller 恰好 18 episodes、三臂齐全；
- forced arm 没有实际 takeover；
- causal A/B/C prefix、candidate ceiling、goal-session receipt 和 accepted anchor
  重新核验；
- 总计恰好 `245 + 25 = 270` 个完整单元。

## 6. 当前 HPC 状态

### V1：启动接线失败，零评测结果

```text
repair array : 16246886
finalizer    : 16246894
```

25/25 array tasks 均在 12--21 秒内以 exit code 2 fail-fast，`complete_metrics=0`；
没有加载模型或运行导航。原因是 repair wrapper 把旧 controller output root 错传成
原 arm 的 `ORIGINAL_RUN_ROOT`，导致原 arm 在错误位置查找唯一 sealed
`factual_b_support`。Finalizer 随后正确拒绝空 repair 集。

V1 run root 与 bundle 保留作失败审计证据：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  lifelong_nnr_controller_repair_20260824/
  lifelong_nnr_controller_repair_20260824T175706Z

/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  lifelong_ctl_repair_3c51adac806be82a
```

该失败不属于任何 controller outcome，不能计为导航失败。

### V2：接线修复后重新提交

V2 wrapper 不再读取或覆盖 controller old-root 变量；原 arm 始终从 NavDP-line
`ORIGINAL_RUN_ROOT` 读取 sealed support population。重新执行相同门禁，仍为
113 tests passed，并生成新的 immutable bundle：

```text
repair array : 16251793
finalizer    : 16251798
partition    : a100_tandon
array        : 0-24%1
time/task    : 01:00:00

run root:
/scratch/yz11502/Research/Nav-axis-uturn-results/
  lifelong_nnr_controller_repair_20260824/
  lifelong_nnr_controller_repair_v2_20260824T194245Z

source bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  lifelong_ctl_repair_9eb4a20c3f94d99e
```

V2 `SOURCE_BUNDLE.sha256` receipt SHA-256：
`e9602afb15401774324f73475a1541f134873d8250c6a25a6f7bc48110c3cb44`。

Pre-repair audit 已完成且 `verified=true`：245 个 original complete、25 个 repair、
原因计数 19/4/2 全部与 manifest 一致。

截至 2026-08-24 03:46（Asia/Shanghai），V2 array 因项目级 `QOSGrpGRES` 处于
pending；finalizer 正常等待
dependency。**尚无任何 repair outcome，也没有新的正式 SR。**不能把“已提交”写成
“已完成”。

监控命令：

```bash
squeue -j 16251793,16251798 \
  -o '%.18i %.12P %.24j %.2t %.10M %.20S %.30R'

sacct -j 16251793,16251798 \
  --format=JobID,State,Elapsed,Start,End,NodeList%20,ExitCode
```

完成后唯一正式入口应为：

```text
<run root>/formal_summary.json
<run root>/post_repair_audit.json
<run root>/FINAL_RECEIPTS.sha256
<run root>/VERIFIED
```

只有 `VERIFIED` 存在、`complete_arms=270` 且五个 independent verifier 全部为 true，
才允许报告 controller-portability 正式结果。

## 7. 当前不得使用的数字

245 个 complete arms 上可以做基础设施诊断，但各 controller 缺失 episode 不同，属于
非随机 complete-case 子集。任何基于 12--16 条 paired-complete episode 得到的 McNemar
值都不能替代正式 18-episode 结果，也不应写入论文主表。正式结论必须等待 repair +
post-audit + independent verification 全部完成。

## 8. 2026-08-25 shared-C 因果修复

旧 portability lifelong 设计在每个 arm 内独立执行 C，因此不同 C 轨迹导致 B2 起点和
因果历史不同。新协议改为：每个 controller 只执行一次 factual C，在读取任何 B2 outcome
前冻结 RGB/pose/decision/goal-session trace；随后两条主臂在同一 GPU、同一已加载进程内
精确重放 A/B/C，只从 B2 candidate ceiling 开始分叉。Forced-native 使用同 GPU 的新
fail-closed hub。新增合约测试与 bundle 门禁共 `128 passed`，controller 专用提交门为
`118 passed`。

ViNT 初始任务：

```text
run root:
/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_controller_20260822/
  lifelong_shared_c_vint_repair_20260824T170803Z
collection: 16288964
```

index 0/1/2/4/5 在 `gh001` 完成；index 3 被调度到 `gh013` 后，严格 A-trace RGB hash
重放在进入 C 前拒绝。该失败再次确认跨节点 Habitat/CUDA 渲染不能支撑 byte-exact factual
trace。失败 partial 已可恢复地移至
`quarantine_failed_collection/003_e9zR4mvMWw7_episode_0003_job16288964_3_gh013`；旧的
seal/eval/aggregate/verify `16288974/16288981/16288990/16289004` 已取消，未生成结果。

修复只改变执行节点，不改变任何 episode、方法、阈值或 outcome selection：缺失 index
`3,6-17` 和后续 B2 eval 固定 `ReqNodeList=gh001`。新 DAG：

```text
repair collection : 16289953
shared-C seal     : 16289954
paired B2 eval    : 16289955
aggregate         : 16289956
independent verify: 16289957
```

机器可读收据：`LIFELONG_SHARED_C_VINT_GH001_REPAIR_20260825.json`。截至提交时没有新
SR；只有最终 `independent_verification.json: verified=true` 才可报告 controller 结果。

## 9. 2026-08-25 17:07 队列更新

gh001 repair collection 与 factual-C seal 已完成。18 个 source histories 中只有 4 条 C
success，且全部来自 1 个 scene cluster；这是合法但极度 underpowered 的 shared-C B2
population。正式 paired job `16289955_[0-17%2]` 仍因
`QOSMaxGRESPerUser` pending，aggregate `16289956` 与 verifier `16289957` 等待依赖。

截至该快照没有任何 shared-C B2 arm 被执行，因此没有 ViNT 新 SR。即使后续 4 条均完成，
也只能作为同一场景中的机制/接口 pilot，不能替代 18-episode controller-portability
正式矩阵或支撑 controller-agnostic headline。
