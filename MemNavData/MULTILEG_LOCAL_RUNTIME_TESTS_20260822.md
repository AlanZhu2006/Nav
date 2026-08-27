# Multi-leg / 真机运行时本机测试报告(2026-08-22)

本机(48GB RTX)真栈(真实 LingBot + NavDP 权重 + protocol-v4 hub),
数据为 fresh HM3D `eval_5/ep_0005` 的 756 帧真实 episode 序列。
五项测试,全部对活服务器,非 mock。

## 1. Eager depth cache 基准(新 goal 首证的核心权衡)

| 模式 | ingest | 首证(300 帧记忆,新 goal) | VRAM |
|---|---:|---:|---:|
| lazy | 231 ms/帧 | 26.1 s(anchor 在半程)| ~1.3 MB/帧,平坦 |
| eager | 418 ms/帧 | **1.09 s**(reloc 135 ms)| **~62 MB/帧,线性** |

eager 把首证做到 24×加速,但 62 MB/帧意味着 613 帧行走 ≈ 45 GB——
48 GB 卡上没有余量。**部署建议:≤500 帧任务开 eager;更长任务用 lazy
或等检查点方案(见 §5)。**

## 2. k=5 goal-session 真栈验证 + 首证延迟-记忆曲线(lazy)

5 个 goal 铺满 750 帧,会话 1→5、ceiling 逐会话正确、anchor 全部命中
目标帧 ±4、跨切换 NavDP 暖启动+队列校验全过,750 帧无内存泄漏。

**关键曲线:lazy 首证延迟 ≈ 0.198 s × anchor 原始帧号**(16.4s@150 →
45.6@300 → 75.2@450 → 105.0@600 → 135.2@750,五点近乎完美线性)。
Final14 的 anchor-latency 相关(r=.817)现在有了干净的斜率。真机 613 帧
行走的晚期目标,lazy 模式下机器人要静止约 2 分钟。跨会话复用同一 goal
(C2 类)因 anchor-depth 缓存只要 **0.58 s**。

## 3. RoPE 外推质量探针(flow-gate off,位置顶到 750)

强制 750 帧全部提交(RoPE 位置 = 原始帧号),同 5 个 goal:
**5/5 certificate accept,anchor 全部精确命中**,与门控对照无差别;
VRAM 与门控模式相同(sliding window 使逐帧 KV 有界)。
**结论:LingBot 在 RoPE 位置 ≥750 无质量退化,5-leg 步预算的外推顾虑
解除**(门控下 2500 步也只用几百位置)。回放延迟与门控无关——
anchor 深度回放按原始帧数计费,这解释了 §2 斜率。

## 4. 事务竞态复现尝试(task-36 类,hub 路径)

同 SHA 帧连发 ×5、同/异帧交替、15 连发零间隔:全部 200,无
"different transaction"。**hub 路径不可复现;维持瞬态判定**,最终以
HPC retry2(16151299)是否复现为准。

## 5. 发现的优化设计(待议,未实现)

lazy 与 eager 之间存在明确甜点:**每 N 帧保存一个流状态检查点,
anchor 回放从最近检查点起步**。N=32 时首证上界 ≈7 s(与行走长度
无关),VRAM ≈62MB×(750/32) ≈ 1.5 GB。属冻结邻接改动,需按 Final14
既定路线做 decision-equivalence 验证(proposal/certificate/bearing/
动作零变化)后才可用于正式实验。

## 6. 顺手修复

- realworld hub 补透传 `cec_goal_session_index / _started /
  goal_start_frame / candidate_ceiling`(与 portability hub 对齐,
  17/17 测试过);
- 运维教训:`pkill -f` 会自匹配调用方 shell 的命令行(本会话两次
  exit 144 的元凶),模式加括号如 `"[m]emnav_server.py"`,且 kill 与
  含明文路径的启动命令必须分开执行。
