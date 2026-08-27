# k 段 multi-leg 构造合约(设计稿,未冻结)

状态:DRAFT。目的:把 role-pair 构造推广到 k 个连续目标,修掉 strict-v4
真五段 smoke 暴露的两个结构性缺陷(expert-A 支持假设、早段 Novel 失败全链
删失),并把 lifelong NNR 扩样已验证的 factual-prefix 配对法一般化。

## 1. 两种估计量,分开设计,不混报

**E1 逐段归因(主)——factual-prefix 配对。**
先用冻结 native 策略把整条 k 段序列执行一次并封存(factual 轨迹)。对每个
待归因段 j:所有臂逐帧重放 factual 段 1..j-1,在段 j 分叉。产出 k-1 个
干净的配对对比,无删失、物理前缀逐位一致。代价:测不到 treatment 效应的
跨段复合。这是 NNR 扩样 C→B2→C2 设计的直接推广。

**E2 复合生存(副)——真实执行分叉。**
各臂从段 1 起真实执行。主指标:prefix survival(完成 ≥1/2/…/k 段的
episode 比例)、首败前完成段数、k 段 joint SR/SPL。条件 SR 只作诊断。
E2 不做配对显著性主张(首次分叉后轨迹不可配对),只报生存曲线与
scene-cluster CI。

## 2. 每段的 role 合约(相对"实际在线历史"定义,不看生成期资产)

role 串示例:`N-N-R-G-R`(N=Novel,R=Revisit,G=goal-return 重复目标)。

- **R(j)**:段 j 的目标图必须被段 1..j-1 的 factual 在线 RGB 支持——
  result-blind 预审计,复用 lifelong 扩样的 factual-B 口径
  (max covis >= 0.20 冻结分母,>= 0.50 单列 strong);
- **N(j)**:必须对段 1..j-1 的**全部累计历史**不支持(max covis 低于
  Novel 上限)。注意:k 越大,保持 Novel 越难——role 串在冻结前必须过
  一次可构造性审计,场景太小就降 k 或换 role 串,不允许事后放宽;
- **G(j)**:重复使用某个先前段的目标 JPEG,专测 goal-session 重开
  (B2/C2 类)。receipt 必须记录其"当时 vs 现在"的支持差;
- 所有 role 只用于事后分析,运行时零可见(与既有纪律一致)。

## 3. 已知机制约束(全部已核实,不再是未知数)

- 帧上限:flow-gate 使 RoPE 位置只按 committed keyframe 计,2500 步
  episode 实耗几百位置,2048 上限不构成约束;入口护栏已落
  (MULTILEG_FRAME_CAP_AUDIT_20260822.md);
- reset 的 `episode_len` 必须传 k 段总步数,让 FLOW_TIERS 取对档;
- goal-session 生命周期(重复目标重开会话)已修复并有逐动作 receipt,
  96 契约测试覆盖;
- `exclude_recent=32`:任何段的预算不得低于 40 步,否则下一段的 Revisit
  候选可能被 recent 排除窗掏空——写进构造检查;
- 首证延迟随历史增长(新 goal 最深 46s):E2 的 per-leg 步预算要给首证
  留余量,或在段边界静止等待首证(真机已是该语义)。

## 4. 指标(预注册,含效率 co-primary)

每个估计量同时报告:SR 类(逐段配对 gain/loss、McNemar、cluster CI;
E2 加 prefix survival)+ **效率 co-primary**:both-success 的 paired
steps / path / SPL——supported 段上 SR 会饱和,效率差(pilot 里 B2
−45% path)必须预注册,不做 post-hoc。

## 5. Population 定位

- 仿真:consumed NNR / strict-v4 资产,**机制证据**口径(与 2026-08-21
  fresh-scene 讨论一致,不升级 headline);
- 外部确认:真机连续任务(天然 fresh)。依赖项:v3 hub 需把
  `begin_revisit` 泛化为可重复的 goal-session 边界
  (`revisit_query → memory_recording` 回边),服务端机制已就绪。

## 6. 冻结前 TODO

1. 选定 k 与 role 串,跑可构造性审计(result-blind);
2. E1 的臂集合:all_prior / initial_leg_only / forced_reject_native
   (第三臂实现中);
3. 真机 hub 多会话扩展(小改动,见 §5);
4. 过一遍 HPC_HARDENING 三件套后再提交。
