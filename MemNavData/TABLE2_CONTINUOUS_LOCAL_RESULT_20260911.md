# Table II 连续执行本机进度与构造发现

日期：2026-09-11，22:48 CST 状态截面。**开发小测，不是正式配对总体。**

## 1. 方法身份：本轮仍是严格面积证书

用户追问后，已直接检查新运行的原始收据，而不是根据方法简称推断：

- `certified_relocalization_authority_policy = strict_certificate`；
- query/reference 内点覆盖面积均要求至少 0.05；
- PnP 内点至少 16，重投影 RMSE 最多 2 px。

本轮使用修复后的输入与 controller，不代表使用无面积规则。
`certificate_without_coverage` 是已有显式消融选项，本次没有传入；其既有报告说明默认
方法仍为 strict。本轮不得写成“无面积 GEM”的结果，也不在运行中切换方法身份。
没有额外新增按房间面积判定目标的门槛。离线共视范围与上述 5% 内点凸包面积是不同量。

## 2. 连续执行入口已实现并真实运行

- `table2_continuous_chain.py`：自身末态传递、记忆帧连续、失败后未尝试、固定起始分母。
- `table2_continuous_local.py`：复用原修复版执行器；episode 只 reset 一次；不重放参考前缀。
- `verify_table2_continuous_local.py`：独立核对动作前后坐标、SPL、reset 次数、原 first40
  尺度收据，以及段间记忆连续性。GT 只在任务构造、环境碰撞和评分侧使用。
- 相关单元/回归测试 225 passed，其中新连续链 11 项。

本小测的后继目标从每臂自己的实际历史构造，因此不能自动声称两臂后续都是同目标。
正式共同目标协议尚未冻结，也没有提交 HPC。

## 3. 实际结果（截至上述状态截面）

| 开发来源 / 方法 | A | B | C | 含义 |
|---|---|---|---|---|
| 小场景 pLe4，native NRR | 成功，82 步 | 无合规目标 | 未尝试 | 构造阻断，不是 B 导航失败 |
| 声明源 0，native NRR | 成功，152 步 | 失败，600 步 | 未尝试 | 正确保留前段失败，未借用 GEM 状态 |
| 声明源 0，GEM NRR | 成功，152 步 | 成功，92 步 | 成功，36 步 | 真正自身历史连续三段完成 |
| 声明源 1，native NNN | 失败，600 步 | 未尝试 | 未尝试 | 前向构造不保证策略成功 |
| 声明源 1，GEM NNN | 失败，600 步 | 未尝试 | 未尝试 | 与 native 的 RGB/pose 逐项相同 |
| 补充源 0，native NNN | 成功，152 步 | 失败，600 步 | 未尝试 | 开发接口覆盖，不替换源 1 |
| 补充源 0，GEM NNN | 成功，152 步 | 失败，600 步 | 未尝试 | A/B 的 RGB/pose 均与 native 逐项相同 |
| 补充源 0，GEM NRN | 成功，152 步 | 成功，92 步 | 成功，287 步 | 回访后的 Novel 目标切换已实际完成 |

声明源 0、1 都来自原 18 场景声明的最前两个来源，场景为 `1LXtFkjw3qL`，分别
`episode_0000`、`episode_0001`。不是根据此次 SR 选择或替换来源。

源 0 的 GEM 记忆帧连续为 `0–151 → 152–243 → 244–279`。
一次 reset、零前缀重放、三个阶段使用同一个 first40 尺度收据，独立 verifier 通过。
两臂 A 的实际 RGB/pose 逐项一致，GEM A 接管为 0；这一事实仅限本例。

GEM B/C 的目标距离分别为 3.230 / 2.541 m，初始方向为 −157.5° / 7.9°。
C 对 A 的最大共视为 0.7038、对 B 仅 0.0797，满足 A-only 支持定义：
这条开发例确实调用了更早 A 的记忆，而非只依赖紧邻 B 的画面。
不能据单个场景例子推断三段总体 SR 或统计显著性。

四条预定运行已全部完成并各自独立复核。源 1 两臂均为实际路径 21.018847 m、
末态目标距离 9.493164 m；前向初始角 −28.3°没有保证本例成功。这不是第二个 Novel
的难度问题，因为本例根本没有进入 B。
随后增加的源 0 NNN 仅补 Novel 后段接口覆盖，选择依据与开发性质已在本机协议中预先说明。

### 补充 NNN 的新核对

- 两臂独立 verifier 都通过：一次 episode reset、零前缀重放、原尺度收据不变，
  A/B 记忆帧为 `0–151 → 152–751`，失败后没有执行 C。
- B 的测地距离 6.049818 m、起始路线角 −16.7374°、历史/当前共视均为 0。
  A 为 6.032420 m、−29.6218°；这两段在本例中同属 6–9 m 前向任务。
- NavDP 的 19 次 A 请求和 75 次 B 请求分别收到正确的新目标 SHA；B 没有继续使用 A 图。
- GEM A/B 接管次数均为 0，两个阶段的实际位置、yaw、RGB 哈希与 native 完全相同。
- B 的实际路径为 19.195020 m，最终距目标 11.193784 m，SPL 为 0。

这排除了本例中的错误目标图、段间重置及 GEM 干预解释，但不能凭单例证明 B 普遍比 A 难，
也不能声称统一前向后 SR 必须接近。尚未做策略内部的完整失败归因。
本例不额外更换目标、调低证书阈值或延长预算。

### 补充 NRN：回访后 Novel-C 成功

源 0 的 GEM NRN 已完整结束，独立 verifier 通过。它重新实际执行 A/B，不借用已存轨迹；
重执行 A/B 的 RGB/pose 与此前 NRR 完全相同，然后选择不同角色的 C 目标。

- C-Novel 初始测地距离 5.270857 m、路线角 7.8929°，对完整 A+B 历史及当前视图共视均为 0。
- C 用 287 ticks 成功到达，实际路径 5.006359 m、最终距离 0.995275 m、SPL 1.0。
- C 的 36 次规划均未由 GEM 接管；NavDP 收到的目标图 SHA 全部与新 C 一致。
- 全链只 reset 一次，记忆 `0–151 → 152–243 → 244–530` 连续，仍使用原 first40 收据。

因此“执行过 Revisit 就无法再切换到 Novel”不是必然的接口问题。不能据此推断 NNN/NRN
的总体差异，因为此例是已知 A/B 成功的开发来源，C 目标也不同。
本批不再为了凑出三个 Novel 全成功而补换来源。

## 4. 为什么不能搬用 native 的 C 菜单

另一条已消费的 pLe4 B-Revisit 配对中，两臂都成功，终点距离仅差 0.0216 m、yaw 差
20.40°，但 native B 观察了 358 帧，GEM B 只观察了 64 帧。
对原 native-C 的四个固定候选重新渲染并按各自真实 A+B 历史测量：

| 原 C 类型 | native 最大历史共视 | GEM 最大历史共视 | 结论 |
|---|---:|---:|---|
| Novel，2–4 m | 0.0589 | 0.1086 | GEM 侧不满足 Novel<0.10 |
| Novel，4–6 m | 0.0514 | 0.2531 | GEM 侧不满足 Novel<0.10 |
| Revisit，2–4 m | 0.7078 | 0.3191 | GEM 侧不满足本轮 supported-Revisit 范围 |
| Revisit，4–6 m | 0.7301 | 0.4615 | 同上 |

这不是 GEM 遗忘或定位失败的证据，是两条实际路径观察范围不同。
支持不足也不能直接改称 Novel；当前视图可见性还需要单独检查。

进一步本机供给诊断：使用原 24 个 Novel 空间候选与不超过 384 个 Revisit 视觉检查，
发现同时满足两臂几何/支持条件的候选：Novel 7、Revisit 56。
按原 materializer 的 JPEG 解码图和深度量化重新核对后计数相同，分别也是 7、56 个
不同目标图哈希；并非由重复图像堆出的计数。
这些是候选计数，不是独立 episode 或 SR；只说明“首个 native 合规目标”不等于
“没有双方合规目标”。原图/轨迹未改变，未放宽任何在线证书条件。
诊断使用 native 历史位姿扰动提案，不声称完成了对称、正式的共同目标构造。

## 5. 当前产物与下一步

目录：

`/home/asus/Research/Nav-graph-blind/.diagnostics/table2_continuous_local_20260911_v1/`

- `own_history_goal_audit/audit.json`：原四个固定 C 目标的两份历史复查。
- `common_goal_budget_probe/audit.json`：原有限预算内的共同目标供给诊断首轮。
- `common_goal_budget_probe_jpeg_checked/audit.json`：对齐 JPEG/深度存储后复核，计数不变。
- `source_000_nrr_native/independent_verification.json`：native 自身链复核。
- `source_000_nrr_cec/independent_verification.json`：GEM 三段连续完成复核。
- `source_001_nnn_native/independent_verification.json`：A 失败后的正确终止。
- `source_001_nnn_cec/independent_verification.json`：A 失败后的正确终止，已复核。
- `source_000_nnn_interface_native/independent_verification.json`：A 成功、B 失败，已复核。
- `source_000_nnn_interface_cec/independent_verification.json`：同上，两臂实际 A/B 一致。
- `source_000_nrn_interface_cec/independent_verification.json`：三段成功，Novel-C 无 GEM 接管。

七条声明来源接口运行均已完成，另有最早小场景的一条构造阻断。没有本批后台 GPU 任务；
私有服务已由原 runner 退出，不影响其他已有进程。相关回归测试再次运行，225 passed。

下一步仍是冻结共同任务的连续比较方式，明确实际历史支持与构造损耗的处理；
不能直接把本单臂在线构造小测扩大为同目标配对 Table II。
推荐与代价已整理为 `TABLE2_CONTINUOUS_COMMON_TASK_DRAFT_20260911.md`：两臂持续保留
自身状态，逐段共同发同一目标图。这是在线发题实验定义，需要确认，不等于预先固定三图。
用户已同意暂不处理面积条件，本轮继续 strict，不引入新的证书消融变量。

未修改旧 v2/v3 封存包、论文、controller 或证书默认参数；未 commit/push、未新增 HPC 作业。
