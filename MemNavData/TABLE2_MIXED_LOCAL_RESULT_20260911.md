# Table II 双角色分支：本机实际导航进展

日期：2026-09-11。本机小测已完成并独立复核；正式总体尚未冻结，不替换论文 Table II。

## 最新结论

新的 **实际 A→Revisit-B→{C-Novel,C-Revisit}** 分支已完成，
另一个 B-Novel 分支也完成了两臂对照，但因 native-B 失败，没有伪造后续 C。
本轮两个新 A 中一个成功；以下 B/C 查询均来自这一个成功 A、同一个场景。

| 阶段/查询 | Native 成功 / SPL | GEM 成功 / SPL | 配对增 / 损 |
|---|---|---|---|
| B-Novel | 0/1 / 0 | 0/1 / 0 | +0/−0 |
| B-Revisit | 1/1 / 0.2007 | 1/1 / 1.0000 | +0/−0 |
| C-Novel（接 native Revisit-B） | 1/1 / 1.0000 | 1/1 / 1.0000 | +0/−0 |
| C-Revisit（接 native Revisit-B） | 0/1 / 0 | 1/1 / 1.0000 | +1/−0 |

这里每行只有一个查询，并且共享父 A，不是四个独立场景。
C-Revisit 的 +1/−0 对应双侧 exact McNemar p=1，不能声称统计显著增益。
两组 Novel 的全部候选和实际动作都与 native 一致；
两次 Revisit 中看到一次路径效率改善和一次失败挽回，属于本机单例观察。

已修复两个**目标构造**问题：把目标距离限制错用到扰动前源帧，以及只看最早 6 帧导致遗漏 B 历史。
模型、在线证书、控制预算和评分规则未改。
完整 50/50 C 主表仍未得到；下一步应先审计共同距离/朝向供给，再冻结采集规模，
补足 native Novel-B 的成功来源，不应凭这一个场景启动未经设计的大数组。

## 1. 本轮实现了什么

新增 `table2_mixed_local.py`，复用已经修复的单目 NavDP/GEM、RGB 输入、
标准碰撞执行器与后向目标处理。新增的是实验编排，不是新导航方法：

1. A 用独立初始朝向和共同目标照片规则，从空历史实际执行。
2. B-Novel / B-Revisit 从相同 A 末态出发，各自做 native/GEM 配对。
3. 每个成功 native-B 只与其自己的 A 拼接，建立独立的 A+B 前缀。
4. C-Novel / C-Revisit 对完整实际 A+B 定义支持；C 运行前等额选择两种 B 来源。
5. 保存最后动作后的坐标，独立积分实际位移计算 SPL。

运行入口只接收目标图和共享前缀，不读取 Novel/Revisit 标签。
GT 几何仅用于目标构造、标准仿真碰撞与评分，不传给策略作为目标方向或落点修正。
这里仍是 1 m 位置成功判定，不是自主视觉 STOP 验证。

## 2. 新 A 的实际结果

预先固定两个已用于开发的场景，不因 A 失败换目标。

| 场景 | A 成功 | 动作数 | 初始最短路 / m | 实际路径 / m | SPL |
|---|---:|---:|---:|---:|---:|
| gxdoqLR6rwA | 0 | 600 | 5.069 | 17.512 | 0 |
| pLe4wQe7qrG | 1 | 122 | 2.238 | 2.363 | 0.9471 |

因此是 **1/2 实际 A 成功**，只够接口验证，不能用于估计整体方法 SR。
失败 A 留在采集分母；没有传送到其目标构造后续查询。
成功 A 的 122 帧均由实际运动记录重渲染，逐帧 JPEG 摘要一致。

原始目录：`.diagnostics/table2_mixed_actual_local_20260911_v2`。
独立复核：`verification_after_pose_index_recovery/independent_verification.json`，
`verified=true`。旧首轮 verifier 失败记录保留：当时完整单目位姿日志已保存，
但没有导出逐 rollout 索引。通过帧序号和 JPEG 摘要逐条匹配补回索引，
未重算网络、未改变任何轨迹或 SR。后续入口已增加正常导出。

另有 v1 在导航开始前因监督进程误导入 Habitat 依赖而退出；已改为轻量坐标常量导入，
不需要向模型环境额外安装仿真依赖。v1 不含导航结果。

## 3. 修复的是构造条件错用，不是在线阈值

原本把最终目标的 2–9 m 距离限制提前套在了历史源帧上。
但目标会在源帧附近平移、转动再渲染，两者距离并不相同。

已在上述成功 A 上找到实证：

| 项目 | 值 |
|---|---:|
| 历史源帧 | 39 |
| 源位置到当前末态最短路 | 1.866 m |
| 扰动后目标最短路 | 2.086 m |
| 源位置平移 | 0.22 m |
| 目标视角偏转 | 24° |
| 对合法历史最大共视 | 0.7069 |
| 与源 RGB 的像素 MAE | 46.50 |

此目标满足全部已定条件，不能因为源帧小于 2 m 而提前丢弃。
修复仅删除源帧距离过滤，**仍严格保留最终目标 2–9 m、共视 [0.55,0.90]、
扰动范围和像素差要求**，GEM 在线证书未变。

离线证据：`.diagnostics/table2_revisit_source_filter_20260911_v1/audit.json`。
此检查未读取 B/C 策略成功率。

## 4. 新 B/C 续跑

目录：`.diagnostics/table2_mixed_actual_local_20260911_v3`。
通过 `--resume-a-root` 引用 v2 的两条实际 A，**新采集 A 次数为 0**。
新目录保留原 A 成败，核对导航代码和权重一致，不覆盖旧结果。
原 B-Novel 的图像、位置、朝向和前缀摘要保持一致。

已在任何 B 导航前固定以下目标：

| 查询 | 最短路 | 相对方向类别 | 最大历史共视 |
|---|---:|---|---:|
| B-Novel | 6.478 m | rear | 0 |
| B-Revisit | 2.086 m | rear | 0.7069 |

**状态：四臂 B 配对全部完成，独立复核 `verified=true`，运行过程中源文件未变化。**

| 查询 | Native 成功 / SPL | GEM 成功 / SPL | 配对增 / 损 |
|---|---|---|---|
| B-Novel | 0/1 / 0 | 0/1 / 0 | +0/−0 |
| B-Revisit | 1/1 / 0.2007 | 1/1 / 1.0000 | +0/−0 |

- B-Novel：两臂实际路径均为 20.731 m，600 步后目标距离均为 3.189 m。
  GEM 的 75 次规划中接管为 0；75 组候选及 600 个实际动作精确一致，深度收据也通过核对。
- B-Revisit：native 358 步、10.392 m；GEM 64 步、1.052 m。
  两者均进入同一个 1 m 成功区域。效率差只来自一个查询，不是可外推的统计结论。
- 所有末态、初态、历史重放、单目输入及实际位移 SPL 均经过独立复核。

新正式规则下的 C 本次 **0 个入选前缀 / 0 次查询导航**：native Novel-B 失败；
native Revisit-B 成功，但当时小测仅取最早 6 个源帧，C 只构造出 Novel。
未把这批不足额来源混成正式 50/50 结果。

这两个目标距离不匹配，因此即使都跑完，也只能比较各自查询上的 native/GEM，
不能据此把 B-Novel 与 B-Revisit 的 SR 差归因于角色。

## 5. 单独归档的旧前缀 C 接口验证

为验证真实 A+B→C 接口，额外使用一条已经消费过的 actual-mono native A+B。
它不是 expert 轨迹，但 A 使用旧初始化规则，因此与第 2 节的新 A 分开记账。
没有用它填补新设计的 C 来源配额。

- 实际 A：105 帧；实际成功 native Novel-B：481 帧；合计 586 帧。
- 全部实际 RGB 的重渲染摘要一致；B 起点等于 A 的最后动作后末态。
- C-Revisit 可构造；在本机有限候选池中没有构造出对完整历史无支持的 C-Novel。
- 单个 C-Revisit 查询 native/GEM 严格配对：

| 方法 | 成功 | 动作数 | 实际路径 | SPL |
|---|---:|---:|---:|---:|
| native | 1/1 | 285 | 10.211 m | 0.3982 |
| GEM | 1/1 | 128 | 3.120 m | 1.0000 |

成功标签为 +0/−0。路径差仅是单例观察，不是总体效率增益。
SPL=1 采用既有 `success × L/max(L,P)` 定义和 1 m 成功半径，
不表示精确走到目标相机坐标，也不表示机器人自行决定停车。

目录：`.diagnostics/table2_consumed_prefix_probe_20260911_v1`；
`independent_verification.json` 为 `verified=true`。
明确标记 `formal_result=false`、`old_initialization=true`、不声称 C 来源 50/50。

## 6. 本机通过后，正式冻结前仍须完成的构造检查

- 在更完整的候选供给上确定各角色共同的距离分布与朝向组成，不能以 SR 筛目标。
- 已修复小测最多取 6 个早期历史源帧的限制。现在按固定时间间隔覆盖完整实际 A+B，
  不只看早期 A；支持来源依旧如实记 A-only/B-only/both/intermediate。
- C 必须有两种成功 native-B 来源且各自能构造两个 C 角色，才能实现事先约定的等额混合。
  供给不足应报告 attrition，而不是放宽 Novel 定义或用其他分支历史替代。
- 逐阶段配对与 GEM 自主 A→B→C joint 不同，最终表头和正文不得混写。

## 7. 测试与资源边界

`test_table2_mixed_local.py`、`test_table2_novel_sampling.py`、
`test_repaired_fullmono_local.py`、`test_simulation_authority_boundary.py`：
**58 passed**（B 完成前为 57，之后新增 C 候选必须覆盖 B 的回归测试）。
包括历史隔离、末态衔接、禁止 GEM-B 替代 native-B、支持来源、
角色信息隔离、单目输入和固定执行参数。

所有测试使用本机独立端口 19781/19782；保留真机驻留服务，不操作机器人。
本轮没有新 HPC 提交，没有修改论文，也没有 commit/push。

## 8. 新实际 A→Revisit-B→C 的完整分支接口验证

目录：`.diagnostics/table2_new_actual_c_interface_20260911_v1`。
独立于 v3 的 balanced-C 选择器，固定标记非正式、非 50/50；
它不是旧初始化前缀，也不是新生成 expert 轨迹。

复用 v3 已通过审计的实际 A（122 帧）+ 成功 native Revisit-B（358 帧），
共 480 帧。不重新执行 A/B；只增加 C 的两角色、两方法小测。

修复源帧时间覆盖后，在任何 C 导航前固定：

| C 查询 | 初始最短路 | 朝向类别 | 最大 A 支持 | 最大 B 支持 | 源帧 |
|---|---:|---|---:|---:|---:|
| Novel | 5.207 m | front | 0 | 0 | — |
| Revisit | 2.079 m | rear | 0.7981 | 0.8628 | 135 |

源帧 135 已属于实际 B，证明最早 6 帧的限制确实遗漏了合规候选。
但这个目标同时被 A 和 B 看见，支持类型是 **both**，不能冒充 B-only 新记忆。
原 C-Novel 图像、目标位姿和前缀摘要保持不变。

**当前状态：四次 C 导航全部完成，独立复核 `verified=true`，运行过程中源文件未变化。**

| 查询/方法 | 成功 | 动作数 | 实际路径 / m | 最终目标距离 / m | SPL |
|---|---:|---:|---:|---:|---:|
| C-Novel / native | 1 | 122 | 4.5663 | 0.9705 | 1.0000 |
| C-Novel / GEM | 1 | 122 | 4.5663 | 0.9705 | 1.0000 |
| C-Revisit / native | 0 | 324 | 10.0238 | 6.5869 | 0 |
| C-Revisit / GEM | 1 | 66 | 1.1061 | 0.9731 | 1.0000 |

native C-Revisit 按原实现的 `stuck` 规则终止，`blocked_step_count=21`；
不是输入回执异常，不是新增加的终止条件，也不是把未跑满 600 步误记成超时。
C-Novel 的 16 组规划候选和 122 个实际动作精确一致，GEM 接管为 0。
对两组 Novel 的逐次深度图、图像与 first-40 尺度收据也做了配对核查。
所有目标和 C 的选择均在读取 C 导航结果前固定。

完整时间采样下，当前可构造 C 双角色的参考来源为 Novel-B 0 条、Revisit-B 1 条；
因此等额 C 正式总体仍为 0。此小测只证明一条真实新初始化分支能够运行，
不能写作四种 B→C 组合已经完成，更不能写作 GEM 自主 A→B→C joint。
两角色仍有距离/方向差异，因此本接口小测不承担角色难度比较。

本次专用 19781/19782 服务已正常关闭。18888/8888 的真机驻留服务及其原 PID 保持不变。

## 9. 运行入口与结果索引

- 新 A/B/C 编排：`MemNavData/table2_mixed_local.py`。
- 单条实际 A+B 的 C 接口检查：`MemNavData/table2_consumed_prefix_probe.py`。
- 构造与范围约定：`MemNavData/TABLE2_MIXED_LOCAL_PROTOCOL_20260911.md`。
- 新 A 原记录：`.diagnostics/table2_mixed_actual_local_20260911_v2/summary.json`。
- A+B 配对及历史复核：`.diagnostics/table2_mixed_actual_local_20260911_v3/verification/independent_verification.json`。
- 新初始化 C 分支复核：`.diagnostics/table2_new_actual_c_interface_20260911_v1/independent_verification.json`。
- 旧初始化 C 单例：`.diagnostics/table2_consumed_prefix_probe_20260911_v1/independent_verification.json`，仅单独记账。

## 10. 后续 HPC 编排（2026-09-11 07:35）

已完成独立 source ID、全局 C 来源平衡及跨节点历史恢复的 HPC 编排。
91 项本机回归测试和远端 exact-container 预检通过后，提交了 4 个 HM3D 场景、
8 个新实际 A 起点的有限 pilot：`17347454/17347455/17347456/17347457`。
提交后首项等待 `QOSGrpGRES`，其余等待依赖，尚无本轮新 SR。
这不是正式新 Table II，也没有改变上面的本机结果或 controller。
详细状态、版本、路径与协议见 `TABLE2_MIXED_HPC_PREPARATION_20260911.md`。
