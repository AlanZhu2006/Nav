# 修复版 actual mono-A → mixed-role：本机端到端结果

2026-09-08。导航、独立复算、全部第一视角视频已完成。HPC 未提交，生产默认、论文、真机未改。

## 1. 结论

**修复版可以从新执行的 mono-A 建立历史，再完成同历史的 native / raw / CEC 查询。**
两条固定源任务中 A 成功 1 条，只构造出 **1 个历史、1 个场景、2 个角色查询、6 个方法运行**。
该历史上 CEC 的 Novel/Revisit 都成功；native 只成功 Novel，raw 只成功 Revisit。
这是完整链的集成验证，不是统计确认，不能据此宣布新版正式 SR 或替换论文表格。

与此前四历史桥接最关键的区别：**A 也重新执行且只向模型提供单目深度，不再重放旧 metric-A。**
失败 A 没有被替换、补采或排除在 source 总账之外。

## 2. 配置与构造

- 协议：[REPAIRED_FULLMONO_LOCAL_PROTOCOL_20260908.md](REPAIRED_FULLMONO_LOCAL_PROTOCOL_20260908.md)。
- 固定 `gxdoqLR6rwA/episode_0000`、`pLe4wQe7qrG/episode_0000` 两条已消费 MP3D source。
  只复用源起点、Goal-A 图和场景；新 A 由策略执行，不用 expert/旧 A 位姿作为行动路径。
- 先执行完两条 A，再构造全部查询，最后跑 query；没有按 query SR 增补 source。
- 共有修复：正确 RGB、单目深度逆 padding 到 RGB 栅格、bounded pursuit、一次标准 Habitat `try_step`。
  raw/CEC 共用后向 PointGoal 的真实转身适配，转身每步写入新 RGB，转后重新规划。
- 原 first40 高度尺度、2.5 m residual、证书、8-tick 重规划、600-tick 上限、1 m 成功半径未改。
  CEC 稀疏 reference depth 使用 `canonical`；没有同时切换成另一版缓存或 learned 定位。
- Novel/Revisit 只在构造与 evaluator 可见，模型请求不读 role、GT 目标、最短路或 simulator odometry。
- 本轮保持旧构造器相机元数据，未暗改已知约 1.16% 的 FY 差异；正式包的 K 版本须单独明确。

## 3. 新 A 和构造损耗

| source scene | A 成功 | 动作数 | 实际平移路程 | 最终距 A | 后续历史 |
|---|---:|---:|---:|---:|---:|
| gxdoqLR6rwA | 0 | 371 | 8.416 m | 1.222 m | 0，`mono_a_failed` |
| pLe4wQe7qrG | 1 | 105 | 3.912 m | 0.968 m | 1 |

gx 的 147 个动作收到零 XY 参考，跟踪器保持不动，最后按原停滞规则结束。
最后一次 critic 为 −0.03556，并非低于 −0.5 后的搜索回退；这是能直接确定的执行机制。
源 Goal-A 图位置与计分位置仅差约 0.3 mm，不能将 1.222 m 末距归因于二者明显错位。
**尚不能仅凭这一例确定 frozen NavDP 为什么提前输出零轨迹**，也不能将全部变化归因于 tracker。
本轮没有为救该例放宽成功半径、取消上游输出处理或恢复“零参考自行前进”。

pLe 新历史生成的查询：

- Novel：初始测地距离 7.286 m，历史最大共视 0，按冻结构造顺序属于 rear 分层。
- Revisit：初始测地距离 2.245 m，历史最大共视 0.718；源 frame39 附近偏移 0.22 m、旋转 18°。
- 构造器同时计算出的 hard-support 视图没有运行，不按观测结果追加其他任务。

## 4. 六臂完整查询结果

全部共享同一条新 mono-A、同一 query 起点；各 role 内目标也相同。运行顺序为 raw → CEC → native。

| Query | Controller | 成功 | 动作数 | 实际平移路程 | 最终距离 | SPL | 显式转身动作 |
|---|---|---:|---:|---:|---:|---:|---:|
| Novel | native | 1 | 481 | 16.829 m | 0.972 m | 0.43291 | 0 |
| Novel | raw fixed | 0 | 275 | 2.427 m | 5.488 m | 0 | 54 |
| Novel | CEC | 1 | 481 | 16.829 m | 0.972 m | 0.43291 | 0 |
| Revisit | native | 0 | 600 | 19.612 m | 2.573 m | 0 | 0 |
| Revisit | raw fixed | 1 | 71 | 1.313 m | 0.990 m | 1 | 36 |
| Revisit | CEC | 1 | 69 | 1.277 m | 0.994 m | 1 | 35 |

CEC vs native：两个角色查询合计 +1/−0；CEC vs raw 同样 +1/−0。
它们来自**一个历史/场景**，不是两次独立场景验证；不把 6 个臂当作 N=6。
Revisit 上 raw 与 CEC 都成功，不能据此称 CEC 更强的定位带来了额外 SR。

SPL 由初始测地距离和实际动作前后 XY 位移重算，包含最后一步的真实落点。
Revisit 在进入 1 m 成功半径时结束，实际路程可以短于初始测地距离；SPL 截断为 1 合乎原定义。
但原地转身不增加平移路程，因此 SPL=1 不代表没有动作/时间代价。

## 5. 接管、转身和不接管等价

- **Novel：**CEC 61 次规划均未接管。与 native 的 61 组完整候选、critic、输入图像、
  单目深度输入/输出哈希和 481 个实际动作逐一完全一致。不是只比较成功标签或短前缀。
- **Revisit：**CEC 6 次规划均接管。初始 PointGoal 在后方，真实转身约 157.27° 后重新规划并前进。
  raw 也走相同适配接口，约 157.62°；没有赋予 CEC 独有的低层转向能力。
- 转身真实平移为 0，但 LingBot 在固定高度尺度下估出约 0.111 m（CEC）/0.106 m（raw）平移，
  yaw 差约 2.69°/3.04°。这条仍成功，**并不表示高度尺度消除了纯旋转漂移**。
- raw 在 unsupported Novel 上持续使用历史方向，出现两段显式重朝向和 15 次低 critic 回退；
  CEC 未干扰 native。这是本例直接观测，不推广为所有 Novel 都有相同现象。

## 6. 独立检查与收尾修复

- 2 A + 6 query 共 8 次运行全部完成，合计 2,453 个实际动作、301 次规划深度数组。
- 独立复算动作前后链、末端坐标、SR/SPL、bounded steering、标准 `try_step`、图像/深度输入，
  并核对新 A 的采集、复制、构造与各 arm replay 身份。
- 所有 runtime 代码在导航期间没有改变；原 source snapshot 和所有旧运行输出保留。
- 初次收尾 verifier 因旧 query auditor 要求非空 manifest 而退出，**不是导航失败**。
  修复仅区分“采集失败的空 population”和“可运行查询 manifest”，验证失败 A 的哈希与 attrition 后
  保留空结果；非空查询仍执行原完整 auditor。另加入完整不接管深度哈希配对检查。
- 修复后的 verifier 已通过；新增 5 个空构造回归用例，覆盖合法失败、无可构造几何、丢失 attrition、
  错误 manifest 哈希、错误 retained 数量。统一预检 **162 passed / 12 Pyparsing 弃用警告**。
- 初次 `failure.json`、`logs/verification.log` 保留，修复后的结果见 `independent_verification.json`；
  最终状态见 `completion_receipt.json`，不把旧失败日志伪装成从未发生。

8 段第一视角视频均从实际保存的姿态逐帧重渲染，含末动作后终点；首帧 RGB 哈希逐一吻合。
播放为 **10 FPS 的 control ticks，不是实测实时帧率**；没有重新运行策略。
已检查 CEC Revisit 的转身/推进画面以及失败 A 的末段停滞画面。

本轮 8 次 rollout 的进程 wall time 合计约 13.73 分钟，包含审计与重复 A replay，不含全部启动/构造/导出开销。
机器期间也有另一真机工作区服务使用 GPU，不能把这组 wall time 当部署延迟或 HPC 时长承诺。

## 7. 仍然明确的边界

1. 一条成功历史不足以判断总体增益、A 成功率或正式查询构造率；两个 source 都是已消费场景。
2. 自定义落点吸附与 30% 缩步重试不再控制运动，但标准 `try_step` **仍使用 NavMesh**，
   低层跟踪仍使用理想位姿。不是无 GT 环境、刚体物理或 Go2 机体避障确认。
3. 成功仍为 evaluator GT 平面距离 <1 m，不是自主视觉 STOP，也不要求末端朝向。
4. 显式真实朝向适配改变执行接口；不能称所有运动都由 diffusion 输出，更不能混用新旧执行版论文数字。
5. 本轮没有新增调参、训练、graph rescue、运行时失败门或 HPC 正式结果。

## 8. 下一步

本机端到端检查已通过，下一步是将同一修复版打包成可迁移的 HM3D 核心评测入口，
先在实际 HPC 环境执行一个完整 source/role-arm 单元验证依赖与耗时，再提交新 A 与三臂的正式批次。
正式包须明确真实 K 的来源与构造版本、源清单/构造率、canonical reference depth 版本及对照组共同接口。
不只补旧失败案例，也不直接拿新版 CEC 与旧 native 相减。

准备方案：[REPAIRED_HM3D_CORE_EVAL_PLAN_20260908.md](REPAIRED_HM3D_CORE_EVAL_PLAN_20260908.md)。

## 9. 路径

完整原始结果：

```
/home/asus/Research/Nav-graph-blind/.diagnostics/repaired_fullmono_local_20260908/e2e_v1/
```

其下 `goal_a/`、`construction/`、`evaluation/`、`source_snapshot/` 保留全链收据；
`first_person/` 包含全部 8 段成功/失败视频。

最直接的 Revisit 对照：

```
first_person/pLe4wQe7qrG__revisit__native.mp4
first_person/pLe4wQe7qrG__revisit__raw_fixed.mp4
first_person/pLe4wQe7qrG__revisit__cec.mp4
```

运行入口 `MemNavData/run_repaired_fullmono_local.py`；独立复算
`MemNavData/verify_repaired_fullmono_local.py`；测试 `MemNavData/test_repaired_fullmono_local.py`。
本轮私有端口 21710/21711 服务均已退出，未停止或修改其他工作区服务，未 commit/push。
