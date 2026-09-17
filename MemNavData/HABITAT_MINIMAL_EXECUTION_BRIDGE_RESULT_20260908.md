# 本机最小执行修复：四历史六臂完整结果

日期：2026-09-08。24/24 正常完成；独立 verifier 通过；全部 24 段第一视角录像已导出。
原运行源码/权重未变；本轮私有服务已在收尾时退出。

## 1. 核心结论

修复不是完全无影响。旧执行器下 native 为 1/4、CEC 为 3/4；有限参考跟踪与标准环境步进下，
分别为 0/4、2/4。进一步纠正 RGB 后，这四例的成功标签没有再变化。

这同时否定两种过早判断：不能说旧执行器没有影响，也不能说 CEC 的全部收益都是执行器造成。
修复后 CEC 相对 native 的两个 gain 病例仍是 gxdoq、yqst；但只有四个已消费历史，
不能用于总体等价、显著增益或免重跑证明。

## 2. 完整成功标签

所有列的分母都是同四条历史，不把 24 个运行当作 24 个独立 episode。

| 场景 | native 旧执行/BGR | native 新执行/BGR | native 新执行/RGB | CEC 旧执行/BGR | CEC 新执行/BGR | CEC 新执行/RGB |
|---|---:|---:|---:|---:|---:|---:|
| gxdoqLR6rwA | 0 | 0 | 0 | 1 | 1 | 1 |
| pLe4wQe7qrG | 0 | 0 | 0 | 0 | 0 | 0 |
| yqstnuAEVhm | 0 | 0 | 0 | 1 | 1 | 1 |
| mJXqzFtmKg4 | 1 | 0 | 0 | 1 | 0 | 0 |
| 合计 | 1/4 | 0/4 | 0/4 | 3/4 | 2/4 | 2/4 |

- 各策略“新执行/BGR 对旧执行/BGR”均 +0/−1，exact McNemar p=1.0。
- 新执行下 RGB 对 BGR：两策略均 +0/−0，p=1.0。
- 每种执行/颜色配置中 CEC 对 native 均 +2/−0，p=0.5。

非显著不等于等价。这里最有价值的是逐动作因果链，而不是 N=4 的 p 值。

## 3. 改变结果的机制

### 零参考执行：有直接配对证据

mJX 的 CEC 两臂，第一次动作前状态与完整参考路径完全相同。
旧 PP 从零 XY 参考发出约 1.947 cm 实际移动，后续恢复非零轨迹并到达；
新执行器持有零参考，151 tick 实际位移为零，按原停滞规则退出。
首个不同动作就是 tick 0，不是模型随机误差累积。

mJX-native 也受影响：前 56 tick 状态一致，tick 56 遇到零参考；旧 PP 继续移动约 1.829 cm，
新执行器保持。此后路径分开，旧臂最终到达，新臂停滞。这不是只影响 CEC 的公共接口问题，
但也不能认为它在所有策略上的暴露比例相同。

pLe-CEC 同样在 tick 0 出现零参考下的约 3.729 cm 隐式移动；旧臂后来走 3.233 m 仍失败，
新臂完全保持也失败。该例初始认证 bearing 对 GT 直线目标的离线误差约 0.2303°，
后方 PointGoal 被 frozen NavDP 裁成侧向短目标；不能先归咎于 LingBot 定位漂移。

### NavMesh API：有路径影响，不都改变成功标签

相同前态、相同参考路径下，pLe-native 的 tick 53 首次因旧 snap 与标准 try_step 产生
1.942 cm 位置差；yqst-native 的 tick 304 为 0.719 cm。后续路径明显分开，但这两例均失败。
因此不能从同样失败推导执行接口无影响。

### RGB：输入错误已证实，SR 变化未证实

正确颜色改变了轨迹和路程，但本 N=4 没有改变成功标签。gxdoq-CEC 从 117 tick/3.995 m
变为 115 tick/3.905 m；yqst-CEC 从 115 tick/3.575 m 变为 106 tick/3.651 m。
不按这几例的好坏选择保留正确或错误颜色。

## 4. 核验与可视化

独立核验从动作前后坐标重算实际路程、最终距离和 SPL，包括最后动作；
重新加载当次 NavMesh 复算每个环境步；检查共享 A、首张 RGB、同色首条预测、
每 8 tick 规划和实际模型输入收据。不是只复读汇总 SR。

已查看 mJX 成功/失败臂的第一视角终帧：新 CEC 与初始窗前画面保持一致，与零位移日志对应；
旧 CEC 终帧已朝向室内通道。native 新执行终帧靠近局部表面。图像只辅助核对位置/朝向变化，
不由此推定精确接触点或所有深度误差。

根目录（绝对路径）：

`/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_minimal_repair_20260908/bridge_v1`

- `summary.json`：全部终局。
- `independent_verification.json`：`verified=true`。
- `first_person/render_receipt.json`：24 视频、逐帧记录、SHA。
- `first_person/mJXqzFtmKg4__cec__legacy_snap__legacy_bgr.mp4`：旧 CEC 到达。
- `first_person/mJXqzFtmKg4__cec__bounded_standard__legacy_bgr.mp4`：新 CEC 零参考停滞。
- `evaluation/*/*/full_plan_outputs.jsonl`：完整候选和实际所选轨迹。
- `evaluation/*/*/executor_actions.jsonl`：命令、环境响应与真实前后坐标。

视频播放为每 tick 0.1 s 的模拟时间，不包含推理等待；不是部署延迟演示。

## 5. 范围和后续

这是已有 actual metric-NavDP-A 历史上的 mono Revisit 查询诊断，不是新 full-mono-A、
Novel safety 或论文正式确认。旧深度仍保持 LingBot 方形栅格；独立 raster 修复试验随后进行。
新执行使用标准 NavMesh 碰撞近似和理想状态反馈，不声称无 GT 物理仿真或自主 STOP。

生产默认与论文旧数字未修改。后向目标的输入域适配尚未自动并入；不把修复后停住说成问题全部解决。
重跑依赖和最小架构见 `HABITAT_REPAIR_RERUN_SCOPE_20260908.md`。
