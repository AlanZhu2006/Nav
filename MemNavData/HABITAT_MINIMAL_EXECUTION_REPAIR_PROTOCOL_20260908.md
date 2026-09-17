# Habitat 最小执行修复：今晚冻结方案（2026-09-08）

## 目标与范围

修复与审计现有 Habitat + frozen NavDP + CEC 的执行接口；不迁移 Isaac，不更换 X，
不调 CEC 阈值、2.5 m、模型权重或相机。不动真机/HPC/论文，不修改历史结果。
本轮只新增隔离模块/runner，生产 evaluator 和驻留服务的默认行为保持不变。

“完整审计”指本轮实际使用的 RGB→深度/历史→目标→NavDP→轨迹→执行→指标链路；
不意味着测试能证明没有任何未知 bug。每一项标明源码检查、单元验证或闭环验证的证据层级。

## 1. 当前已知问题与不变项

1. 原 PP 会在零平移参考下继续产生非零推进。
2. 原控制与环境碰撞耦合，含自定义 snap/30% 重试。
3. 新发现 base NavDP 的 PIL RGB JPEG 集成也有 RGB/BGR 差异；须独立做颜色对照。
4. 标准 Habitat try_step **仍使用 GT NavMesh 碰撞近似**；不是 GT-free 动力学。
   场景仅在环境层阻挡运动，不输入 bounded pursuit，不供模型选路。
5. 后向 PointGoal 的裁剪、低 critic 侧向搜索、短轨迹 mask 属于当前 frozen NavDP
   的输入/输出语义，不在这轮同时改动，也不把它们伪装成到达。

## 2. 最小候选修复

`bounded_pursuit.command(pose, reference, limits)` 不接受地图、角色或真目标。
保留原 lookahead/曲率/速度上限；命令长度不超过被跟踪参考点的距离，零参考保持原位/朝向。
1e-8 m 是浮点零容差，不是成功或证书门槛。短轨迹修正不保证跟踪任意回环/后向路径；
仍是原 forward pursuit 的有限步长版本，不声称 MPC 或障碍规划。

环境只执行一次 `try_step(start, requested_end)`，允许标准滑移；不额外 snap、缩步或搜方向。
不通过改碰撞包络或放宽物理倾斜阈值追求 SR。Bullet 圆柱支线暂不纳入本轮。

## 3. 冻结样本与六臂

使用既有 four-history manifest 的全部四条，按其既有顺序，不按本轮结果选择：
gxdoqLR6rwA、pLe4wQe7qrG、yqstnuAEVhm、mJXqzFtmKg4。
仅 natural Revisit 查询；这些都是已消费的机制诊断，不用于新显著性确认或 Novel safety 声明。

每历史六臂，偶数索引按下列顺序，奇数倒序：

1. native / legacy snap + legacy BGR
2. native / bounded standard collision + legacy BGR
3. native / bounded standard collision + corrected RGB
4. CEC / bounded standard collision + corrected RGB
5. CEC / bounded standard collision + legacy BGR
6. CEC / legacy snap + legacy BGR

前两种同色条件比较执行层整体变化；后两种同执行条件比较颜色变化。
逐动作同时保存原控制 + standard collision 的影子结果用于定位最初差异，
但不把影子一步结果冒充完整反事实 SR。

共享 actual-online metric-NavDP-A 历史；查询使用 LingBot 单目深度、canonical 参考深度。
不是重新生成的 full-mono-A，也不是 full-physics-A。
600 个动作 tick，8 tick 重规划，步长/角速度/相机/种子保持既有值；无新增恢复分支。
GT 平面距离 < 1 m 为 evaluator 到达定义；不是 autonomous STOP。
统计实际位移并保存最后动作后终点；异常退出不伪造失败或成功。

## 4. 执行顺序

1. 彩色实际客户端/解码/预处理测试，坐标与 zero/short/no-retry 命令测试。
2. 本机数据与渲染预检；相关深度交易、历史/角色隔离、SPL 与原有控制测试。
3. 独立短 smoke，只检查启动/回执/配对。短 smoke 不进入正式六臂总账。
4. 六臂 × 四历史连续本机评测，使用新的不可覆盖输出目录和私有服务。
5. 独立重算动作、路径、终点、成功、同色首次计划及共享历史，导出失败与成功视频。

若测试暴露影响数值的代码错误，先停止本次正式解释，保存失败尝试，以新版本/目录重新冻结；
不得修改正在运行的源码再混合前后结果。不同色不要求首次计划相同；同色只更换执行器需核对。

## 5. 投稿与重跑决策

N=4 只定位接口差异，不能凭 p>0.05 或相同总 SR 宣称等价。
旧表格仍属于旧明确执行协议，不能直接改名为修复版。
如果要在主表报告修复版本，需重新生成其实际 A 历史并补同协议 native/raw/CEC 主比较；
依赖执行器且受结论变化影响的闭环消融再跟进。离线检索/定位不因执行器变化自动重跑。
颜色问题对各历史表格的实际覆盖需核对冻结代码，不能仅据当前源码宣称全项目同样受影响。
