# CEC 在线历史几何复用：主线与真机集成

日期：2026-09-07。用户同意将已完成的本机实验整理为正式实现，并加入最新真机更新。
不提交 HPC，不改论文数字，不启动或热切换真机。

## 主架构变化

同一因果 RGB 流进入 frozen LingBot。当前 depth 提供给 mono NavDP；同时保留各
可检索历史帧的 depth/confidence，未来 DINO / SP / LightGlue 选中 anchor 后直接
读取该帧几何进行 PnP、certificate 和原有 2.5 m bearing residual。

新增的是历史几何读写复用，不是 learned relocalizer，不新增训练、第二条几何流、
路由阈值或控制权。全历史缓存仍线性增长，本轮没有加入淘汰策略。

## 正式入口与旧版兼容

```bash
--certified_relocalization --certified_reference_depth_source online_history
```

- 新参数同时进入 MemNavAgent 构造器和 server CLI，不再依赖实验 monkey patch。
- online_history 自动按 stride=1 保存全部可检索帧，初次 certificate 使用保存的
  在线深度；若某帧缺失，不偷偷切换为 canonical 重放。
- 不能同时开启 eager 双流深度；不混入 stride=8 的旧长程 route 实验。
- `certified_relocalization_status.default_reference_depth_source` 和每个 proof 的
  `reference_depth_source` 均披露实际来源。
- 底层公共 API / CLI 缺省仍是 canonical，以免旧仿真命令无声改变。新的主线
  启动配置明确选择 online_history，而不是靠改变旧脚本的隐含默认值迁移结果。
- `MemNavData/run_realworld_memnav_server.sh` 新启动默认 online_history；显式
  `CEC_REFERENCE_DEPTH_SOURCE=canonical` 可复现旧路径。
- 旧实验参数 `route_sparse` 仍可由 Python 调用，避免损坏已冻结实验脚本；新部署
  面向用户只使用 canonical / online_history 两个名称。

## 已有证据

组件：4 histories／19 个跨视角／无匹配查询，9 Revisit 接受、10 Novel 拒绝，
两臂决策一致；9 个接受 bearing 最大差异 0.854°。

闭环：4 histories／8 natural queries／16 rollouts，SR 无新增 gain/loss，Revisit
3/4 对 3/4，Novel 0/4 对 0/4；Revisit SPL 0.66303 对 0.65403。首次 Revisit
证书阶段中位数 20.330 s 对 0.126 s，初始 CPU 历史深度占用约 397–815 MiB。

这不是严格数值等价，也不是正式泛化确认。旧论文主表仍属于原有重放实现，不能
把旧 SR / SPL 标成该新版本跑出来的。无须因此抹掉旧结果或立即重跑全部 HPC。

## 真机同步范围

`/home/asus/Research/MemNav-RealWorld` 的配置、RTX 启动脚本、常驻模型签名、
Hub 状态／proof 收据和文档同步到 `unitree-dog:/home/unitree/MemNav-RealWorld`。
新 resolved config 明确包含 `cec.historical_depth_source=online_history`。
旧 immutable config 缺该字段时仍按 canonical 解析，其原始文件不改写。

实际 LingBot / NavDP 仍运行在 RTX，本次缓存位于 RTX 主机 CPU 内存，不传到
Jetson，也不要求重新下载或重新拍摄 Survey。已有 sealed Survey 在新实验初始化时
按原流程重放 RGB，即可建立新在线几何存储；模型常驻不意味着跳过 Survey replay。

本次不修改 Jetson 速度、转身、重规划、到达或急停逻辑。当前常驻服务不热更新；
下一次用户正常启动时，根据签名和配置判断是否需要重新加载模型。

部署侧完整说明：`MemNav-RealWorld/CEC_ONLINE_HISTORY_UPDATE_20260907_CN.md`。

## 正式入口验证

`check_cec_online_history_deployment.py` 用当前真机仓库 wrappers、Hub 和两个
私有模型服务回放 240 帧已有 RGB，实际执行到 NavDP 生成轨迹为止，不连接执行器。
source=online_history、CEC accept、历史几何重放 0 帧、轨迹 `[1,24,3]` 且有限。
233 帧历史深度占 500,155,936 bytes，release 后清为 0。单次 GPU Hub 调用约
0.931 s，certificate 143.5 ms；不能当作真机全链路延迟或新增导航 SR。

两端语法检查与 Jetson 配置解析通过；旧配置导出仍为 canonical。同步前对比原文件
SHA，并保留两端备份；不复制或覆盖 Jetson 独有的未提交控制器改动。
