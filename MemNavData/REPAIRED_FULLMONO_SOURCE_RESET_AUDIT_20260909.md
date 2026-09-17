# 四条实际 A 的状态重置与因果尺度复查

2026-09-09。只读检查已完成的 `17253177_0` 原始归档；没有重跑导航，
没有改变已提交 source prefix 或任何方法参数。

检查对象：同一个私有服务进程中顺序执行的四条 A。关注它们是否重新建立各自状态，
而不是将上一条的尺度或历史继续用于下一条独立任务。

| episode | 规划深度收据数 | 首帧 index / 已观察帧数 | 第一次激活尺度的 index | 冻结 scale_hat |
|---|---:|---:|---:|---:|
| 0000 | 16 | 0 / 1 | 40 | 2.0931870032710864 |
| 0001 | 7 | 0 / 1 | 40 | 2.8686062894114106 |
| 0002 | 39 | 0 / 1 | 40 | 2.63765693122211 |
| 0003 | 48 | 0 / 1 | 40 | 2.174045630870824 |

四条各自的前五次规划（index 0/8/16/24/32）均为 `bootstrap_zero_depth`；
四次起始图像SHA不同。index 40起各自只出现一个不变的scale receipt SHA。
每条 receipt 均注明：scale prefix 为0..39、冻结于40次观测之后、相机高度0.5m、
`whole_episode_ground_cache_consumed=false`。没有把“frame 40”误当成第40张图：这里是零基索引，
首个使用冻结尺度的规划对应已观察41帧。

原底座 `repaired_fullmono_c8cf8c60e7efd55f` 的 `policy_agent.reset` 确实将帧计数、
scale K/V、首40尺度收据与图像/几何缓存清空，并调用 LingBot 主干与camera head的
`clean_kv_cache()`。因此源码及本批实际收据相符，未见四条A之间沿用前一条状态的证据。
原逐动作 verifier 另外已对每条独立检查连续的frame indices、当前RGB哈希和深度栅格转换。

尺度数值不是简单地将0.5除以原始地面高度：既有底座使用固定
`GROUND_BIAS_CORRECTION=1.15`，本批四条均未触发scale clamp。
该系数不是本轮增加或调出的参数；本次不更改。论文当前只定义一次估计的
`s_hat_40`，没有声称采用无修正的纯高度比公式，故这里不作论文公式修改。

这个复查支持本批状态重置/尺度因果性正常；不能由四条任务推断尺度准确性，
也不能将两条A失败归因于地面估计或其他单一原因。

## 证据

- 归档：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_fullmono_actual_a_20260909/complete_scene_development_3e90959723d4b66b/scene_0/artifacts.tar.gz`
- 归档SHA：`095c5a81d54978d0821797f1a608e029e3aff225f6048fb36d74bb316b405e37`
- 对象：`task/goal_a/rJhMRvNn4DS/episode_000{0,1,2,3}/navdp_http_receipts.jsonl`
- 既有A独立verifier SHA：`5a2aebb438567d4b0f0193de741b048d911e84fd2e793d29d0597edcb3c6fc7a`
- 四个尺度receipt SHA，按原episode顺序：
  - `b4c1c909a762971f68ef206f3a503f980ea80799388ddedbff190e80b4ec6b87`
  - `fd277e043e9da39e7e3a331fd27ef3eba7a3e5bb0be4ae6fd8e7d9b0f27009c2`
  - `80a9163c0d1e795ddc1acefcc5164b4f0b495c89bb391d7cc051b1457f948849`
  - `382d1e79bc28568869744be2b6e1cb7833f7599a8cdcd24a90ba39025441aa2d`

## 新查询的历史重放边界（静态追踪与首项实际核验）

重放入口为`eval_shared_online_role_pairs.replay_prefix` →
`shared_online_double_revisit_runtime.replay_online_a`。它读取新A保存的实际RGB，
逐帧校验图像SHA；全部物理观测进入几何历史，原规划观测进入NavDP FIFO，重放时不抽样diffusion。
该重放路径不读取或发送GT深度图。

不能仅看底层`replay_online_a`或`base.srv_memory`的默认实现判断最终模型输入：
它们会构造旧的执行位移诊断字段，但本次实际调用前还有封存的运行包装器。
`repaired_role_queries_cb812054dc470e53/run_habitat_minimal_repair_local.py`
（SHA `0f1c56cc15ad0be9d2667d9024af14db2f25c5ca9a2865a2103e651efeddda0e`）
在任何重放/查询开始之前替换HTTP边界；固定`MINIMAL_FRONT_GOAL=heading_on`时，
`rgb_only_memory_form`从发往memnav的表单剥离六个执行里程计字段：
`executed_translation_m`、`executed_yaw_rad`、`executed_forward_m`、`executed_left_m`、
`executor_local_se2_source`、`executor_local_se2_contract`。

这项处理同时覆盖重放和在线查询，不依赖`query_active=true`。
边界日志保留original/sent/removed字段名；因此日志中出现原字段名不表示模型收到了这些值。
固定endpoint bearing另由LingBot估计的当前pose与验证目标pose形成，不使用动作坐标路线分支。

现有独立verifier要求每次memnav请求均无上述字段、LingBot逐帧`motion_receipt_recorded`
均为空、重放图像匹配且diffusion重采样数为0。`17262832_0`现已完成并通过这些检查；
又直接流式读取其原始归档复查全部HTTP边界与几何帧，结果一致。
三臂各有101帧A重放；六个执行字段未发送，全部几何帧的motion receipt为空。
详见 [首项实际结果](REPAIRED_FULLMONO_QUERY_GATE_RESULT_20260909.md)。

理想仿真状态仍用于轨迹跟踪、碰撞和到达评分，不能由“模型边界无GT”外推为整个模拟器无GT。

## 查询期间的检索历史边界

再次读取实际封存底座中的`policy_agent.py`（SHA
`ff01cc12f5484ce4b31bfdf76f859abe426b4870c7566d287087ae657d8e652f`）：
目标首次注册时记录`goal_start_frame`，候选上界不超过该帧减一。
CEC候选与一般检索均应用这一上界，不把追逐当前目标时新采集的画面重新纳入
本次目标的历史候选。新RGB仍进入因果几何状态，但“写入状态”不等于
“可用于当前目标的历史检索”。这是现有代码边界，本轮没有新增冻结或修改检索行为。
