# 低共视本机闭环：启动与运行状态

2026-09-10，Asia/Shanghai。已消费数据上的机制开发，不是论文确认结果。

## 已完成

1. 固定两个真实低共视历史及配套 Novel：history 2、12，共 4 个查询、4 个 arm、16 条导航。
   Revisit 测地距离 3.084 / 2.492 m；Novel 6.482 / 6.412 m。本轮隔离低共视，不检验长距离。
2. 增加显式 CLI `--certified_authority_policy certificate_without_coverage`，默认仍为
   `strict_certificate`。请求和响应都保留真实策略身份；不是把新规则伪装为旧证书。
3. 复用修复后的 RGB/深度输入、bounded executor 和 front-goal bridge，不另写控制器。
   CLI 禁止把此次面积消融与 learned rescue、metric/route-tangent 等其他改动混用。
4. 52 项相关单元/回归测试通过，四臂 CLI 预检通过，`git diff --check` 通过。
5. 按共享 SSH 手册复用 yz11502 的连接，取回两场景 GLB、原 A RGB、trace、相机载体、
   四目标 RGB；474 个文件核验通过。没有取回目标真值深度，也没有提交 HPC GPU 作业。
   bundle 79,251,061 bytes，SHA256：
   `c095fb9edeaeeae1fbd2875a1aca8e11c221447748ae713ff11e6502eb4c1f24`。

## 首次启动遇到的资源冲突

`navigation_v1` 的两个私有服务启动成功，第一条 native 在重放 A 的第 51 帧前后
发生 CUDA OOM。准备期间另一项目的训练开始使用约 21.58 GiB，加上原真机常驻服务
约 7 GiB 和测试模型，显存耗尽。此时尚未执行 query 导航：完成 rollout=0，SR=无。

这不是方法失败，也没有缩短历史、减少候选、降低精度或停掉其他项目来绕过。
此次两个私有服务已由监督进程退出清理；原 8888/18888 真机服务未动，失败日志保留。

## 重启方式

`navigation_v2` 使用完全相同的 16 条协议，先等待 GPU 连续三次（间隔 30 s）可用显存
≥36 GiB，然后自动启动私有 21810/21811 服务并执行。等待上限六小时；若代码在等待期间
变动则不运行过期版本。不是重启真机服务，也不是使用它们跑仿真。

本机路径：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/low_covisibility_no_coverage_20260910_Raarrp/
  navigation_v1/                      # 首次 OOM，保留证据
  navigation_inputs_v1/               # 474 项已校验输入
  navigation_v2_launch.json           # 此次监督进程 PID 与完整命令
  navigation_v2_supervisor.log        # 等待、启动及逐臂结果
  navigation_v2/gpu_wait.json         # 显存等待状态
  navigation_v2/progress.json         # 当前 query/arm（导航启动后）
  navigation_v2/summary.json          # 只包含已完成 rollout
  navigation_v2/independent_verification.json  # 全部完成且审计通过后生成
```

执行器、末步坐标、深度输入、首帧与历史配对均保存可复算证据。未接管 CEC 与 native
做逐动作比对。完成前不报总体 SR，不把部分已完成或选择性正例当完整结果。

本次未改默认 CEC、论文或真机；没有 commit/push。

### 08:21 更新

`navigation_v2` 已满足显存等待条件并自动启动。第一条 253 帧 A 历史已重放完毕，
native query 已实际执行超过 40 步；不是仍在只做定位重放。当前尚无完成 rollout，
因此不报告新 SR。其余 15 条由同一监督进程按固定顺序继续，最终结果以 summary
及 independent verification 为准。

## 最终结果：16/16 完成，独立审计通过

2026-09-10 08:55 全部完成。没有异常 rollout 混入导航失败。两个 Revisit 的全候选域
共视为 0.185 / 0.265；两个配套 Novel 均为零共视。

| 方法 | Revisit | Novel | 合计（仅此机制样本） |
|---|---:|---:|---:|
| native | 1/2 | 1/2 | 2/4 |
| raw fixed | 2/2 | 0/2 | 2/4 |
| 原 CEC | 1/2 | 1/2 | 2/4 |
| CEC without coverage | 2/2 | 1/2 | 3/4 |

去面积规则对原 CEC 为 +1/−0；对 raw 也是 +1/−0；各自 exact McNemar p=1.0。
这是小样本机制证据，不能称为显著、泛化提升或已确认优于 raw。

关键 Revisit（history12 / task71）：两臂都选择 anchor100；旧规则在 Fundamental
参考图覆盖预检查拒绝，新规则获得31个 PnP 内点、RMSE1.293px，并在86步内到达。
raw 同样成功，85步。另一 Revisit 的两种 CEC 均拒绝，并精确沿用已成功的 native。
两个 Novel 的新旧 CEC 都不接管，与 native 逐动作/轨迹/深度输入一致；raw 在第二个
Novel 上干预并从 native 的成功变成失败。

独立 verifier 已核对16条的末步位置与 SPL、bounded执行、深度输入、同起点/目标/A历史、
真实授权策略身份和所有不接管场景的 exact fallback。私有服务已经正常关闭。
这支持在原159查询总体上扩大单因素消融，不支持现在修改默认 CEC 或论文主表。
