# 连续当前状态与稀疏目标关系：Pi3X 固定观测读出实验

日期：2026-09-06。承接 [Pi3X 时序上下文归因](PI3X_TEMPORAL_CONTEXT_ATTRIBUTION_20260906.md)。本轮是两条已消费轨迹上的本机机制诊断，不是新的闭环 SR 结果。

后续已补完剩余两条：[四 query / 三 scene 结果](PI3X_SHARED_STATE_FOUR_QUERY_RESULT_20260906.md)。连续状态分工的方向改善在四条上均出现；但弱支持 Novel 的动态目标关系比固定首次目标更准确，因此不能把本轮前两例解读为“目标永远只估计一次最好”。仍无新 SR。

## 1. 结果与直接判断

在相同记录观测上，把 Pi3X 每次联合重建的当前状态，换成连续 LingBot 状态，并通过共同历史帧对齐坐标与单位，方向长尾明显减小。再固定首次目标关系，有小幅进一步改善。

两条 query 原本获授权的 64 个重规划时刻：

| 读出方式 | 中位方位误差 | P95 | 最大误差 | ≤30° | >90° |
|---|---:|---:|---:|---:|---:|
| Pi3X 联合重建当前与目标 | 16.28° | 95.46° | 178.34° | 37/64 | 12/64 |
| 连续 LingBot 当前状态 + 每次更新的 Pi3X 目标关系 | 3.06° | 28.03° | 32.74° | 62/64 | 0/64 |
| 连续 LingBot 当前状态 + 首次固定的 Pi3X 目标关系 | 2.19° | 25.93° | 32.07° | 63/64 | 0/64 |

**主要改善在第二行就已出现。** 因而本轮支持的是连续状态与稀疏目标定位的分工，不能将收益全部归于“缓存第一次目标”。第二行也包含共同历史坐标对齐，不能解释为直接替换一个未经对齐的 pose 数组。

64 个时刻来自两条轨迹，且第一条贡献 49 个；这是描述性统计，不是 N=64 的独立验证，不计算 episode-level 显著性。没有运行 NavDP、执行动作或得到新 SR；Pi3X 原正式替换决定仍为未通过，canonical CEC 不变。

## 2. 为什么做这个实验

前轮已经区分了两个量：

- 目标方位误差：预测 bearing 与当前位置到目标位置的直线方向之差。
- 路线方向差：预测 bearing 与最短路首段之差。

本轮只评价前者。它对应相对定位是否准确，不表示该方向能直接绕过墙壁。

原正式数据中，Pi3X 首次接受的方位并不差，部分后续更新才出现大幅偏转。前轮固定当前图、更换因果 bridge 的干预，又将不稳定项定位到 current-to-anchor 关系。因此，本轮不重训 selector、不改 proof 阈值、不改 2.5 m 接口，而是比较状态更新方式。

三种读出在查看本轮输出前写入 `.diagnostics/pi3x_persistent_goal_20260906/PROTOCOL.md`。案例本身来自已经看过的异常，故属于事后开发诊断，不称为新预注册确认。

## 3. 数据与可比性

| query | 原 Pi3X / CEC 结果 | query 开始前帧数 | 完整 RGB 序列 | 重规划时刻 | 原 accepted | 固定 anchor |
|---|---|---:|---:|---:|---:|---:|
| 8WUmhLawc2A / episode_0000 / Revisit | 失败 / 成功 | 234 | 668 | 55 | 49 | 140 |
| V2XKFyX4ASd / episode_0004 / Revisit | 成功 / 成功 | 152 | 267 | 15 | 15 | 39 |

从原 run 拉取完整因果 RGB 序列共 935 帧，以及 2 张目标图，共 46,226,462 bytes。所有历史 JPEG 均按原记录 SHA 核验；目标 JPEG 按封存目标核验，937 个文件全部一致。

原 run：

`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/`

这两条是旧 Final14 metric-controller 条件下的实际记录，不是新采集的 full-mono rollout。三种读出使用同一条原 Pi3X 轨迹的图像与物理状态，没有把 CEC 自己走出的观测拿来替代 Pi3X 的观测。

本轮直接调用几何模型，不运行 learned proof heads。原 accept/reject 只用于划分统计子集；新增读出不继承未经检验的新授权，也没有把原 reject 改成 accept。首次 anchor 沿用原记录，没有重新做一次完整 DINO 检索实验。

## 4. 连续状态和坐标对齐

### 4.1 LingBot 的输入与因果性

复用现有 `LingBotStream` 和 `extract_trajectory`：

- RGB 按原始顺序读入；不输入目标图、GT pose 或 sensor depth。
- 初始块 S=8，window=32，max_frame_num=2048，SDPA，BF16。
- flow threshold=25 px，max gap=30；沿用原 600-step 合约对应的 tier，不按本次实际截取长度选择 tier。
- 初始 8 帧组成原有启动块；之后逐帧因果处理。所有被评分 query 时刻均晚于启动块。
- 旧 pose 输出写入后不被未来帧重新优化；本轮不调用全轨迹地面尺度汇总。

原 extractor 的 flow 判据会读取自身预测深度，这仍是 RGB 模型输出，并非外部米制深度。KV 只保留选中的 keyframe，不代表未写入 KV 的当前帧没有 pose 输出。

### 4.2 两个单目重建不能直接相减

令 `L` 为 LingBot 的连续坐标系，`P` 为某次 Pi3X 重建。两者均使用 camera-to-world 位姿。用同一历史 anchor `a` 对齐旋转，以 `a−8` 和 `a+8` 的相机基线对齐单位：

```text
s = ||t_L(a+8) − t_L(a−8)|| / ||t_P(a+8) − t_P(a−8)||
g_L = t_L(a) + s R_L(a) R_P(a)^T [t_P(goal) − t_P(a)]
r_t = R_L(t)^T [g_L − t_L(t)]
b_t = normalize([r_t,z, −r_t,x])
```

`[r_t,z, −r_t,x]` 表示相机坐标中的前向与左向分量。固定的是目标在连续坐标系中的位置关系，不是机器人坐标下永远不变的 bearing；机器人移动或转向时，bearing 仍随当前位姿更新。

共同支持帧分别为 `[132,140,148]` 与 `[31,39,47]`，都早于首次 query。初次两模型单位比分别为 0.198922 和 0.192230。这不是 GT 米制标定，也不使用相机高度；它只把两个单目重建表达在可比较的内部单位下。

动态目标对照每次从 Pi3X 重估上述目标关系；固定目标对照只使用首次结果。二者的连续 LingBot 状态完全相同。共同基线可能受估计误差影响，因此这个对齐不构成一般性的无漂移保证。

GT 仅在所有读出完成后计算 endpoint 方位误差，不参与输入、对齐或目标选择。

## 5. 逐 query 结果

### 5.1 8W：原 Pi3X 失败案例

原 accepted 的 49 个时刻：

| 读出 | 中位误差 | P95 | ≤30° | >90° |
|---|---:|---:|---:|---:|
| Pi3X 联合 | 30.30° | 94.59° | 24/49 | 10/49 |
| 连续状态 + 动态目标 | 6.40° | 28.87° | 47/49 | 0/49 |
| 连续状态 + 初始目标 | 5.62° | 26.59° | 48/49 | 0/49 |

固定目标读出并非一直无误差：最大仍为 32.07°，最后时刻为 14.68°。原 step 392 的误差为 97.36°，两种连续状态读出为 14.71° / 13.41°。因此目前消除了该轨迹上观测到的大翻转，但仍有残余定位误差。

### 5.2 V2：原本已成功的案例

全部 15 个时刻原本均 accepted：

| 读出 | 中位误差 | P95 | ≤30° | >90° |
|---|---:|---:|---:|---:|
| Pi3X 联合 | 6.99° | 150.06° | 13/15 | 2/15 |
| 连续状态 + 动态目标 | 1.09° | 4.20° | 15/15 | 0/15 |
| 连续状态 + 初始目标 | 0.78° | 2.19° | 15/15 | 0/15 |

step 104 的 178.34° 变为 3.45° / 2.10°。这是方向稳定性改善，不是挽回一条失败，因为原 episode 已成功。

### 5.3 全部重规划时刻

补充检查全部 70 个时刻：≤30° 次数依次为 40/70、67/70、69/70，>90° 次数为 12、0、0。新增的 6 个原拒绝时刻仅用于观测诊断，不赋予它们控制权。

## 6. 这对“学习不如工程拼接”的解释有什么影响

本轮不支持笼统的“目标关系学不出来”。在这两条已消费轨迹上，Pi3X 首次关系与连续状态组合后能持续给出较准确的方位，主要异常来自每次稀疏联合重建与历史当前状态之间的连接。

更有根据的任务分解是：

```text
连续因果 RGB → 持续更新当前几何状态
历史地址 + 目标图 → 稀疏估计目标相对历史的关系
两者在共同历史坐标中组合 → 当前相对目标 bearing
```

这与当前 CEC 的时序分工一致，但不证明 Pi3X 已可替换 SuperPoint、LightGlue、PnP 和 certificate。开放集拒绝、首次关系可靠性、换目标时的状态生命周期仍需验证。把 Pi3X 再永久叠加到 CEC 上也不自动降低系统复杂度。

若研究 learned 替代，本轮支持先研究“稀疏关系估计”这一职责，而非再次让同一小模型同时承担检索、当前定位、目标定位、授权和路线规划。它是有实测依据的研究入口，不是已经获得更高 SR 的新方法。

## 7. 核验、成本与边界

- 另一套 4×4 齐次变换与 dot-product 角度公式，复算全部 70 个时刻；最大误差差异分别为 `2.6721e-5°` 与 `9.9646e-5°`，验证通过。
- 100 个合成全局相似变换对齐测试通过；这检查公式的坐标不变性，不检查模型位姿真值。
- LingBot 输出全部有限；旋转行列式约在 `[0.99999962, 1.00000040]`。
- 本机 RTX 4090；LingBot 为 Torch 2.8.0+cu128，Pi3X 为 Torch 2.5.1+cu124，均 BF16，在各自环境依次运行。不是原 H100 同进程闭环复现。
- 原 H100 在这两条上的 endpoint >90° 共 13 次，本机原 Pi3X 读出共 12 次；不掩盖跨运行时差异。三臂对照基于本机同一套封存观测与几何输出。
- LingBot 两条离线提取耗时约 135.22 s 和 43.38 s；这些数不包含完整部署、检索、网络通信和 NavDP，不能作为系统实时性或加速比。
- 首次本地调用漏传 flow 模式的 interval sentinel，原 extractor 在流式计算前拒绝；只修正诊断调用并重跑，失败日志保留。最终模型加载为 0 missing / 0 unexpected weights。
- 本轮没有新闭环 SR、显著性或非劣效结论。改变方向后机器人会获得不同的新观测，离线改善不能直接外推为成功。
- 本轮不能证明空间长程问题已解决。目标直线方位准确仍不等于可执行路线正确，也不排除更长序列中的 LingBot 漂移。

## 8. 文件与下一步

所有新诊断文件位于 `.diagnostics/pi3x_persistent_goal_20260906/`：

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 本轮固定的三读出对照 |
| `inputs_manifest.json`、`inputs/` | 封存 RGB、SHA、原授权与仅评分用 GT |
| `extract_continuous_state.py`、`continuous_state.json` | 因果 LingBot 重放及逐帧 pose |
| `extract_pi3x_relations.py`、`pi3x_relations.json` | 全部 70 次 Pi3X 几何读出 |
| `compare_readouts.py`、`readout_comparison.json` | 三读出逐时刻结果与汇总 |
| `alternate_formula_verification.json` | 另一套公式的复算收据 |
| `continuous_state_attempt0.log`、`continuous_state_console.log`、`pi3x_console.log` | 启动与推理日志 |

只重新计算统计，无需 GPU：

```bash
python .diagnostics/pi3x_persistent_goal_20260906/compare_readouts.py
```

重新提取连续状态与 Pi3X（分别在已验证的环境）：

```bash
/home/asus/miniconda3/envs/memnav/bin/python -u \
  .diagnostics/pi3x_persistent_goal_20260906/extract_continuous_state.py \
  --root .diagnostics/pi3x_persistent_goal_20260906

/home/asus/miniconda3/envs/pi3x/bin/python -u \
  .diagnostics/pi3x_persistent_goal_20260906/extract_pi3x_relations.py \
  --root .diagnostics/pi3x_persistent_goal_20260906
```

上述提取脚本将结果打印到 stdout，本轮日志与 JSON 已分别归档；复现时不要覆盖原文件。

下一项最小验证：沿用同一固定对齐方式，检查前轮剩余两条已消费异常 query（包括误激活 Novel），不按新结果选择阈值。若关系读出仍稳定，再设计少量真正闭环的开发对照，同时检查原本成功的轨迹有无损失；不能只跑失败案例，也不能继承旧 proof 作为已验证的新授权。

当前已结束本机推理，GPU 利用率回到 0%，显存回到本轮开始时已有的 6,556 MiB。本轮临时传输服务和转发已关闭，共享 SSH master 保留；未提交或取消 HPC GPU 作业，未操作机器人，未改生产代码或论文，未 commit/push。
