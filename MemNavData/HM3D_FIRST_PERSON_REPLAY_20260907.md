# HM3D 长距离第一视角回放（2026-09-07）

已将用户指定的 `005_eF36g7L6Z9M / episode_table3_survey_336` 三臂记录在本机
重新渲染为第一视角。不是重跑策略，不产生新的 SR，也没有改写正式结果。

## 视频位置

本机目录：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/first_person_replay_20260907_h10x6s/videos/
```

| 文件 | 内容 | 时长 | 帧数 |
|---|---|---:|---:|
| `first_person_comparison.mp4` | 左 Native / 中 CEC endpoint / 右 route tangent，含原始目标图 | 81.167 s | 974 |
| `mono_native_first_person.mp4` | Native 纯 RGB，无文字覆盖 | 81.167 s | 974 |
| `mono_cec_endpoint_first_person.mp4` | 原始 CEC endpoint 纯 RGB，无文字覆盖 | 52.250 s | 627 |
| `mono_cec_route_tangent_first_person.mp4` | 路线引导诊断臂纯 RGB，无文字覆盖 | 62.834 s | 754 |

全部为 H.264 / yuv420p / 12 fps。并排视频为 1440×504；纯 RGB 为原始 480×270。
每秒播放 12 个记录状态，不代表实验墙钟速度。并排版中先结束的 arm 保持其明确保存的
终点位置和朝向，并显示 `Terminal view held`；没有继续模拟该 arm。

## 来源与重放方式

远端原始 run：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_tangent_20260903/fresh_751efdcbb436c99b
```

- 回放输入为该 run 的 `evaluation/005_eF36g7L6Z9M_episode_table3_survey_336/`。
- `source_receipt.json` 位于本机视频目录的上一级，绑定三个原始 plans JSON、completion、
  role_pairs、goal.jpg、GLB 与 NavMesh；传输前后哈希一致。
- 相机与封存代码一致：480×270，HFOV 从 `fx=355.81464` 计算，离地高度 0.5 m。
- 使用逐动作 `rollout_traces.query` 中的 Habitat Y-up 位置和 yaw，追加日志明确保存的
  `end_position/end_yaw_rad`。不插值，不平滑，不推测末端姿态，不额外转向目标。
- 加载封存 NavMesh，但不执行寻路、碰撞推进或导航策略。
- 该例原本就是受控 causal RGB survey 下的长距离诊断，不是 actual NavDP-A 的主表实验。
  选择此例是因为用户已经指定对应轨迹视频，不作新增总体性能结论。

## 核验

- 用原实验 `JPEG quality=95` 编码抽查的重渲染帧：Native `11/11`、endpoint `8/8`、
  route tangent `9/9` 与日志内原始 RGB JPEG SHA-256 一致。这是在视频压缩之前的检查；
  不声称有损 H.264 解码帧逐字节等于原始 JPEG。
- 原始目标 JPEG 与未压缩目标重渲染 RGB 的平均绝对像素差为 `2.0085/255`。
- 四个 MP4 均经 ffprobe 检查，并完整解码通过，未报告视频解码错误。
- 首帧、最终帧及视频第 32 秒的实际解码帧已人工式视觉检查。
- 位姿末端处理、高度轴、因果时间索引等测试：`5 passed`。
- 全部输入在渲染前后 SHA 不变；没有修改论文、正式结果、阈值或正在运行的服务器。

完整渲染与编码约 `11.4 s`，不含资产传输及开发验证时间。

## 复现

```bash
/home/asus/miniconda3/envs/habitat/bin/python \
  MemNavData/render_recorded_pose_first_person.py \
  --inputs .diagnostics/first_person_replay_20260907_h10x6s \
  --out-dir .diagnostics/first_person_replay_20260907_h10x6s/videos_new \
  --fps 12

/home/asus/miniconda3/envs/memnav/bin/python -m pytest \
  MemNavData/test_render_recorded_pose_first_person.py -q
```

输出目录必须是新目录，避免覆盖已有视频。`render_receipt.json` 保存相机参数、抽帧哈希
检查、各臂原始终止结果、视频帧数及视频 SHA。

开发记录：首次 smoke 因未传入 pinned NavMesh 而结束，已补齐原始 NavMesh 后通过；
Habitat 环境没有 pytest，因此测试使用已有 memnav 环境，没有安装新依赖。
Habitat 会提示场景缺少语义标注，本任务只渲染 RGB，不请求语义传感器；实际 RGB 哈希
抽查通过。临时 localhost HTTP 进程已关闭，共享 SSH master 保留不动。
