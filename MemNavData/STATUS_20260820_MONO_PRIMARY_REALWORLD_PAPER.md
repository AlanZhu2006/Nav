# 2026-08-20：Mono-primary 架构、实机 release 与论文重构

本文接续 `STATUS_20260820_FULLMONO_AND_FRESH_CONFIRMATION.md`，记录 23:00 后的架构
决策。正式实验数字仍以各 protocol/result/independent verifier 为准。

## 1. 决策

项目正式以完整单目系统为方法架构：

```text
one causal RGB stream
  -> one frozen LingBot streaming geometry state
       -> dense short-range depth -> frozen NavDP observation encoder
       -> sparse long-range CEC proof -> scale-free bearing / abstain
  -> one frozen NavDP trajectory policy
```

这不是把 LingBot 和 NavDP 写成两个 action experts。LingBot 提供几何 readout，NavDP
仍是唯一轨迹生成器。方法短语冻结为：

> one causal stream, two time scales, one frozen policy

Metric RGB-D 结果不删除，而是降为明确的受控证据：

- Gate D：只改变 controller depth；
- Final14 / actual-online NNR / 大 HM3D：固定 metric depth，只改变 memory authority；
- Full-Mono HM3D：actual mono Goal-A、actual-online history 与 mono query control 的组合测试。

禁止把三类独立 population 横减，也禁止把 metric-depth 隔离实验改写成 end-to-end mono。

## 2. 实机 protocol-v2 代码修正

旧 real-world hub 是 CEC + metric native。现在改为：

- NavDP server 固定 `depth_source=monocular_sidecar`；
- hub reset 强制并验证 mono depth source，客户端不能切回 sensor depth；
- client depth 仍可上传以兼容 Jetson wire contract，但 hub 不向 policy 转发；
- 每条 NavDP plan 必须带 mono receipt，并证明 policy 没有消费 metric depth；
- certificate reject 且本帧成功进入 shared stream：精确回退 mono-native；
- shared LingBot stream append 失败：不能再伪装成 native fallback，锁存
  `reset_required`，交给 stale-plan 与 0.35 s watchdog 停车；
- hub protocol version 从 1 升为 2。

本地 `realworld_cec_hub`、monocular runtime 与组合协议共 34 tests passed；Python
compile、上下位机 shell syntax 均通过。

## 3. Jetson 同步

目标：`nvidia@tegra-ubuntu`，当前解析到 `10.209.73.32`。

Live overlay：

```text
/home/nvidia/twork/NavDP/deployment/go2/offboard
```

Immutable release：

```text
/home/nvidia/twork/NavDP/deployment/go2/releases/
  cec_mono_20260820_d656b9d9ae30de73
```

Rollback：

```text
/home/nvidia/twork/NavDP/deployment/go2/releases/
  rollback_pre_mono_20260820_d656b9d9ae30de73
```

更新前上下位机均无机器人执行进程。release 旁路同步并复核 SHA 后，才用带备份的方式更新
live overlay。临时 protocol-v2 health hub + SSH loopback tunnel 的 Jetson preflight 5/5
通过；未加载权重、未启动相机、ROS adapter 或 Go2 bridge，零运动命令，smoke 后临时进程
全部停止。

完整回执：`REALWORLD_MONO_RELEASE_RECEIPT_20260820.json`；当前操作手册：
`REALWORLD_GO2_FULLMONO_DEPLOYMENT_20260820.md`。

## 4. 未越过的真机门

LingBot first-40 scale 需要安装后 D435i optical center 的真实离地高度。启动脚本现在要求
显式设置 `CEC_CAMERA_HEIGHT_M`，不再静默使用默认值。health-only smoke 中的 0.5 m 只是
字段占位，没有 reset 或推理，不能作为标定。

下次运动前必须完成：高度测量、camera-only disabled adapter 10 分钟、frame-40 receipt
审计、左右 bearing 符号检查、tunnel/MemNav 两类 fault injection。D435i aligned depth 可留在
Jetson 做碰撞急停，但论文口径必须是“monocular navigation policy with a local depth safety
layer”，不是整台机器人没有 depth sensor。

## 5. 论文重构

论文工作标题改为：

> Certified Episodic Compass: Proof-Before-Control Memory for Monocular
> Continual ImageGoal Navigation

重写范围：title、abstract、introduction、problem formulation、完整方法图、experiments、
results 顺序、analysis、limitations、conclusion、tables、supplement、evidence ledger 与 figure
plan。

方法图现在把 dense/sparse 两条路径都画成实线；metric depth 不在方法图中，只在实验作为
control arm。Results 先报告小规模 Full-Mono composition，再报告更高统计功效的 metric-depth
memory-isolation 结果。Abstract 同时公开 Full-Mono raw memory `11/16` 高于 CEC `9/16`，避免
制造 mono superiority 的假象。

允许写：

- 当前方法架构在 policy boundary 是 mono；
- reused-scene HM3D 已建立 complete causal-RGB composition；
- Final14 建立 proof-before-control 相对 always-on raw 的 mixed-role 优势；
- metric 与 mono 的性能差距仍存在。

禁止写：

- 所有 headline 数字都是 end-to-end mono；
- mono 与 metric non-inferior；
- Full-Mono CEC 超过 raw memory；
- fresh-scene Full-Mono 已确认；
- 已有真机导航 SR。

## 6. Fresh HM3D 状态快照

截至 2026-08-20 23:21，fresh Full-Mono DAG 仍在 source generation：9 个 source task 已
完成，array index 9--53 因 `QOSGrpGRES` 排队。Goal-A、construction、query、summary 和
independent verifier 均未开始，因此没有 fresh SR，论文没有读取或写入任何 partial outcome。

Tectonic 0.16.9 临时工具链已完成真实构建：`paper/main.pdf` 共 9 页，Conclusion 与
References 从第 8 页开始，正文满足 8-page main-text boundary，references 延续到第 9 页；
`paper/supplementary.pdf` 共 3 页。主稿和 supplement 均无 undefined reference/citation、
overfull box 或 LaTeX error；仅有 XeTeX/Times fallback font warning，不影响源码在官方
pdfLaTeX/Overleaf 下复核。

## 7. 下一步

1. 在 Overleaf/pdfLaTeX 再做一次官方模板构建和最终匿名化检查；
2. 等 fresh DAG 的 summary + independent verifier 后再决定是否升级 full-mono claim；
3. 实测相机光心高度并做 camera-only disabled adapter smoke；
4. 通过静态验收后，再做系绳低速真机 trial，形成真机 latency、fallback 和成功率结果。
