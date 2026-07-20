# MemNav 任务提交与依赖检查

本目录中的 Slurm 任务在每次提交前都必须执行本清单。检查范围不只是 Python
包，还包括代码版本、容器/Conda、数据和缓存、权重、输出路径，以及 Slurm
前置任务依赖。

本轮 sparse-keyframe 的实验结果、可靠性消融和结论边界见仓库根目录
[`MEMNAV_SPARSE_KEYFRAME_VALIDATION.md`](../../../MEMNAV_SPARSE_KEYFRAME_VALIDATION.md)。

长任务不得直接裸提交。必须先运行与正式任务使用同一部署目录、容器、Conda、
数据、权重和脚本的零步预检；正式任务必须通过 `afterok` 依赖预检成功。

## 1. 固定代码版本

- 只从个人子目录的独立分支修改；不要直接修改母目录工作树。
- 提交和推送后记录完整 commit SHA。
- 集群使用独立、不可变的部署目录；记录其中的 `DEPLOY_COMMIT`。
- 任务入口会拒绝部署仓库或 LingBot checkout 中任何 tracked modification；
  commit SHA 相同但文件被现场修改也不能运行。
- 本地与部署端的关键脚本 SHA256 必须一致。
- 提交前至少运行：

```bash
git status --short
git diff --check
bash -n scripts/train_memnav/<job>.sbatch
python -m py_compile <changed-python-files>
```

## 2. 验证实际运行环境

以下命令必须在任务使用的同一 Apptainer/overlay/Conda 环境里运行：

```bash
command -v python
python -V
python - <<'PY'
import torch
import transformers
import diffusers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('diffusers', diffusers.__version__)
print('cuda', torch.cuda.is_available())
assert torch.cuda.is_available()
PY
```

- 使用 `command -v python`，不要依赖集群可能包装过的 `which`。
- 运行单元测试的环境还必须能 `import pytest`；生产环境不应因缺少纯测试依赖
  而被误报为模型代码失败。
- 正式 `.sbatch` 在模型构造前执行 `python -m pip check`；任何缺包或版本冲突都
  必须使预检失败，不能只因为 `import torch` 成功就继续提交长任务。
- 必须验证 `PYTHONPATH` 包含部署目录的 `InternNav` 和
  `InternNav/src/diffusion-policy`。
- 容器、overlay、Conda 初始化脚本及环境目录必须可读。

## 3. 验证数据、缓存和权重

提交前打印并核对以下绝对路径：

```text
MEMNAV_ROOT_DIR
MEMNAV_FEATURE_ROOT
LINGBOT_REPO
LINGBOT_WEIGHTS
MEMNAV_DINO_WEIGHTS
```

- 记录 LingBot 和 DINO 权重的 SHA256。
- 保持 `MEMNAV_STRICT_FEATURE_COVERAGE=1`。
- 修复后的生成数据使用
  `MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION=1`；不要混入旧 identity extrinsic
  标签。
- 每个 source-ready episode 必须同时有 `lingbot_cache.npz` 和
  `lingbot_cam_cache.npz`。
- 稀疏 keyframe 训练必须设置 `MEMNAV_REQUIRE_VERSIONED_CACHE=1`。预计算的
  两个文件必须具有相同的 schema、precompute signature、interval、原始帧数和
  sliding-window；训练不得把旧 dense aggregator 与新 sparse camera cache 混用。
  `train_memnav_mp3d.sbatch` 已将它设为 fail-closed 默认值；缺少 signature 或
  expected commit 时任务会主动终止。
- 同时把预计算日志中的完整 SHA256 写入
  `MEMNAV_EXPECTED_CACHE_SIGNATURE`；零步预检会扫描全部 episode 的小型 metadata
  和 `.npy` header，并在加载模型前拒绝不同生成批次或截断 payload。
- 正式预检还必须设置 `MEMNAV_EXPECTED_CODE_COMMIT`；任务会把部署目录的
  `git rev-parse HEAD` 与它比较，并打印 LingBot commit 及两份权重 SHA256。
- parquet、RGB、DINO CLS、`cam_pose_enc` 的帧数及 goal 索引必须一致。
- `MEMNAV_WINDOW`、`MEMNAV_NUM_SCALE`、`MEMNAV_MAX_FRAME_NUM` 必须与预计算
  几何一致，且最大帧数能覆盖最长 episode。
- 新 sparse 训练默认 `MEMNAV_USE_POSE_RELIABILITY_CONDITIONING=0` 且
  `MEMNAV_W_POSE_RELIABILITY=0`：旧 head 在 30 个 paired 样本上近似常数，且不能
  识别错误 retrieval anchor；它只保留为日志诊断，不再无依据地乘低 semantic gate。
- Range auxiliary 默认保持关闭（`MEMNAV_W_AUX_RANGE=0`），anchor 仍默认为全
  teacher-forced（`MEMNAV_ANCHOR_TF_START=1`、`END=1`）。2026-07-18 的 400-step
  range+live 对照虽然把 range-code MAE 降低约 9.6%，同一 fixed-64 上的 action
  noise MSE 却恶化约 28.8%。随后在正式 batch size 4 的本机梯度诊断中，已乘
  `0.2` 权重的 range 梯度在共享 adapter 上仍是 action 梯度的约 29 倍（中位数），
  且 7 个含 revisit 的 batch 中有 2 个方向冲突。
- 新实验可显式设置 `MEMNAV_AUX_RANGE_GRAD_CAP_RATIO`：正值会投影掉与 action
  反向的 range 梯度，再把剩余范数限制为 action 梯度的指定比例；`0` 保留旧
  backward。启用时必须同时记录 range weight、SmoothL1 beta、gradient cap 和完整
  teacher-forcing schedule。checkpoint metadata 会保存这些值，W&B 必须记录 raw
  cosine、raw/corrected norm ratio、cap scale 和 conflict fraction。该机制仍是实验
  选项，当前只支持单进程训练（多进程 DDP 会 fail closed），不能在完整 DDPM 和
  最终导航评测前改成生产默认值。2026-07-19 的正式 `cap=0.25` 对照虽然比旧
  range+live arm 降低 `6.60%` 的 full-DDPM action MSE，仍比无 range baseline
  恶化 `32.44%`，且只有 3/64 paired 样本改善；该配置已被拒绝，range loss 必须
  继续默认关闭。后续任务不得复用 `0.25` 作为推荐值，除非先完成新的本机 paired
  验证并建立不会由 auxiliary 直接改写 action 表示的隔离结构。
- Long-route decision curriculum 是独立、默认关闭的数据采样实验。只有显式设置
  `MEMNAV_SAMPLING_MODE=decision_curriculum` 才启用；默认 `random_leg` 和固定验证集
  的采样及 fingerprint 与旧 checkpoint 保持一致。默认候选定义为：距目标至少
  128 帧，且当前到未来 16 帧的近程位移方向与当前到最终目标的直线方向相差至少
  45 度；有候选的 goal-sample 以 0.5 概率从候选池均匀抽 k，否则沿用原均匀采样。
  该定义只读取训练标签选择 k，不进入模型输入，推理不需要未来轨迹。必须记录：
  `MEMNAV_DECISION_CURRICULUM_PROB`、`MEMNAV_DECISION_LOOKAHEAD_FRAMES`、
  `MEMNAV_DECISION_MIN_REMAINING_FRAMES`、`MEMNAV_DECISION_MIN_ANGLE_DEG`，以及 W&B 的
  `decision_hard_fraction`、`decision_route_angle_deg`、
  `action_loss_decision_hard/easy`。本机完整协议和结论边界见
  [`runs/2026-07-20-long-decision-curriculum-local.md`](runs/2026-07-20-long-decision-curriculum-local.md)。
- Residual route sketch 也是独立、默认关闭的结构实验。启用时必须同时记录
  `MEMNAV_USE_ROUTE_SKETCH=1`、`MEMNAV_ROUTE_HORIZONS`、
  `MEMNAV_W_ROUTE_DIRECTION`、`MEMNAV_ROUTE_LR_MULTIPLIER` 和
  `MEMNAV_ROUTE_CURVATURE_EMPHASIS`。它只从推理时已有的
  current/revisit/novel memory 预测多时域局部方向；未来 action 只用于 label-side
  loss，不能进入 forward 输入。旧 checkpoint 缺少的 route 参数仅允许通过
  zero-residual migration 补齐，启用后第一次推理必须和旧模型逐字段完全一致。
  route loss 读取 detached 的旧表示，不能单独反传污染旧 backbone；只有 diffusion
  主 loss 可以决定三个 residual scale 是否打开。residual 还必须由模型自己预测的
  short/long-horizon 曲率门控；GT 曲率只能重加权训练 loss，不能作为推理 gate。
  提交长跑前必须完成同 checkpoint、同 seed、同样本序列的 route-off/on 配对，并在
  完整 DDPM 上同时检查 3-leg Goal C、span>=256、2-leg 回归、goal sensitivity、
  route 角误差和实际 residual scale。
  2026-07-21 的本机配对实验已拒绝当前 v2：full-DDPM action MSE 从
  `0.107667` 恶化到 `0.110549`（+2.68%），2-leg 恶化 4.05%，且 route head
  在总体 h2/h8/h24 上没有击败“永远向前”的常数基线。该配置不得提交长任务；
  完整结果和 RNG 配对证据见
  [`runs/2026-07-21-residual-route-sketch-local.md`](runs/2026-07-21-residual-route-sketch-local.md)。
  以后任何 route-off/on 对照还必须逐日志核对 `diffusion_noise_mean/std` 和
  `diffusion_timestep_mean`，不能只凭相同命令行 seed 声称配对。
- 预计算必须同时满足应用日志 `errors=0` 和 Slurm
  `COMPLETED, ExitCode=0:0`；脚本捕获错误后继续运行不算成功。
- stdout、checkpoint、W&B 和诊断输出目录必须在 `sbatch` 前存在且可写，
  因为 Slurm 会在脚本正文执行前打开日志文件。

## 4. 单元测试和零步预检

先运行与改动相关的单元测试。MemNav 核心测试示例：

```bash
PYTHONPATH=InternNav:InternNav/src/diffusion-policy \
python -m unittest discover \
  -s InternNav/tests/unit_test -p 'test_memnav*.py' -v
```

然后复用正式训练 `.sbatch` 做零步预检，只缩短训练本身：

```bash
NAME=memnav_preflight_<commit> \
MEMNAV_REPORT_TO=none \
BATCH_SIZE=2 EPOCHS=0 NUM_WORKERS=0 \
sbatch --time=00:30:00 --export=ALL \
  scripts/train_memnav/train_memnav_mp3d.sbatch
```

预检必须实际完成：容器挂载、Conda 激活、关键 imports、CUDA、严格数据扫描和
fingerprint、权重加载、模型/Trainer 构造以及零步退出。只有最终状态为
`COMPLETED, ExitCode=0:0` 才通过。

需要验证真实 backward/optimizer/checkpoint 时，使用有界短跑而不是等待任务超时：

```bash
MEMNAV_MAX_TRAIN_STEPS=10 \
MEMNAV_LOGGING_STEPS=1 MEMNAV_SAVE_STEPS=5 \
MEMNAV_REPORT_TO=none RESUME_FROM_CHECKPOINT=none \
sbatch --time=00:30:00 --export=ALL \
  scripts/train_memnav/train_memnav_mp3d.sbatch
```

`MEMNAV_MAX_TRAIN_STEPS=-1`（默认）保持正式任务按 epoch 训练；任何正式长跑都应在
任务记录中明确写出该值。

从已完成 checkpoint 做“相同起点、不同 treatment”的控制实验时，不要使用
`RESUME_FROM_CHECKPOINT`：resume 会同时恢复旧 optimizer、scheduler、global step 和
RNG，而且会拒绝 curriculum 引起的 dataset fingerprint 改变。应使用权重初始化：

```bash
MEMNAV_INIT_CHECKPOINT=/absolute/path/checkpoint-400/memnav.ckpt \
RESUME_FROM_CHECKPOINT=none \
MEMNAV_LR=1e-5 MEMNAV_SEED=0 \
MEMNAV_MAX_TRAIN_STEPS=200 \
sbatch --export=ALL scripts/train_memnav/train_memnav_mp3d.sbatch
```

入口会检查初始化文件存在、打印其 SHA256，并拒绝同时指定初始化 checkpoint 和
非空 resume。配对 arm 必须记录完全相同的初始化 SHA256、学习率、seed、batch size、
step budget 和固定验证 fingerprint；只有 treatment（例如 sampling mode）可以不同。

## 5. 长任务必须依赖预检

```bash
PREFLIGHT_JOB=$(sbatch --parsable <preflight-options> <script>)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}

TRAIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREFLIGHT_JOB}" \
  --time=8:00:00 \
  <training-options> <script>)
TRAIN_JOB=${TRAIN_JOB%%;*}

scontrol show job "${TRAIN_JOB}"
```

若还依赖缓存补算，正确链路为：

```text
cache/precompute --afterok--> zero-step preflight --afterok--> long training
```

不能用 `afterany` 代替 `afterok`。提交后从 `scontrol show job` 核对
Dependency、TimeLimit、WorkDir、StdOut/StdErr、partition、GPU、CPU、内存和
所有导出变量。

## 6. 提交后验证和留档

```bash
squeue -j <job_ids> -o '%.18i %.24j %.10T %.10M %.9l %.26R'
sacct -j <job_ids> -X --format=JobID,State,Elapsed,ExitCode,NodeList -P
squeue --start -j <job_id>
```

`RUNNING` 只表示 Slurm 已启动，不代表模型正常。还要检查日志已经进入 Python、
数据集和模型前向，GPU 有实际计算，W&B 指标更新，并在保存步产生完整的模型、
optimizer、scheduler、RNG、trainer state 和 metadata。

每次任务至少记录：

```text
commit / deployment directory:
dataset and feature roots / fingerprints:
weights and SHA256:
container / Conda / Python / package versions:
preflight JobID / final state / exit code:
long JobID / exact afterok dependency:
resources / time limit:
W&B run ID:
stdout / stderr:
first complete checkpoint:
```

任一项未知时，应明确写“等待预检”或“被依赖阻塞”，不能表述为训练已经正常。
