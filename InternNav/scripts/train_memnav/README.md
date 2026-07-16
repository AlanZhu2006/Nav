# MemNav 任务提交与依赖检查

本目录中的 Slurm 任务在每次提交前都必须执行本清单。检查范围不只是 Python
包，还包括代码版本、容器/Conda、数据和缓存、权重、输出路径，以及 Slurm
前置任务依赖。

长任务不得直接裸提交。必须先运行与正式任务使用同一部署目录、容器、Conda、
数据、权重和脚本的零步预检；正式任务必须通过 `afterok` 依赖预检成功。

## 1. 固定代码版本

- 只从个人子目录的独立分支修改；不要直接修改母目录工作树。
- 提交和推送后记录完整 commit SHA。
- 集群使用独立、不可变的部署目录；记录其中的 `DEPLOY_COMMIT`。
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
- parquet、RGB、DINO CLS、`cam_pose_enc` 的帧数及 goal 索引必须一致。
- `MEMNAV_WINDOW`、`MEMNAV_NUM_SCALE`、`MEMNAV_MAX_FRAME_NUM` 必须与预计算
  几何一致，且最大帧数能覆盖最长 episode。
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
