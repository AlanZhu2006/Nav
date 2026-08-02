# MemNav 长任务提交检查清单

每次提交训练都要验证代码、环境、数据、cache、权重和输出路径。不能因为
`sbatch` 返回了 JobID 就认为任务可运行。

## 提交前

1. 在独立工作树完成修改、测试、commit 和 push，记录完整 commit SHA。
2. 运行：

   ```bash
   git diff --check
   bash -n scripts/train_memnav/train_memnav_mp3d.sbatch
   python -m py_compile <所有改动的 Python 文件>
   python -m pytest -q tests/unit_test
   ```

3. 确认部署目录 `/scratch/lg154/Research/Nav-axis-uturn/InternNav` 指向该
   commit，不从其他人正在修改的共享主工作树启动；提交时显式导出同一路径的
   `REPO_ROOT`。
4. 核对 `MEMNAV_ROOT_DIR`、`MEMNAV_FEATURE_ROOT`、`LINGBOT_REPO`、
   `LINGBOT_WEIGHTS`、`MEMNAV_INIT_CKPT` 和输出目录。
5. sparse cache 训练必须设置 `MEMNAV_REQUIRE_VERSIONED_CACHE=1`，且
   `MEMNAV_WINDOW/NUM_SCALE/MAX_FRAME_NUM` 与预计算一致。

正式脚本会自动运行 `dependency_preflight.py` 和 `gpu_preflight.py`。前者检查
真实 Python import、路径、一个 cache pair 的 header/schema、warm-start 权重结构及
写权限；后者检查每张可见 GPU 能否创建 CUDA context。

## 先提交真实 batch 预检

预检和正式训练必须使用同一份 `.sbatch`、容器、overlay、Conda、数据、cache 与
权重。预检会构造完整模型并执行一个真实 revisit batch 的前向和反向：

```bash
PREFLIGHT_JOB=$(REPO_ROOT=/scratch/lg154/Research/Nav-axis-uturn/InternNav \
  NAME=<run>_preflight \
  MEMNAV_PREFLIGHT_ONLY=1 MEMNAV_REPORT_TO=none NPROC=1 BATCH_SIZE=1 NUM_WORKERS=0 \
  sbatch --parsable --time=00:30:00 --gres=gpu:1 --export=ALL \
  scripts/train_memnav/train_memnav_mp3d.sbatch)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
```

只有 `sacct` 显示 `COMPLETED` 且 `ExitCode=0:0`、日志包含
`[full-preflight] PASS`，才允许长任务开始。

## 长任务必须依赖预检成功

```bash
TRAIN_JOB=$(REPO_ROOT=/scratch/lg154/Research/Nav-axis-uturn/InternNav \
  NAME=<run> MEMNAV_PREFLIGHT_ONLY=0 \
  sbatch --parsable --time=08:00:00 --dependency="afterok:${PREFLIGHT_JOB}" \
  --export=ALL scripts/train_memnav/train_memnav_mp3d.sbatch)
TRAIN_JOB=${TRAIN_JOB%%;*}
```

提交后用以下命令核对依赖、时限、资源和状态：

```bash
scontrol show job "${TRAIN_JOB}"
squeue -j "${PREFLIGHT_JOB},${TRAIN_JOB}" -o '%.18i %.28j %.10T %.10M %.9l %.28R'
sacct -j "${PREFLIGHT_JOB},${TRAIN_JOB}" -X --format=JobID,State,Elapsed,ExitCode,NodeList -P
```

必须记录：commit、部署目录、数据/cache/权重路径、预检 JobID 与最终状态、长任务
JobID、W&B run、stdout/stderr 路径，以及第一个完整 checkpoint。`PENDING
(Dependency)` 是正常等待；`RUNNING` 仍需检查日志、GPU 利用率和 W&B 是否真正更新。
