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

3. 确认自己账号下的独立部署目录（例如
   `/scratch/<user>/Research/Nav-axis-uturn/InternNav`）指向该 commit，不从其他人
   正在修改的共享主工作树启动；提交时显式导出同一路径的 `REPO_ROOT`。日志和
   新 checkpoint 写入该目录，共享数据/权重仅只读复用。
4. 核对 `MEMNAV_ROOT_DIR`、`MEMNAV_FEATURE_ROOT`、`LINGBOT_REPO`、
   `LINGBOT_WEIGHTS`、`MEMNAV_INIT_CKPT` 和输出目录。
5. sparse cache 训练必须设置 `MEMNAV_REQUIRE_VERSIONED_CACHE=1`，且
   `MEMNAV_WINDOW/NUM_SCALE/MAX_FRAME_NUM` 与预计算一致。

正式脚本会自动运行 `dependency_preflight.py` 和 `gpu_preflight.py`。前者检查
真实 Python import（包括生产 `train.py` 的 MemNav 模型选择路径）、路径、一个 cache
pair 的 header/schema、`MEMNAV_INIT_CKPT -> il.ckpt_to_load` 配置接线、warm-start 权重
结构及写权限；后者检查每张可见 GPU 能否创建 CUDA context。真实 batch 阶段还会将
模型内四个关键 tensor 与源 checkpoint 逐值比较，防止“文件存在但训练没有加载”。

## 先提交真实 batch 预检

预检和正式训练必须使用同一份 `.sbatch`、容器、overlay、Conda、数据、cache 与
权重。预检先构造完整模型并执行一个真实 revisit batch 的前向和反向，然后再通过
正式 `scripts/train/train.py` 入口运行一个 optimizer step。这样既验证 gate 梯度，
也覆盖正式入口、Trainer、DataLoader 和模型选择式依赖：

```bash
PREFLIGHT_JOB=$(REPO_ROOT=/scratch/<user>/Research/Nav-axis-uturn/InternNav \
  NAME=<run>_preflight \
  MEMNAV_PREFLIGHT_ONLY=1 MEMNAV_TRAIN_MAX_STEPS=1 \
  MEMNAV_REPORT_TO=none NPROC=1 BATCH_SIZE=1 NUM_WORKERS=0 \
  sbatch --parsable --time=00:30:00 --gres=gpu:1 --export=ALL \
  scripts/train_memnav/train_memnav_mp3d.sbatch)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
```

只有 `sacct` 显示 `COMPLETED` 且 `ExitCode=0:0`、日志同时包含
`[full-preflight] PASS` 与 `ENTRYPOINT-PREFLIGHT PASS`，才允许长任务开始。

## 长任务必须依赖预检成功

```bash
TRAIN_JOB=$(REPO_ROOT=/scratch/<user>/Research/Nav-axis-uturn/InternNav \
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

本次 gate curriculum 的固定 all-leg 8 小时实验可直接运行：

```bash
bash scripts/train_memnav/submit_gate_curriculum_8h.sh
```

它封装的仍是上述 preflight/`afterok` 流程，不会跳过任何检查。

## 固定 checkpoint A/B 评测

训练曲线来自不同随机 batch，不能单独证明新 checkpoint 泛化更好。使用
`fixed_checkpoint_eval.py` 做 checkpoint 对照时，两个模型必须共享完全相同的：

- 独立 episode 清单、当前帧 `k`、目标和标签；
- diffusion noise 与 timestep（每个样本重复多次后取均值）；
- W32/schema-v2 cache、LingBot 权重、坐标与 metric-scale 配置；
- 推理路径：`model.eval()`，不使用 GT-positive anchor，也不使用 teacher gate。

正式 A/B 默认平衡抽取 novel Goal-A、novel covis、2/3-leg shallow/deep 六组，
每条轨迹最多一个样本，并保存逐样本 JSONL、分组 summary 和 paired-bootstrap
置信区间。提交仍采用真实路径预检 `--afterok` 正式评测：

```bash
bash scripts/train_memnav/submit_fixed_checkpoint_eval.sh
```

预检会用两个真实 checkpoint 跑六组各一个样本，覆盖容器依赖、GPU、dataloader、
cache schema、LingBot、两次权重装载、真实 retrieval anchor、pose 和 diffusion decoder；
只有预检 `COMPLETED 0:0` 后，72-sample/8-trial 的正式任务才启动。评测结果写入个人
`/scratch/<user>/Research/...-results/`，不会写共享主仓库。
