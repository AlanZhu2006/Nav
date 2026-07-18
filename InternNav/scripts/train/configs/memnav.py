import os
from pathlib import Path

from internnav.configs.model.memnav import memnav_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import IlCfg


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

# HPC-friendly defaults: env vars override the developer-desktop paths so a
# SLURM job only needs `export MEMNAV_ROOT_DIR / LINGBOT_REPO / LINGBOT_WEIGHTS`.
_ROOT_DIR = os.environ.get(
    'MEMNAV_ROOT_DIR',
    '/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data',
)
_LINGBOT_REPO = os.environ.get(
    'LINGBOT_REPO',
    '/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map',
)
_LINGBOT_WEIGHTS = os.environ.get(
    'LINGBOT_WEIGHTS',
    '/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt',
)
_DINO_WEIGHTS = os.environ.get(
    'MEMNAV_DINO_WEIGHTS',
    str(Path(__file__).resolve().parents[3] / 'checkpoints' / 'depth_anything_v2_vits.pth'),
)
# Frames may live in a read-only squashfs overlay (mp3d pt1.sqf) while caches are
# written to a SEPARATE writable tree — MEMNAV_FEATURE_ROOT points at that tree
# (None = old behavior: cache sits beside the frames). window/num_scale/max_frame_num
# MUST match how the caches were precomputed (mp3d: window=32, num_scale=8, mfn=4096).
_FEATURE_ROOT = os.environ.get('MEMNAV_FEATURE_ROOT') or None
_WINDOW_SIZE = int(os.environ.get('MEMNAV_WINDOW', '32'))
_NUM_SCALE = int(os.environ.get('MEMNAV_NUM_SCALE', '8'))
_MAX_FRAME_NUM = int(os.environ.get('MEMNAV_MAX_FRAME_NUM', '4096'))
_STRICT_FEATURE_COVERAGE = _env_bool('MEMNAV_STRICT_FEATURE_COVERAGE', True)
_REQUIRE_VERSIONED_CACHE = _env_bool('MEMNAV_REQUIRE_VERSIONED_CACHE', False)
_EXPECTED_CACHE_SIGNATURE = os.environ.get('MEMNAV_EXPECTED_CACHE_SIGNATURE', '')
_USE_POSE_RELIABILITY_CONDITIONING = _env_bool(
    'MEMNAV_USE_POSE_RELIABILITY_CONDITIONING', False
)
_REQUIRE_GENERATED_POSE_CONVENTION = _env_bool(
    'MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION', False
)
_DATA_SPLIT = os.environ.get('MEMNAV_DATA_SPLIT', 'train')
_VALIDATION_FRACTION = float(os.environ.get('MEMNAV_VALIDATION_FRACTION', '0.1'))
_SPLIT_SEED = int(os.environ.get('MEMNAV_SPLIT_SEED', '0'))

memnav_exp_cfg = ExpCfg(
    name='memnav_train',
    model_name='memnav',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    checkpoint_folder='checkpoints/%s/ckpts',
    log_dir='checkpoints/%s/logs',
    local_rank=0,
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=1000,
        batch_size=8,
        lr=1e-4,
        num_workers=4,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type='cosine',
        logging_steps=int(os.environ.get('MEMNAV_LOGGING_STEPS', '10')),
        save_interval_steps=int(os.environ.get('MEMNAV_SAVE_STEPS', '25')),
        save_total_limit=int(os.environ.get('MEMNAV_SAVE_TOTAL_LIMIT', '3')),
        max_grad_norm=float(os.environ.get('MEMNAV_MAX_GRAD_NORM', '1.0')),
        gradient_accumulation_steps=int(os.environ.get('MEMNAV_GRAD_ACCUM', '1')),
        eval_samples=int(os.environ.get('MEMNAV_EVAL_SAMPLES', '16')),
        eval_interval_steps=int(os.environ.get('MEMNAV_EVAL_STEPS', '25')),
        eval_batch_size=int(os.environ.get('MEMNAV_EVAL_BATCH_SIZE', '4')),
        eval_seed=int(os.environ.get('MEMNAV_EVAL_SEED', '0')),
        # -1 keeps epoch-based production behavior.  A small positive value is
        # reserved for clean, bounded optimizer/gradient smoke runs.
        max_train_steps=int(os.environ.get('MEMNAV_MAX_TRAIN_STEPS', '-1')),
        bf16=_env_bool('MEMNAV_BF16', False),
        tf32=_env_bool('MEMNAV_TF32', True),
        save_filter_frozen_weights=True,
        load_from_ckpt=False,
        ckpt_to_load='',
        report_to=os.environ.get('MEMNAV_REPORT_TO', 'wandb'),
        # data + frozen-LingBot paths (override via MEMNAV_ROOT_DIR / LINGBOT_REPO / LINGBOT_WEIGHTS)
        root_dir=_ROOT_DIR,
        feature_root=_FEATURE_ROOT,
        strict_feature_coverage=_STRICT_FEATURE_COVERAGE,
        require_versioned_cache=_REQUIRE_VERSIONED_CACHE,
        expected_cache_signature=_EXPECTED_CACHE_SIGNATURE,
        require_generated_pose_convention=_REQUIRE_GENERATED_POSE_CONVENTION,
        data_split=_DATA_SPLIT,
        validation_fraction=_VALIDATION_FRACTION,
        split_seed=_SPLIT_SEED,
        sampling_mode=os.environ.get('MEMNAV_SAMPLING_MODE', 'random_leg'),
        sampling_seed=int(os.environ.get('MEMNAV_SAMPLING_SEED', '0')),
        lingbot_repo=_LINGBOT_REPO,
        lingbot_weights=_LINGBOT_WEIGHTS,
        # Required pretrained initialization for the trainable six-channel
        # current+goal DINO-S trunk. MemNav refuses to silently train it from random.
        novel_backbone_weights=_DINO_WEIGHTS,
        image_size=518,
        random_digit=False,
        # memory-partition geometry — MUST match the precompute (mp3d: 32/8/4096).
        # Read by MemNav_Dataset (window_size/num_scale) and LingBotStream (window/
        # num_scale/max_frame_num) so training reproduces the cached streaming exactly.
        window_size=_WINDOW_SIZE,
        num_scale=_NUM_SCALE,
        max_frame_num=_MAX_FRAME_NUM,
        # goal_append_warm's live-recompute depth before streaming the goal (deeper than
        # window_size on purpose): window_size's cold start at the window boundary starves
        # the goal's pose estimate (no real predecessors); goal_warm=64 empirically matches
        # a true continuous-stream oracle (scripts/diag_lingbot_pose_accuracy.py), while the
        # nominal window leaves ~30% avoidable error on the table. Eviction stays at
        # window_size — the model's own KV eviction trims back to that during the warm
        # recompute, which measured the same accuracy as never evicting at all.
        goal_warm=64,
        # policy / diffusion
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        num_diffusion_iters=10,
        # Raw-DINO revisit gate calibration.  These defaults were measured on
        # the training split; the learnable slope/bias operate on
        # (cosine - center) / width so both have healthy O(1) gradients.
        gate_center=float(os.environ.get('MEMNAV_GATE_CENTER', '0.94')),
        gate_width=float(os.environ.get('MEMNAV_GATE_WIDTH', '0.04')),
        gate_slope_init=float(os.environ.get('MEMNAV_GATE_SLOPE_INIT', '1.6')),
        gate_bias_init=float(os.environ.get('MEMNAV_GATE_BIAS_INIT', '0.0')),
        gate_lr_multiplier=float(
            os.environ.get('MEMNAV_GATE_LR_MULTIPLIER', '10.0')
        ),
        # Long-range geometry confidence is separate from semantic revisit
        # confidence.  It sees only online consistency cues and is supervised by
        # raw-bearing agreement during training.
        pose_scale_window=int(os.environ.get('MEMNAV_POSE_SCALE_WINDOW', '64')),
        pose_reliability_hidden=int(
            os.environ.get('MEMNAV_POSE_RELIABILITY_HIDDEN', '16')
        ),
        pose_reliability_init=float(
            os.environ.get('MEMNAV_POSE_RELIABILITY_INIT', '0.99')
        ),
        use_pose_reliability_conditioning=_USE_POSE_RELIABILITY_CONDITIONING,
        pose_reliability_lr_multiplier=float(
            os.environ.get('MEMNAV_POSE_RELIABILITY_LR_MULTIPLIER', '5.0')
        ),
        # loss weights (consumed by MemNavTrainer)
        w_retrieval=1.0,   # ranking InfoNCE (which candidate frame matches)
        w_gate=1.0,        # revisit/novel gate BCE (is there a match at all)
        # Scale-invariant translation-direction auxiliary. Metric x/y remains a
        # diagnostic because LingBot pose has a per-sequence canonical scale.
        w_aux_direction=float(os.environ.get('MEMNAV_W_AUX_DIRECTION', '0.2')),
        # Optional supervision for the *adapted* range coordinate consumed by
        # revisit_head.  It targets asinh(GT endpoint distance / observed-prefix
        # step / window), never a global LingBot-units-to-metres conversion.
        # Default-off preserves exact behavior of existing checkpoints/runs.
        w_aux_range=float(os.environ.get('MEMNAV_W_AUX_RANGE', '0.0')),
        aux_range_beta=float(os.environ.get('MEMNAV_AUX_RANGE_BETA', '0.1')),
        # Optional action-safe gradient surgery for the shared range coordinate.
        # A positive ratio removes range-gradient components that oppose the
        # diffusion action gradient, then caps the remainder relative to it.
        # Default-off preserves previous objective/backward behavior exactly.
        aux_range_grad_cap_ratio=float(
            os.environ.get('MEMNAV_AUX_RANGE_GRAD_CAP_RATIO', '0.0')
        ),
        # Scheduled exposure to retrieval's live anchor closes the train/eval
        # mismatch.  1->1 is the legacy all-positive teacher-forcing baseline;
        # experiments can decay toward a lower probability without a code edit.
        anchor_teacher_forcing_start=float(
            os.environ.get('MEMNAV_ANCHOR_TF_START', '1.0')
        ),
        anchor_teacher_forcing_end=float(
            os.environ.get('MEMNAV_ANCHOR_TF_END', '1.0')
        ),
        anchor_teacher_forcing_decay_steps=int(
            os.environ.get('MEMNAV_ANCHOR_TF_DECAY_STEPS', '0')
        ),
        w_pose_reliability=float(
            os.environ.get(
                'MEMNAV_W_POSE_RELIABILITY',
                '0.2' if _USE_POSE_RELIABILITY_CONDITIONING else '0.0',
            )
        ),
        ddp_find_unused_parameters=True,
    ),
    model=memnav_cfg,
)
