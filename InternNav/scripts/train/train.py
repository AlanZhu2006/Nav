import hashlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from pathlib import Path
import torch.distributed as dist

from typing import Optional

import torch
import tyro
from pydantic import BaseModel
from torch.utils.data import Subset
from transformers import TrainerCallback, TrainingArguments

from internnav.model.utils.logger import MyLogger
from internnav.model.utils.utils import load_dataset
from scripts.train.configs import (
    cma_exp_cfg,
    cma_plus_exp_cfg,
    rdp_exp_cfg,
    seq2seq_exp_cfg,
    seq2seq_plus_exp_cfg,
    navdp_exp_cfg,
    logoplanner_exp_cfg,
    memnav_exp_cfg,
)
import sys
from datetime import datetime

class TrainCfg(BaseModel):
    """Training configuration class.

    Fields below model_name are optional overrides applied on top of the
    per-model exp_cfg selected via --model-name. Leave unset to use the
    values defined in scripts/train/configs/<model_name>.py.
    """

    name: str = 'cma_train'  # Experiment name
    model_name: str = 'cma'  # includes cma/seq2seq/rdp/navdp/logoplanner/memnav variants

    # il.* overrides
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    epochs: Optional[int] = None
    lr: Optional[float] = None
    root_dir: Optional[str] = None
    dataset_navdp: Optional[str] = None
    ckpt_to_load: Optional[str] = None
    load_from_ckpt: Optional[bool] = None
    resume_from_checkpoint: Optional[str] = None

    # exp-level overrides
    torch_gpu_ids: Optional[list[int]] = None
    seed: Optional[int] = None


def _apply_overrides(exp_cfg, cli: 'TrainCfg') -> None:
    """Apply non-None CLI overrides onto exp_cfg in place."""
    il_fields = {
        'batch_size', 'num_workers', 'epochs', 'lr',
        'root_dir', 'dataset_navdp', 'ckpt_to_load', 'load_from_ckpt',
        'resume_from_checkpoint',
    }
    exp_fields = {'torch_gpu_ids', 'seed'}
    for field in il_fields:
        val = getattr(cli, field)
        if val is not None:
            setattr(exp_cfg.il, field, val)
            print(f'[override] il.{field} = {val}')
    for field in exp_fields:
        val = getattr(cli, field)
        if val is not None:
            setattr(exp_cfg, field, val)
            print(f'[override] {field} = {val}')


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(self, run_name: str, exp_cfg_dir: Path | None = None):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f'checkpoint-{state.global_step}'  # noqa: F841


def _make_dir(config):
    config.tensorboard_dir = config.tensorboard_dir % config.name  
    config.checkpoint_folder = config.checkpoint_folder % config.name
    config.log_dir = config.log_dir % config.name
    config.output_dir = config.output_dir % config.name
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir,exist_ok=True)
    if not os.path.exists(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder,exist_ok=True)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir,exist_ok=True)


def main(config, model_class, model_config_class):
    try:
        """Main training function."""
        _make_dir(config)

        print(f"=== Start training ===")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Environment variables:")
        print(f"  RANK: {os.getenv('RANK', 'Not set')}")
        print(f"  LOCAL_RANK: {os.getenv('LOCAL_RANK', 'Not set')}")
        print(f"  WORLD_SIZE: {os.getenv('WORLD_SIZE', 'Not set')}")
        print(f"  MASTER_ADDR: {os.getenv('MASTER_ADDR', 'Not set')}")
        print(f"  MASTER_PORT: {os.getenv('MASTER_PORT', 'Not set')}")

        if config.model_name in ("navdp", "logoplanner", "memnav"):
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            rank = int(os.getenv('RANK', '0'))
            
            # Set CUDA device for each process
            device_id = local_rank
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
            print(f"World size: {world_size}, Local rank: {local_rank}, Global rank: {rank}")
            
            # Initialize distributed training environment
            if world_size > 1:
                try:
                    dist.init_process_group(
                        backend='nccl',
                        init_method='env://',
                        world_size=world_size,
                        rank=rank
                    )
                    print("Distributed initialization SUCCESS")
                except Exception as e:
                    print(f"Distributed initialization FAILED: {str(e)}")
                    world_size = 1

            print("="*50)
            print("After distributed init:")
            print(f"LOCAL_RANK: {local_rank}")
            print(f"WORLD_SIZE: {world_size}")

        if dist.is_initialized():
            print(f"Dist WORLD_SIZE: {dist.get_world_size()}")
            print(f"Dist RANK: {dist.get_rank()}")
        else:
            print("Distributed NOT initialized")

        # ------------ load model ------------
        model_cfg = model_config_class(model_cfg=config.model_dump())
        if config.il.ckpt_to_load:
            print(f"load model from:{config.il.ckpt_to_load}")
        model = model_class.from_pretrained(pretrained_model_name_or_path=config.il.ckpt_to_load, config=model_cfg)
        if config.model_name in ("navdp", "logoplanner", "memnav"):
            model.to(device)
            # Check that all parameters and buffers are on the correct device
            for name, param in model.named_parameters():
                if param.device != device:
                    print(f"Parameter {name} is on wrong device {param.device}, should be moved to {device}")
                    param.data = param.data.to(device)

            for name, buffer in model.named_buffers():
                if buffer.device != device:
                    print(f"Buffer {name} is on wrong device {buffer.device}, should be moved to {device}")
                    buffer.data = buffer.data.to(device)
            
            # If distributed training, wrap the model with DDP
            if world_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True
                )
        # ------------ load logger ------------
        train_logger_filename = os.path.join(config.log_dir, 'train.log')
        if dist.is_initialized() and dist.get_rank() == 0:
            train_logger = MyLogger(
                name='train', level=logging.INFO, format_str='%(asctime)-15s %(message)s', filename=train_logger_filename
            )
        else:
            # Other processes use console logging
            train_logger = MyLogger(
                name='train', level=logging.INFO, format_str='%(asctime)-15s %(message)s'
            )
        transformers_logger = logging.getLogger("transformers")
        if transformers_logger.hasHandlers():
            transformers_logger.handlers = []
        if config.model_name in ("navdp", "logoplanner", "memnav") and local_rank in [0, -1]:  # Only main process or non-distributed
            transformers_logger.addHandler(train_logger.handlers[0])
        transformers_logger.setLevel(logging.INFO)


        # ------------ load dataset ------------
        eval_dataset_data = None
        if config.model_name == "navdp":
            from internnav.dataset.navdp_dataset_lerobot import NavDP_Base_Datset
            train_dataset_data = NavDP_Base_Datset(config.il.root_dir,
                                    config.il.dataset_navdp,
                                    config.il.memory_size,
                                    config.il.predict_size,
                                    config.il.batch_size,
                                    config.il.image_size,
                                    config.il.scene_scale,
                                    preload = config.il.preload,
                                    random_digit = config.il.random_digit,
                                    prior_sample = config.il.prior_sample)
        elif config.model_name == "logoplanner":
            from internnav.dataset.logoplanner_dataset_lerobot import LoGoPlanner_Dataset
            train_dataset_data = LoGoPlanner_Dataset(
                config.il.root_dir,
                preload_path=config.il.dataset_navdp,
                memory_size=config.il.memory_size,
                predict_size=config.il.predict_size,
                batch_size=config.il.batch_size,
                image_size=config.il.image_size,
                scene_data_scale=config.il.scene_scale,
                preload=config.il.preload,
                random_digit=config.il.random_digit,
                prior_sample=config.il.prior_sample,
                context_size=config.il.context_size,
                context_image_height=config.il.context_image_height,
                context_image_width=config.il.context_image_width,
                depth_max=config.il.depth_max,
                depth_min=config.il.depth_min,
            )
        elif config.model_name == "memnav":
            from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
            train_dataset_data = MemNav_Dataset(
                config.il.root_dir,
                predict_size=config.il.predict_size,
                image_size=config.il.image_size,
                lingbot_repo=config.il.lingbot_repo,
                feature_root=getattr(config.il, 'feature_root', None),
                window_size=getattr(config.il, 'window_size', 8),
                num_scale=getattr(config.il, 'num_scale', 8),
                strict_feature_coverage=getattr(config.il, 'strict_feature_coverage', True),
                require_generated_pose_convention=getattr(
                    config.il, 'require_generated_pose_convention', False
                ),
                data_split=getattr(config.il, 'data_split', 'all'),
                validation_fraction=getattr(config.il, 'validation_fraction', 0.1),
                split_seed=getattr(config.il, 'split_seed', 0),
                sampling_mode=getattr(config.il, 'sampling_mode', 'random_leg'),
                sampling_seed=getattr(config.il, 'sampling_seed', 0),
            )
        else:
            if '3dgs' in config.il.lmdb_features_dir or '3dgs' in config.il.lmdb_features_dir:
                dataset_root_dir = config.il.dataset_six_floor_root_dir
                dataset_type = '3dgs'
            elif 'grutopia' in config.il.lmdb_features_dir:
                dataset_root_dir = config.il.dataset_grutopia10_root_dir
                dataset_type = 'grutopia'
            else:
                dataset_root_dir = config.il.dataset_r2r_root_dir
                dataset_type = 'r2r'
            train_dataset_data = load_dataset(dataset_root_dir, 'train', logger=train_logger, dataset_type=dataset_type)
            global_batch_size = config.il.batch_size * len(config.torch_gpu_ids)

        if config.model_name == 'memnav' and int(getattr(config.il, 'eval_samples', 0)) > 0:
            if getattr(config.il, 'data_split', 'all') != 'train':
                raise ValueError(
                    'Fixed validation requires MEMNAV_DATA_SPLIT=train so held-out scenes '
                    'are not present in the training population.'
                )
            fixed_val = MemNav_Dataset(
                config.il.root_dir,
                predict_size=config.il.predict_size,
                image_size=config.il.image_size,
                lingbot_repo=config.il.lingbot_repo,
                feature_root=getattr(config.il, 'feature_root', None),
                window_size=getattr(config.il, 'window_size', 8),
                num_scale=getattr(config.il, 'num_scale', 8),
                strict_feature_coverage=getattr(config.il, 'strict_feature_coverage', True),
                require_generated_pose_convention=getattr(
                    config.il, 'require_generated_pose_convention', False
                ),
                data_split='val',
                validation_fraction=getattr(config.il, 'validation_fraction', 0.1),
                split_seed=getattr(config.il, 'split_seed', 0),
                sampling_mode='fixed_leg',
                sampling_seed=getattr(config.il, 'eval_seed', 0),
            )
            eval_count = min(int(config.il.eval_samples), len(fixed_val))
            generator = torch.Generator().manual_seed(int(getattr(config.il, 'eval_seed', 0)))
            revisit_indices, novel_indices = [], []
            for index, sample in enumerate(fixed_val.samples):
                k, _ = fixed_val._sample_k_and_digit(
                    sample, int(sample['k_lo']), int(sample['k_hi'])
                )
                *_masks, null_pos = fixed_val._build_label(sample, k)
                (novel_indices if null_pos else revisit_indices).append(index)
            target_revisit = min(eval_count // 2, len(revisit_indices))
            target_novel = min(eval_count - target_revisit, len(novel_indices))
            if target_revisit + target_novel < eval_count:
                target_revisit = min(
                    len(revisit_indices), eval_count - target_novel
                )
            rev_order = torch.randperm(len(revisit_indices), generator=generator)[:target_revisit]
            nov_order = torch.randperm(len(novel_indices), generator=generator)[:target_novel]
            indices = sorted(
                [revisit_indices[i] for i in rev_order.tolist()]
                + [novel_indices[i] for i in nov_order.tolist()]
            )
            eval_dataset_data = Subset(fixed_val, indices)
            subset_manifest = (
                f'{fixed_val.dataset_fingerprint}\n'
                + ','.join(str(index) for index in indices)
            )
            eval_dataset_data.dataset_fingerprint = hashlib.sha256(
                subset_manifest.encode('utf-8')
            ).hexdigest()
            print(
                f'[eval] fixed held-out subset: {len(indices)}/{len(fixed_val)} samples; '
                f'revisit={target_revisit}, novel={target_novel}; '
                f'fingerprint={eval_dataset_data.dataset_fingerprint}'
            )

        # ------------ data_loader ------------
        if config.model_name in ['cma', 'seq2seq']:
            from internnav.dataset.cma_lerobot_dataset import CMALerobotDataset, cma_collate_fn
            from internnav.trainer import CMATrainer
            policy_trainer = CMATrainer
            train_dataset = CMALerobotDataset(
                config,
                config.il.lerobot_features_dir,
                config.il.use_iw,
                dataset_data=train_dataset_data,
                inflection_weight_coef=config.il.inflection_weight_coef,
                lmdb_map_size=config.il.lmdb_map_size,
                batch_size=config.il.batch_size,
            )
            collate_fn = cma_collate_fn

        elif config.model_name == 'rdp':
            from internnav.dataset.rdp_lerobot_dataset import RDP_LerobotDataset, rdp_collate_fn
            from internnav.trainer import RDPTrainer
            policy_trainer = RDPTrainer
            train_dataset = RDP_LerobotDataset(
                config,
                config.il.lerobot_features_dir,
                dataset_data=train_dataset_data,
                batch_size=config.il.batch_size,  
            )
            collate_fn = rdp_collate_fn(global_batch_size=global_batch_size)
        elif config.model_name == 'navdp':
            from internnav.dataset.navdp_dataset_lerobot import navdp_collate_fn
            from internnav.trainer import NavDPTrainer
            policy_trainer = NavDPTrainer
            train_dataset = train_dataset_data
            collate_fn = navdp_collate_fn
        elif config.model_name == 'logoplanner':
            from internnav.dataset.logoplanner_dataset_lerobot import logoplanner_collate_fn
            from internnav.trainer import LoGoPlannerTrainer
            policy_trainer = LoGoPlannerTrainer
            train_dataset = train_dataset_data
            collate_fn = logoplanner_collate_fn
        elif config.model_name == 'memnav':
            from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
            from internnav.trainer import MemNavTrainer
            policy_trainer = MemNavTrainer
            train_dataset = train_dataset_data
            collate_fn = memnav_collate_fn

        # ------------ training args ------------
        is_memnav = config.model_name == 'memnav'
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            run_name=config.name,
            remove_unused_columns=False,
            deepspeed='',
            gradient_checkpointing=False,
            bf16=bool(getattr(config.il, 'bf16', False)) if is_memnav else False,
            tf32=bool(getattr(config.il, 'tf32', False)) if is_memnav else False,
            per_device_train_batch_size=config.il.batch_size,
            per_device_eval_batch_size=(
                int(getattr(config.il, 'eval_batch_size', config.il.batch_size))
                if is_memnav else config.il.batch_size
            ),
            gradient_accumulation_steps=(
                int(getattr(config.il, 'gradient_accumulation_steps', 1)) if is_memnav else 1
            ),
            dataloader_num_workers=config.il.num_workers,
            dataloader_pin_memory=False,
            optim='adamw_torch',
            learning_rate=config.il.lr,
            weight_decay=float(getattr(config.il, 'weight_decay', 0.0) or 0.0),
            warmup_ratio=float(getattr(config.il, 'warmup_ratio', 0.0) or 0.0),
            lr_scheduler_type=(getattr(config.il, 'lr_scheduler_type', 'cosine') or 'cosine'),
            max_grad_norm=float(getattr(config.il, 'max_grad_norm', 1.0) or 1.0),
            logging_steps=(int(getattr(config.il, 'logging_steps', 10)) if is_memnav else 10),
            num_train_epochs=config.il.epochs,
            eval_strategy=('steps' if is_memnav and eval_dataset_data is not None else 'no'),
            eval_steps=(int(getattr(config.il, 'eval_interval_steps', 25)) if is_memnav else None),
            save_strategy='steps' if is_memnav else 'epoch',
            save_steps=(int(getattr(config.il, 'save_interval_steps', 25)) if is_memnav else 500),
            save_total_limit=(int(getattr(config.il, 'save_total_limit', 3)) if is_memnav else 8),
            report_to=config.il.report_to,
            # Without an explicit label name, HF's prediction_step treats a
            # MemNav batch as unlabeled and bypasses MemNavTrainer.compute_loss,
            # calling model(**batch) directly during scheduled evaluation.
            label_names=['batch_labels'] if is_memnav else None,
            seed=config.seed,
            data_seed=config.seed,
            do_eval=is_memnav and eval_dataset_data is not None,
            prediction_loss_only=is_memnav,
            ddp_find_unused_parameters=config.il.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=100,
            torch_compile_mode=None,
            dataloader_drop_last=True,
            disable_tqdm=True,
            log_level="info"
        )

        # Create the trainer
        trainer = policy_trainer(
            config=config, model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset_data, data_collator=collate_fn
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = config.name
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=config.log_dir)
        trainer.add_callback(ckpt_format_callback)

        resume = getattr(config.il, 'resume_from_checkpoint', None)
        if isinstance(resume, str):
            normalized = resume.strip().lower()
            if normalized in {'', 'none', 'false', '0'}:
                resume = None
            elif normalized == 'auto':
                from transformers.trainer_utils import get_last_checkpoint
                resume = get_last_checkpoint(config.output_dir)
                print(f"[resume] auto -> {resume or 'no checkpoint; starting fresh'}")
        trainer.train(resume_from_checkpoint=resume)
        if train_logger:
            for handler in train_logger.handlers:
                handler.flush()
    except Exception as e:
        import traceback
        print(f"Unhandled exception: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        
        # If distributed environment, ensure all processes exit
        if dist.is_initialized():
            dist.destroy_process_group()
        
        raise


if __name__ == '__main__':
    # Parse command line arguments using tyro
    config = tyro.cli(TrainCfg)

    # Print configuration information
    print('\n' + '=' * 50)
    print('FINE-TUNING CONFIGURATION:')
    print('=' * 50)
    for key, value in vars(config).items():
        print(f'{key}: {value}')
    print('=' * 50 + '\n')

    # Import only the selected policy. Several legacy policies use optional,
    # ignored external repos (for example LongCLIP) that MemNav does not need.
    supported = {
        'seq2seq', 'seq2seq_plus', 'cma', 'cma_plus',
        'rdp', 'navdp', 'logoplanner', 'memnav',
    }
    if config.model_name not in supported:
        raise ValueError(
            f'Invalid model name: {config.model_name}. Supported models are: {sorted(supported)}'
        )
    if config.model_name in {'seq2seq', 'seq2seq_plus'}:
        from internnav.model.basemodel.seq2seq.seq2seq_policy import Seq2SeqModelConfig, Seq2SeqNet
        exp_cfg = seq2seq_exp_cfg if config.model_name == 'seq2seq' else seq2seq_plus_exp_cfg
        model_class, model_config_class = Seq2SeqNet, Seq2SeqModelConfig
    elif config.model_name in {'cma', 'cma_plus'}:
        from internnav.model.basemodel.cma.cma_policy import CMAModelConfig, CMANet
        exp_cfg = cma_exp_cfg if config.model_name == 'cma' else cma_plus_exp_cfg
        model_class, model_config_class = CMANet, CMAModelConfig
    elif config.model_name == 'rdp':
        from internnav.model.basemodel.rdp.rdp_policy import RDPModelConfig, RDPNet
        exp_cfg, model_class, model_config_class = rdp_exp_cfg, RDPNet, RDPModelConfig
    elif config.model_name == 'navdp':
        from internnav.model.basemodel.navdp.navdp_policy import NavDPModelConfig, NavDPNet
        exp_cfg, model_class, model_config_class = navdp_exp_cfg, NavDPNet, NavDPModelConfig
    elif config.model_name == 'logoplanner':
        from internnav.model.basemodel.logoplanner.logoplanner_policy import LoGoPlannerModelConfig, LoGoPlannerNet
        exp_cfg, model_class, model_config_class = (
            logoplanner_exp_cfg, LoGoPlannerNet, LoGoPlannerModelConfig
        )
    else:
        from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy
        exp_cfg, model_class, model_config_class = memnav_exp_cfg, MemNavPolicy, MemNavModelConfig
    exp_cfg.name = config.name
    _apply_overrides(exp_cfg, config)
    exp_cfg.num_gpus = len(exp_cfg.torch_gpu_ids)
    exp_cfg.world_size = exp_cfg.num_gpus

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        exp_cfg.num_gpus <= available_gpus
    ), f'Number of GPUs requested ({exp_cfg.num_gpus}) is greater than the available GPUs ({available_gpus})'
    assert exp_cfg.num_gpus > 0, 'Number of GPUs must be greater than 0'
    print(f'Using {exp_cfg.num_gpus} GPUs')

    main(exp_cfg, model_class, model_config_class)
