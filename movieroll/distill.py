# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""MovieRoll distillation entry point: Stage 2 (ODE regression) and Stage 3 (DMD).

    torchrun --nproc_per_node 8 distill_movieroll.py --mode stage2 \
        --config movieroll/configs/stage2_ode.yaml
    torchrun --nproc_per_node 8 distill_movieroll.py --mode stage3 \
        --config movieroll/configs/stage3_dmd_rolling.yaml

Both modes validate the YAML strictly before NCCL init and model allocation, so a configuration error
costs seconds instead of the minutes it takes to shard three 8.3B models across the mesh. Each stage
imports its own trainer inside its own branch, so one stage never pulls in the other's stack.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# Required top-level sections per mode. The validators check everything else.
_REQUIRED_FIELDS = {
    "stage2": ("exp_name", "logdir", "ode", "optimizer", "data", "video"),
    "stage3": (
        "exp_name",
        "logdir",
        "dmd",
        "dmd_phases",
        "optimizer",
        "fsdp",
        "streaming_config",
        "data",
        "video",
        "sampling",
    ),
}


def setup_logging(rank: int, log_dir: str):
    """Configure the stdlib logging the trainers write through."""
    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.INFO if rank == 0 else logging.WARNING
    log_format = f"[Rank {rank}] %(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = None
    if rank == 0:
        log_file = Path(log_dir) / "train.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger(__name__)
    if rank == 0:
        logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def attach_logdir_handler(rank: int, log_dir: str) -> None:
    """Add a file handler for the trainer's timestamped directory.

    The handler on the pre-timestamp directory stays attached, so rank 0 writes both train.log files.
    """
    if rank != 0:
        return
    log_file = Path(log_dir) / "train.log"
    log_format = f"[Rank {rank}] %(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger(__name__).info(f"Log file updated to: {log_file}")


def init_distributed():
    """Initialise the distributed environment, or fall back to one GPU.

    Returns (rank, world_size, device).

    The launch is detected with `is_torchelastic_launched()`, not with `'RANK' in os.environ`: importing
    `movieroll` fills in RANK / WORLD_SIZE / MASTER_PORT defaults, so an environment check would send a
    plain `python` launch down the NCCL branch too.
    """
    if dist.is_torchelastic_launched():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Bind the process before NCCL initialization so barrier/process-group
        # setup cannot guess the wrong device on multi-GPU nodes.
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        from movieroll.commons.parallel_states import init_nccl_process_group

        init_nccl_process_group(init_method='env://', device_id=device)

    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return rank, world_size, device


def load_config(config_path: str, mode: str, overrides=None):
    """Load a distillation YAML and apply the overrides; the validators check it afterwards."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))

    missing_fields = [f for f in _REQUIRED_FIELDS[mode] if f not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    return config


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="MovieRoll distillation: Stage 2 ODE regression or Stage 3 DMD"
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=("stage2", "stage3"),
        help="stage2 = ODE regression (the few-step initialisation); stage3 = DMD distillation",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the training config YAML",
    )
    parser.add_argument(
        "--override", type=str, action="append", default=[], metavar="KEY=VALUE",
        help="Override a config key, e.g. --override max_steps=4 (repeatable)",
    )
    return parser.parse_args(args)


def _run_stage2(args):
    from movieroll.trainer.ode.trainer import ODERegressionMainTrainer
    from movieroll.trainer.ode.config import validate_stage2_config

    config = load_config(args.config, "stage2", args.override)

    # Reject unsupported Stage2 modes before NCCL and model allocation.
    validate_stage2_config(config, mode="train")

    rank, world_size, device = init_distributed()
    logger = setup_logging(rank, config.logdir)

    # Every DP rank must take the SAME seed: DistributedSampler seeds one generator with seed + epoch and
    # strides a single shared permutation by rank, and the pivot / first-window / context-noise draws are
    # broadcast inside the SP group.
    seed = int(config.get("seed", 42))
    import random as _random

    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if rank == 0:
        logger.info(f"Global seed set: {seed}")

    if rank == 0:
        logger.info("=" * 80)
        logger.info("MovieRoll Stage 2: ODE regression")
        logger.info("Reference: CausVid (https://arxiv.org/abs/2412.07772)")
        logger.info("=" * 80)
        logger.info(f"Experiment: {config.exp_name}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Output: {config.logdir}")
        logger.info(f"Devices: {world_size} GPUs")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
        logger.info("=" * 80)

    try:
        trainer = ODERegressionMainTrainer(
            config=config,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        attach_logdir_handler(rank, config.logdir)
    except Exception as e:
        logger.exception(f"Failed to initialize ODERegressionMainTrainer: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

    try:
        if config.get("infer_only", False):
            # Inference path only: no training step and no 18 GB checkpoint.
            if rank == 0:
                logger.info("infer_only=true -> one validation pass, no training and no checkpoint")
            trainer._validate(0)
        else:
            trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Stage 2 ODE regression completed successfully!")
        logger.info("=" * 80)


def _run_stage3(args):
    from movieroll.trainer.dmd.trainer import DMDTrainer
    from movieroll.trainer.dmd.config import validate_stage3_config

    config = load_config(args.config, "stage3", args.override)

    # Configuration errors must fail before NCCL setup and before loading any model.
    validate_stage3_config(config)

    rank, world_size, device = init_distributed()
    logger = setup_logging(rank, config.logdir)

    if rank == 0:
        logger.info("=" * 80)
        logger.info("MovieRoll Stage 3: DMD distillation")
        logger.info("=" * 80)
        logger.info(f"Experiment: {config.exp_name}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Output: {config.logdir}")
        logger.info(f"Devices: {world_size} GPUs")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
        logger.info("=" * 80)

    try:
        trainer = DMDTrainer(
            config=config,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        attach_logdir_handler(rank, config.logdir)
    except Exception as e:
        logger.exception(f"Failed to initialize DMDTrainer: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Stage 3 DMD training completed successfully!")
        logger.info("=" * 80)


def main(args=None):
    if args is None:
        args = parse_args()
    if args.mode == "stage2":
        _run_stage2(args)
    else:
        _run_stage3(args)


if __name__ == "__main__":
    main()
