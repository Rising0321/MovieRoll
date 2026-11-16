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

"""Saving and resuming, in the three on-disk layouts the training chains use.

Each layout is its own artifact format:

* SFT (`save_checkpoint_without_optim` / `resume_training`) writes one directory per update step holding the
  full-tensor weights, the full-tensor EMA weights, the per-rank scalar-states pickle and a config.json. It
  carries no optimizer state, so an SFT resume restarts the optimizer;
* distillation (`save_multi_model_checkpoint` / `load_multi_model_checkpoint`) writes one sharded
  distributed-checkpoint directory per model, because its models and optimizers are far too large to gather.
  The distillation trainer saves and restores its own EMA shards, which is why nothing here takes an EMA
  argument;
* the ODE trainer borrows only the two optimizer helpers.
"""

import gc
import json
import logging
import os
import pickle as pkl

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from movieroll.utils.torch_utils import count_failed_ranks

logger = logging.getLogger(__name__)


def _raise_if_checkpoint_stage_failed(local_error, context):
    """Synchronize rank-local checkpoint I/O failures before any next collective."""

    # Without a process group there is nothing to synchronize: re-raise the local error unwrapped.
    if not (dist.is_available() and dist.is_initialized()):
        if local_error is not None:
            raise local_error
        return
    backend = dist.get_backend()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    failure_count = count_failed_ranks(local_error is not None, device)
    if failure_count:
        if local_error is not None:
            raise RuntimeError(
                f"{context} failed on {failure_count} rank(s)"
            ) from local_error
        raise RuntimeError(
            f"{context} failed on {failure_count} peer rank(s)"
        )


def _iter_state_tensors(optim_state_dict):
    """Yield every real tensor in an optimizer state dict, skipping scalar markers like use_muon / step.

    Counting these is the only signal that tells a restored optimizer state from one that was read and then
    dropped: distributed checkpointing fills only the keys it is given and ignores whatever else is in storage.
    """
    state = optim_state_dict.get("state", optim_state_dict)
    if not isinstance(state, dict):
        return
    for per_param in state.values():
        if not isinstance(per_param, dict):
            continue
        for v in per_param.values():
            if torch.is_tensor(v) and v.dim() > 0:
                yield v


def _materialize_optimizer_state(optimizer):
    """Force the optimizer to allocate its real per-parameter state (momentum / moment1 / moment2).

    Must run before a distributed-checkpoint load: torch's `_init_optim_state` returns early whenever
    `optim.state` is non-empty, and Muon writes a `use_muon` marker per parameter at construction, so the
    state dict handed to the load would carry no momentum keys and the stored ones would be ignored without
    error. lr=0 keeps the step below an identity on the weights, weight decay included.
    """
    params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
    # Already carrying real momentum: nothing to do.
    has_real_state = any(
        torch.is_tensor(v)
        for p in params
        for v in optimizer.state.get(p, {}).values()
    )
    if has_real_state:
        return

    created_grad = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
            created_grad.append(p)

    lrs = []
    for g in optimizer.param_groups:
        lrs.append(g.get("lr", None))
        if "lr" in g:
            g["lr"] = torch.tensor(0.0) if isinstance(g["lr"], torch.Tensor) else 0.0
    try:
        optimizer.step()
    finally:
        for g, lr in zip(optimizer.param_groups, lrs):
            if lr is not None:
                g["lr"] = lr
    for p in created_grad:
        p.grad = None
    optimizer.zero_grad(set_to_none=True)


def save_checkpoint_without_optim(transformer, rank, output_dir, scalar_states, ema_model):
    """Write one SFT checkpoint directory: weights, EMA weights, scalar states and config.

    Every rank must call this -- the full-tensor gathers and the scalar-state all-gather are collectives. Only
    rank 0 writes the files.
    """
    cpu_state = transformer.state_dict()
    cpu_state = {k: v.full_tensor().cpu().clone() for k, v in cpu_state.items()}

    # The EMA class reserves state_dict() for this rank's shards; the checkpoint needs the full tensors, and
    # that gather is a collective, so it runs outside the rank guard below.
    ema_cpu_state = ema_model.gather_full_state_dict()
    global_step = scalar_states.update_steps
    if dist.is_available() and dist.is_initialized():
        scalar_states_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(scalar_states_list, scalar_states.to_dict())

    # Only rank 0 writes the files; every rank had to take part in the collectives above.
    if rank == 0:
        save_dir = os.path.join(output_dir, 'checkpoints', f"step-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        ema_weight_path = os.path.join(save_dir, "ema_diffusion_pytorch_model.safetensors")
        save_file(ema_cpu_state, ema_weight_path)

        scalar_states_path = os.path.join(save_dir, "scalar_states.pkl")
        with open(scalar_states_path, "wb") as f:
            pkl.dump(scalar_states_list, f)

        config_dict = dict(transformer.config)
        if "dtype" in config_dict:
            # the build dtype is a torch.dtype and is not JSON serialisable
            del config_dict["dtype"]
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)


def resume_training(model, checkpoint_dir):
    """Load an SFT checkpoint into `model` and return it with the raw scalar-state list.

    Must be called BEFORE the model is FSDP-wrapped: the load is a plain local one into unsharded fp32
    parameters, which is also what lets the double-stream q/k/v -> qkv merge hook fire. The weights read are
    the EMA ones, deliberately: the same directory also holds the live ones, and EMA training is always on,
    so the EMA weights are the ones worth continuing from. The returned list holds one entry per rank of the run that wrote it, which
    `ScalarStates.from_pretrained` reads at any world size, so it stays a list rather than a dict.
    """
    weight_name = "ema_diffusion_pytorch_model.safetensors"
    weight_path = os.path.join(checkpoint_dir, weight_name)
    scalar_states_path = os.path.join(checkpoint_dir, "scalar_states.pkl")

    data = load_file(weight_path)
    model_weights = {k: v for k, v in data.items() if k != "__metadata__"}

    model.load_state_dict(model_weights, strict=True)

    with open(scalar_states_path, "rb") as f:
        scalar_states_list = pkl.load(f)
    return model, scalar_states_list


def save_checkpoint_with_optim(
    model,
    optimizer,
    rank,
    output_dir,
    step,
    model_name="model",
    clear_cache_after_save=True,
    dp_shard_sp_pg=None,
    is_save_rank=True,
):
    """Write one sharded model + optimizer checkpoint for `model_name`; each rank writes its own shard.

    All ranks must call this or NCCL times out -- the state-dict collection is collective. Under hybrid FSDP
    only the dp_replicate_rank == 0 group writes (`is_save_rank`), with `dp_shard_sp_pg` as the process group
    so the scatter stays inside that group instead of going global.
    """
    from torch.distributed.checkpoint import save as dist_checkpoint_save
    from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

    save_dir = os.path.join(output_dir, 'checkpoints', f"step-{step}")

    if rank == 0:
        print(f"[Rank {rank}] Saving sharded checkpoint to {save_dir}")
        print(f"[Rank {rank}] Saving {model_name} (sharded mode - all ranks save in parallel)...")

    state_dict_options = StateDictOptions(
        full_state_dict=False,  # Sharded mode
        cpu_offload=True,       # Offload to CPU
    )

    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer, options=state_dict_options
    )

    # A state dict collected before the first real optimizer step holds only Muon's `use_muon` markers and no
    # momentum, at a plausible file size, and no later resume would complain -- so report it at write time.
    if optimizer is not None and rank == 0:
        n_state = sum(1 for _ in _iter_state_tensors(optimizer_state_dict))
        if n_state == 0:
            print(f"[Rank {rank}] WARNING: {model_name}'s optimizer state dict holds no momentum tensors "
                  f"(it was probably saved before the first optimizer.step()). This checkpoint can restore "
                  f"the weights, but a resume will start Muon/AdamW momentum from zero.")
        else:
            print(f"[Rank {rank}] {model_name} optimizer state tensors: {n_state}")

    if is_save_rank:
        model_checkpoint_dir = os.path.join(save_dir, f"{model_name}_model")
        dist_checkpoint_save(
            state_dict={"model": model_state_dict},
            checkpoint_id=model_checkpoint_dir,
            process_group=dp_shard_sp_pg,
        )

        if rank == 0:
            print(f"[Rank {rank}] Saved {model_name} model to {model_checkpoint_dir}")

    del model_state_dict
    if clear_cache_after_save:
        torch.cuda.empty_cache()
        gc.collect()

    if is_save_rank:
        optimizer_checkpoint_dir = os.path.join(save_dir, f"{model_name}_optimizer")
        dist_checkpoint_save(
            state_dict={"optimizer": optimizer_state_dict},
            checkpoint_id=optimizer_checkpoint_dir,
            process_group=dp_shard_sp_pg,
        )

        if rank == 0:
            print(f"[Rank {rank}] Saved {model_name} optimizer to {optimizer_checkpoint_dir}")

    del optimizer_state_dict
    if clear_cache_after_save:
        torch.cuda.empty_cache()
        gc.collect()

    metadata_error = None
    if rank == 0:
        try:
            metadata = {
                "step": step,
                "model_name": model_name,
                "checkpoint_type": "sharded",
            }
            metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as exc:
            metadata_error = exc
    _raise_if_checkpoint_stage_failed(
        metadata_error, f"{model_name} checkpoint metadata save"
    )


def save_multi_model_checkpoint(
    models_dict,
    optimizers_dict,
    rank,
    output_dir,
    step,
    clear_cache_between_models=True,
    fsdp_dp_mesh=None
):
    """Save several models (e.g. generator + critic) and their optimizers, one at a time to bound memory.

    All ranks must call this or NCCL times out -- each model's state-dict collection is collective.
    `models_dict` and `optimizers_dict` are keyed by model name.
    """

    # Hybrid FSDP: let a single shard group (dp_replicate_rank == 0) write, so the global
    # scatter_object_list does not time out at large scale (256+ ranks).
    dp_shard_sp_pg = None
    is_save_rank = True  # by default every rank takes part
    if fsdp_dp_mesh is not None and fsdp_dp_mesh.ndim == 2:
        try:
            # fsdp_dp_mesh is the 2D ("dp_replicate", "dp_shard_sp") mesh
            dp_replicate_rank = fsdp_dp_mesh.get_local_rank(mesh_dim=0)  # the dp_replicate dimension
            is_save_rank = (dp_replicate_rank == 0)
            dp_shard_sp_pg = fsdp_dp_mesh.get_group(mesh_dim=1)
            if rank == 0:
                shard_group_size = dist.get_world_size(dp_shard_sp_pg)
                print(f"[Rank {rank}] Hybrid FSDP checkpoint: shard_group_size={shard_group_size}, "
                      f"dp_replicate_rank={dp_replicate_rank}, is_save_rank={is_save_rank}")
        except Exception as e:
            if rank == 0:
                print(f"[Rank {rank}] WARNING: Could not setup hybrid checkpoint ({e}), "
                      f"all ranks will save")
            dp_shard_sp_pg = None
            is_save_rank = True

    directory_error = None
    if rank == 0:
        try:
            save_dir = os.path.join(output_dir, 'checkpoints', f"step-{step}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"[Rank {rank}] Saving checkpoint to {save_dir}")
        except Exception as exc:
            directory_error = exc
    _raise_if_checkpoint_stage_failed(
        directory_error, "checkpoint directory creation"
    )

    for model_name, model in models_dict.items():
        if rank == 0:
            print(f"[Rank {rank}] Saving {model_name}...")

        optimizer = optimizers_dict.get(model_name)

        save_checkpoint_with_optim(
            model=model,
            optimizer=optimizer,
            rank=rank,
            output_dir=output_dir,
            step=step,
            model_name=model_name,
            clear_cache_after_save=True,
            dp_shard_sp_pg=dp_shard_sp_pg,
            is_save_rank=is_save_rank,
        )

        if clear_cache_between_models:
            torch.cuda.empty_cache()
            gc.collect()

    metadata_error = None
    if rank == 0:
        try:
            global_metadata = {
                "step": step,
                "models": list(models_dict.keys()),
            }
            metadata_path = os.path.join(save_dir, "checkpoint_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(global_metadata, f, indent=4)
            print(f"[Rank {rank}] All models saved to {save_dir}")
        except Exception as exc:
            metadata_error = exc
    _raise_if_checkpoint_stage_failed(
        metadata_error, "global checkpoint metadata save"
    )

    torch.cuda.empty_cache()
    gc.collect()

def load_multi_model_checkpoint(
    models_dict,
    optimizers_dict,
    checkpoint_dir,
    strict=True
):
    """Load the sharded layout `save_multi_model_checkpoint` writes; return the step it recorded.

    A model without an entry in `optimizers_dict` gets its weights only.
    """
    from torch.distributed.checkpoint import load as dist_checkpoint_load
    from torch.distributed.checkpoint.state_dict import (
        set_state_dict, StateDictOptions,
        get_optimizer_state_dict, set_optimizer_state_dict,
    )

    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    with open(metadata_path) as f:
        global_metadata = json.load(f)

    step = global_metadata["step"]

    for model_name, model in models_dict.items():
        logger.info(f"Loading {model_name} from sharded checkpoint...")

        optimizer = optimizers_dict.get(model_name)

        model_checkpoint_dir = os.path.join(checkpoint_dir, f"{model_name}_model")
        model_state_dict = {"model": model.state_dict()}
        dist_checkpoint_load(
            state_dict=model_state_dict,
            checkpoint_id=model_checkpoint_dir,
        )

        state_dict_options = StateDictOptions(full_state_dict=False, strict=strict)
        set_state_dict(model, [], model_state_dict=model_state_dict["model"], optim_state_dict={}, options=state_dict_options)
        logger.info(f"Loaded {model_name} model from {model_checkpoint_dir}")

        optimizer_checkpoint_dir = os.path.join(checkpoint_dir, f"{model_name}_optimizer")
        if optimizer is not None and os.path.isdir(optimizer_checkpoint_dir):
            # The per-parameter state has to be materialised first, or the state dict handed to distributed
            # checkpointing has no momentum keys and the stored ones are ignored without error.
            _materialize_optimizer_state(optimizer)
            optim_state_dict = get_optimizer_state_dict(
                model, optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            )
            n_before = sum(1 for _ in _iter_state_tensors(optim_state_dict))
            dist_checkpoint_load(
                state_dict={"optimizer": optim_state_dict},
                checkpoint_id=optimizer_checkpoint_dir,
            )
            set_optimizer_state_dict(
                model, optimizer, optim_state_dict=optim_state_dict,
                options=StateDictOptions(full_state_dict=False),
            )
            if n_before == 0:
                # An empty state dict means the load filled nothing and restored no momentum, silently.
                raise RuntimeError(
                    f"resume: {model_name}'s optimizer state dict is empty, so distributed checkpointing "
                    f"would silently skip every momentum tensor in {optimizer_checkpoint_dir}. Check that "
                    "_materialize_optimizer_state took effect on this optimizer.")
            logger.info(f"Loaded {model_name} optimizer ({n_before} state tensors) "
                        f"from {optimizer_checkpoint_dir}")
        elif optimizer is not None:
            # A weights-only checkpoint is allowed (it can serve as an initialisation), but the momentum it
            # cannot restore has to be said loudly.
            logger.warning(f"{optimizer_checkpoint_dir} does not exist -- {model_name}'s optimizer momentum "
                           f"restarts from zero (Muon's effective horizon is ~20 steps, so the first few "
                           f"dozen steps after the resume are effectively a cold start)")

    print(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
    return step
