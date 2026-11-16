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

"""FSDP wrapping helpers.

`apply_fsdp2` is per-block `fully_shard`, plus a root `fully_shard` that is never resharded after forward.
`apply_fsdp_checkpointing` adds selective activation checkpointing and must run BEFORE `apply_fsdp2`.
`fsdp_device_mesh` picks the mesh `apply_fsdp2` shards over from the parallel topology, and
`broadcast_from_first_replica` aligns the dp_replicate copies of a hybrid-sharded model.
"""

from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

non_reentrant_wrapper = partial(
    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """Apply selective activation checkpointing to `model` in place.

    `p` is the fraction of blocks to checkpoint: 1 checkpoints every block, 1/2 every other one.
    """
    print("--> applying FSDP activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def _select_fsdp_device_mesh(world_mesh, parallel_dims, dp_mesh_dim_names):
    """Pick the device mesh FSDP should use, given dp_mesh_dim_names (torch<=2.6 and >=2.7 both work).

    torch<=2.6: the "dp_shard_sp" dimension that ParallelDims.build_mesh _flatten()s lands in the root mesh's
    mesh_dim_names, so slicing by name is enough. torch>=2.7: that name is only registered in _MeshResources,
    so the mesh is rebuilt explicitly -- the shard group is the contiguous dp_shard x sp ranks, with a
    replicate dimension on top when needed, laid out exactly as slicing by name is on <=2.6. Falling back to
    the whole world_mesh is not an option: fully_shard rejects the 3D mesh of multi-node hybrid
    (shard_world < world).
    """
    _mesh_names = world_mesh.mesh_dim_names or ()
    if all(_d in _mesh_names for _d in dp_mesh_dim_names):
        return world_mesh[tuple(dp_mesh_dim_names)]
    if not _mesh_names:
        # degenerate topology (world=1, every dimension <= 1): an unnamed mesh cannot be sliced, so pass None
        # and use the default process group (a single process does no cross-device communication, so NO_SHARD
        # semantics are unaffected)
        return None
    if "dp_shard_sp" in dp_mesh_dim_names:
        # torch>=2.7: the flattened name is missing from mesh_dim_names, but slicing by name still
        # resolves, so try that first.
        try:
            return world_mesh[tuple(dp_mesh_dim_names)]
        except Exception:
            # slicing unavailable too: rebuild the mesh explicitly (the shard group is the contiguous
            # dp_shard x sp ranks, laid out exactly as slicing by name would)
            _shard_root_dims = tuple(d for d in ("dp_shard", "sp") if d in _mesh_names)
            shard_mesh = world_mesh[_shard_root_dims]._flatten(mesh_dim_name="dp_shard_sp")
            if "dp_replicate" in dp_mesh_dim_names:
                return init_device_mesh(
                    world_mesh.device_type,
                    (parallel_dims.dp_replicate, shard_mesh.mesh.numel()),
                    mesh_dim_names=("dp_replicate", "dp_shard_sp"),
                )
            return shard_mesh
    return world_mesh


def fsdp_device_mesh(parallel_dims, world_mesh, sharding_strategy):
    """The device mesh `apply_fsdp2` shards over, derived from the parallel topology.

    Call it once per entry point and reuse the mesh: on torch>=2.7 `_select_fsdp_device_mesh` may build it with
    `init_device_mesh`, i.e. new process groups.
    """
    if parallel_dims is None:
        raise ValueError("parallel_dims cannot be None. Please ensure parallel_dims and world_mesh are properly initialized.")

    # The requested strategy only has to agree with the mesh: the topology itself is derived from parallel_dims
    # right after, on every path.
    if sharding_strategy == "full":
        assert not parallel_dims.dp_replicate_enabled
    elif sharding_strategy == "hybrid":
        assert parallel_dims.dp_replicate_enabled

    if parallel_dims.dp_shard_enabled or parallel_dims.sp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")     # hybrid: zero3 within each node
        else:
            dp_mesh_dim_names = ("dp_shard_sp",)                    # full shard
    else:
        # hybrid with one-rank shard groups and no SP: a 1-D mesh over every rank, which `fully_shard` shards
        # across all of them (nothing is replicated, so `broadcast_from_first_replica` must not run)
        dp_mesh_dim_names = ("dp_replicate", )

    return _select_fsdp_device_mesh(world_mesh, parallel_dims, dp_mesh_dim_names)


def apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    prefetch_depth: int = 2,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    `reshard_after_forward_policy` is "default" (reshard every block but the last), "always" or "never".
    `prefetch_depth` is how many upcoming modules to prefetch during forward/backward: higher values overlap
    more communication with computation at the cost of extra memory.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    all_blocks = list(model.double_blocks)

    for layer_id, transformer_block in enumerate(all_blocks):
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            reshard_after_forward = layer_id < len(all_blocks) - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # the root module is never resharded
    fully_shard(model, **fsdp_config, reshard_after_forward=False)

    if prefetch_depth > 1:
        for i, block in enumerate(all_blocks):
            fwd_prefetch = [all_blocks[j] for j in range(i + 1, min(i + 1 + prefetch_depth, len(all_blocks)))]
            if fwd_prefetch:
                block.set_modules_to_forward_prefetch(fwd_prefetch)

            bwd_prefetch = [all_blocks[j] for j in range(i - 1, max(i - 1 - prefetch_depth, -1), -1)]
            if bwd_prefetch:
                block.set_modules_to_backward_prefetch(bwd_prefetch)


def broadcast_from_first_replica(models, replicate_group):
    """Copy the first dp_replicate replica's parameter shards to the others, model by model, in the order given.

    Under hybrid sharding every replica loads the checkpoint itself; this makes them start bit-identical. The
    order of `models` and of their parameters pairs up the broadcasts across ranks, so it must be the same on
    every rank.
    """
    src = dist.get_process_group_ranks(replicate_group)[0]
    for model in models:
        for param in model.parameters():
            dist.broadcast(
                param._local_tensor if isinstance(param, DTensor) else param,
                src=src,
                group=replicate_group,
            )
