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

"""Process-group state shared by the transformer, the pipelines and the trainers.

`ParallelDims` builds the device mesh (data-parallel replicate / shard and sequence-parallel dimensions); `nccl_info`
records this process's sequence-parallel group and, for parallel VAE decoding, its VAE group.
"""

import datetime
import os
from dataclasses import dataclass
from typing import List, Optional

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    sp: int
    world_size: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, sp = (
            self.dp_replicate,
            self.dp_shard,
            self.sp,
        )
        for d in (dp_replicate, sp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must be -1 or >= 1; -1 means derive it from world_size"
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * sp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * sp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"sp({sp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.dp_replicate, self.dp_shard, self.sp],
            ["dp_replicate", "dp_shard", "sp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submeshes here so that every required process group is initialized:
        # mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # mesh for parameter sharding
        dp_shard_sp_mesh_dim_names = []
        # mesh for the loss all-reduce
        dp_sp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_sp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_sp_mesh_dim_names.append("dp_shard")
            dp_sp_mesh_dim_names.append("dp_shard")
        if self.sp_enabled:
            dp_shard_sp_mesh_dim_names.append("sp")
            dp_sp_mesh_dim_names.append("sp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_sp_mesh_dim_names != []:
            mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_sp"
            )
        if dp_sp_mesh_dim_names != []:
            mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")

        self.device_mesh = mesh
        if self.sp_enabled:
            self.sp_mesh = mesh['sp']
        if self.dp_enabled:
            self.dp_mesh = mesh['dp']

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def sp_enabled(self):
        return self.sp > 1


class COMM_INFO:
    def __init__(self):
        self.sp_group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_spgroup = 0
        self.parallel_dims = None
        self.vae_group = None
        self.vae_group_ranks = None
        self.rank_within_vaegroup = 0

nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False


def initialize_sequence_parallel_state_v2(parallel_dims, device_mesh):
    if parallel_dims.sp_enabled:
        initialize_sequence_parallel_group_v2(parallel_dims, device_mesh)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_spgroup = 0
        nccl_info.parallel_dims = parallel_dims


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE

def initialize_sequence_parallel_group_v2(parallel_dims, device_mesh):
    rank = int(os.getenv("RANK", "0"))
    nccl_info.sp_size = parallel_dims.sp
    nccl_info.global_rank = rank
    nccl_info.parallel_dims = parallel_dims
    if nccl_info.sp_size > 1:
        set_sequence_parallel_state(True)
        nccl_info.sp_group = device_mesh.get_group(mesh_dim="sp")
        nccl_info.rank_within_spgroup = device_mesh.get_local_rank(mesh_dim="sp")


VAE_PARALLEL_MODES = ("none", "sp", "custom")


def parse_vae_parallel_ranks(spec: str, world_size: int) -> List[List[int]]:
    """Parse "0,1,2,3;4,5,6,7" into disjoint groups of global ranks."""
    if not spec or not spec.strip():
        raise ValueError("--vae-parallel-ranks is required for --vae-parallel-mode custom")
    groups = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        groups.append([int(r) for r in part.split(",") if r.strip()])
    flat = [r for g in groups for r in g]
    if not groups or any(not g for g in groups):
        raise ValueError(f"empty VAE parallel group in {spec!r}")
    if len(flat) != len(set(flat)):
        raise ValueError(f"VAE parallel groups must be disjoint, got {spec!r}")
    bad = [r for r in flat if not 0 <= r < world_size]
    if bad:
        raise ValueError(f"VAE parallel ranks {bad} are outside [0, {world_size})")
    return groups


def initialize_vae_parallel_group(mode: str = "none", ranks: Optional[str] = None):
    """Select the process group PS-VAE's parallel decode is sharded over, store it in ``nccl_info``, return it.

    ``none`` means no parallel decode; ``sp`` reuses the DiT sequence-parallel group; ``custom`` builds the
    groups from ``ranks``, independent of the DiT mesh. ``dist.new_group`` is collective over WORLD, so every
    process must call this with the same arguments, member or not. None when this process has no group.
    """
    if mode not in VAE_PARALLEL_MODES:
        raise ValueError(f"vae parallel mode must be one of {VAE_PARALLEL_MODES}, got {mode!r}")
    nccl_info.vae_group = None
    nccl_info.vae_group_ranks = None
    nccl_info.rank_within_vaegroup = 0
    if mode == "none":
        return None

    rank = dist.get_rank()
    if mode == "sp":
        if nccl_info.sp_size <= 1:
            return None
        group = nccl_info.sp_group
    else:
        group = None
        for group_ranks in parse_vae_parallel_ranks(ranks, dist.get_world_size()):
            candidate = dist.new_group(group_ranks)
            if rank in group_ranks:
                group = candidate
        if group is None:
            return None

    nccl_info.vae_group = group
    nccl_info.vae_group_ranks = dist.get_process_group_ranks(group)
    nccl_info.rank_within_vaegroup = dist.get_rank(group)
    return group


def destroy_sequence_parallel_group():
    """Destroy every process group, the default one included.

    `dist.destroy_process_group()` without a `group=` tears down the whole distributed environment, so this
    is a shutdown call, not a way to drop sequence parallelism and keep training.
    """
    dist.destroy_process_group()


def init_nccl_process_group(**extra_kwargs):
    """`dist.init_process_group(backend="nccl", timeout=3600 s, **extra_kwargs)`, later groups included.

    600 s is too short: loading 33 GB of safetensors plus FSDP init across several nodes exceeds it. Every
    sub-group (the SP group, the shard mesh, the dp_replicate group) is created with timeout=None and falls
    back to `_get_default_timeout`, which reads the name out of the distributed_c10d MODULE namespace, so
    patching c10d's own attributes is what makes the sub-groups inherit 3600 s rather than time out mid-run
    (it is a from-import binding, so setting the attribute on the constants module does not propagate).
    `init_process_group` is looked up on the module at call time and given keywords only.
    """
    timeout = datetime.timedelta(seconds=3600)
    import torch.distributed.distributed_c10d as _c10d
    _c10d.default_pg_nccl_timeout = timeout
    _c10d.default_pg_timeout = timeout
    dist.init_process_group(backend="nccl", timeout=timeout, **extra_kwargs)
