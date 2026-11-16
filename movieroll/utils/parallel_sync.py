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

"""Make every rank of a sequence-parallel group agree on an object, and on its random state.

Sequence parallelism splits one sample across the ranks of an SP group, so those ranks must agree on every
decision that shapes the sample: the batch itself, the drawn timesteps, the task form, the dropped windows.

* ``sync_object_for_parallel_training`` broadcasts a concrete object from the group's first rank;
* ``sync_random_states`` broadcasts the four global RNG states, so that draws made independently on each rank
  (the global ``random.random()`` in ``get_multitask_mask``, ``torch.randn_like`` inside the transport) come out
  the same without having to be broadcast one by one.

Quirks callers rely on: the returned tensors are on CUDA, tuples come back as lists, dispatch is on the exact
type so subclasses such as ``Parameter`` and ``OrderedDict`` take the pickle path, and a value mismatch under
``debug_with_check`` is only logged before being overwritten from the source rank.
"""

import random as _python_random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from movieroll.commons.parallel_states import nccl_info


def _sync_groups(parallel_dims):
    """Process groups to synchronize over, in order. Only sequence parallelism exists in MovieRoll.

    The list holds at most the SP group. With sp == 1 it is empty and every function below returns its
    argument untouched, which is the ODE production topology.
    """
    if parallel_dims is None:
        parallel_dims = nccl_info.parallel_dims
    assert parallel_dims is not None
    if parallel_dims.sp_enabled:
        return [parallel_dims.sp_mesh.get_group()], ["sp"]
    return [], []


def is_src(src):
    """Whether this process is the broadcast source. `src` is a GLOBAL rank, as dist.broadcast expects."""
    return dist.get_rank() == src


def gather_obj(obj, group=None):
    world_size = dist.get_world_size() if group is None else dist.get_world_size(group)
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj, group=group)
    return gathered


def broadcast_object(obj, src, group=None):
    """Pickle `obj` from rank `src` to the rest of `group`."""
    buffer = [obj] if is_src(src) else [None]
    dist.broadcast_object_list(buffer, src=src, group=group)
    return buffer[0]


def broadcast_tensor(tensor, src, group=None):
    """Broadcast a tensor, shape and dtype first. The result is on CUDA on every rank, including the source."""
    if is_src(src):
        tensor = tensor.cuda().contiguous()
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group=group)
    dtype = broadcast_object(dtype, src=src, group=group)
    buffer = tensor if is_src(src) else torch.empty(shape, device="cuda", dtype=dtype)
    dist.broadcast(buffer, src=src, group=group)
    return buffer


def auto_broadcast(obj, src, group=None):
    """Broadcast by structure: tensors as tensors, containers element by element, anything else as a pickle.

    Dispatch is on the EXACT type, so a `Parameter` or an `OrderedDict` goes down the pickle path rather than
    the tensor / dict path. A tuple comes back as a list and a dict comes back as a new dict.
    """
    obj_type = broadcast_object(type(obj), src=src, group=group)

    if obj_type is torch.Tensor:
        return broadcast_tensor(obj, src=src, group=group)
    if obj_type is list or obj_type is tuple:
        length = len(obj) if is_src(src) else None
        length = broadcast_object(length, src=src, group=group)
        if not is_src(src):
            obj = [None] * length
        return [auto_broadcast(item, src=src, group=group) for item in obj]
    if obj_type is dict:
        keys = list(obj.keys()) if is_src(src) else None
        keys = broadcast_object(keys, src=src, group=group)
        return {
            key: auto_broadcast(obj[key] if is_src(src) else None, src=src, group=group)
            for key in keys
        }
    return broadcast_object(obj, src=src, group=group)


def sync_object_for_parallel_training(
    object,
    force_object: bool = False,
    debug_with_check: bool = False,
    trace: str = "arg0",
    parallel_dims=None,
):
    """Return `object` as it is on the first rank of every synchronization group.

    `force_object` pickles the whole object instead of walking its structure. With no synchronization group
    (sp == 1) the object is returned unchanged -- not moved to CUDA, tuples still tuples, dicts still the same
    object.

    `debug_with_check` is for tensors only (train.py uses it on the loss): it asserts that types, shapes and
    dtypes agree across the group, logs a value mismatch, then broadcasts the source rank's value in place into
    `object.cuda().contiguous()` -- the caller's own tensor when that is already contiguous on the current CUDA
    device. A non-tensor raises TypeError even with no synchronization group, and combining it with
    `force_object` raises ValueError. This branch dispatches on `isinstance`, so a `Parameter` takes it.
    """
    sync_groups, group_names = _sync_groups(parallel_dims)

    if not debug_with_check:
        for group in sync_groups:
            src = dist.get_process_group_ranks(group)[0]
            if force_object:
                object = broadcast_object(object, src=src, group=group)
            else:
                object = auto_broadcast(object, src=src, group=group)
        return object

    obj_type = type(object)
    for index, group in enumerate(sync_groups):
        buffer = gather_obj(obj_type, group=group)
        assert all(x == buffer[0] for x in buffer), (
            f"checking group {group_names[index]}, {trace} has different types ({buffer})"
        )

    # Refused only after the type gather above: it is a collective, so a rank raising before it would leave the
    # others waiting in it.
    if force_object:
        raise ValueError(f"debug_with_check does not combine with force_object ({trace})")
    if not isinstance(object, torch.Tensor):
        raise TypeError(f"debug_with_check only supports tensors, {trace} is a {obj_type.__name__}")

    old_device = object.device
    object = object.cuda().contiguous()

    for index, group in enumerate(sync_groups):
        shapes = gather_obj(object.shape, group=group)
        if not all(x == shapes[0] for x in shapes):
            logger.error(
                f"checking group {index} ({group_names[index]} group), {trace} has different shapes ({shapes})"
            )

    for group_id, group in enumerate(sync_groups):
        shapes = gather_obj(object.shape, group=group)
        dtypes = gather_obj(object.dtype, group=group)
        assert all(x == shapes[0] for x in shapes), (
            f"checking group {group_id} ({group_names[group_id]} group), {trace} has different shapes ({shapes})"
        )
        assert all(y == dtypes[0] for y in dtypes), (
            f"checking group {group_id} ({group_names[group_id]} group), {trace} has different dtypes ({dtypes})"
        )
        buffer = [torch.empty_like(object) for _ in range(group.size())]
        dist.all_gather(buffer, object, group=group)
        if not all(torch.allclose(x, buffer[0]) for x in buffer):
            # Only reported, never raised: the broadcast below overwrites the value anyway.
            logger.error(
                f"checking group {group_id} ({group_names[group_id]} group), {trace} not equal, {buffer} "
            )

    for group in sync_groups:
        src = dist.get_process_group_ranks(group)[0]
        dist.broadcast(object, src=src, group=group)

    object = object.to(old_device)

    return object


def _collect_rng_states() -> Dict[str, Any]:
    """Collect the global torch / cuda / numpy / python random states.

    The CUDA entry is the CURRENT device's state repeated once per visible device; `get_rng_state_all`
    would instead touch every visible device on every rank.
    """
    states = {
        "torch": torch.get_rng_state(),
        "python": _python_random.getstate(),
    }
    states["numpy"] = np.random.get_state()
    states["torch.cuda"] = (
        [torch.cuda.get_rng_state() for _ in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return states


def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """Write the four global random states back into this process."""
    torch.set_rng_state(rng_state_dict["torch"])
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    np.random.set_state(rng_state_dict["numpy"])
    # The inner state is rebuilt as a tuple defensively: `random.setstate` requires one, and a transport that
    # round-trips it as a list (json, some pickle protocols for subclasses) would otherwise fail here.
    version, state, gauss = rng_state_dict["python"]
    _python_random.setstate((version, tuple(state), gauss))


def sync_random_states(parallel_dims=None) -> None:
    """Give every rank of the SP group the first rank's global random state.

    Called once per training step before the forward. It is what makes the draws that each rank performs
    independently agree: the global `random.random()` of `get_multitask_mask`, the `randn_like` of the transport,
    the noise sampling. Draws that go through an explicit SP source generator (the window-size draw) do not
    depend on it.
    """
    rng_states = _collect_rng_states()
    rng_states = sync_object_for_parallel_training(
        rng_states, force_object=True, parallel_dims=parallel_dims
    )
    _set_rng_states(rng_states)




__all__: List[str] = [
    "sync_object_for_parallel_training",
    "sync_random_states",
]
