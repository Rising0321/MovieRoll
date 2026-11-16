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

"""FSDP2-aware exponential moving average, shared by SFT and Stage-3 distillation.

Under FSDP2 a parameter is not always the same kind of object. A module whose `reshard_after_forward` is False
stays UNSHARDED after a forward, and `named_parameters()` then hands out a plain `Parameter` holding the full
tensor instead of a `DTensor` holding this rank's shard. The source of truth, however, is always the sharded
storage: whatever is written into an unsharded view is discarded at the next reshard.

A `restore()` that wrote a saved shard back through whatever view happened to be current would therefore leak:
after a sampling forward, the modules left unsharded take the full-tensor path, the training weights written back
are thrown away and the EMA weights survive (measured on two GPUs: 6 of 12 parameters stayed at their EMA value
after a store -> copy_to -> sample -> restore cycle). So every operation that writes reshards first, and the
write itself is chosen by the CURRENT form of the destination parameter, never by an assumption that both sides
match.

One class serves both chains, which need different things from `state_dict` -- full tensors for the SFT
safetensors export, this rank's shards for the distillation checkpoint -- so both exist under separate names:
`state_dict` / `load_state_dict` are the per-rank sharded pair, and `gather_full_state_dict` is the collective
that produces the full tensors a released checkpoint holds.
"""

from typing import Dict, List

import torch
import torch.distributed
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor


def clean_checkpoint_wrapped_module_prefix(key: str) -> str:
    """Strip the `_checkpoint_wrapped_module` segments activation checkpointing inserts into parameter names."""
    return key.replace('._checkpoint_wrapped_module.', '.').replace('_checkpoint_wrapped_module.', '')


class FSDPEMAModel:
    """Shadow copy of an FSDP2 model's parameters, updated as an exponential moving average.

    `model` must be an `FSDPModule`. Every method that writes to the model or reads its parameters reshards
    first, so the shadow and the model always agree about what a parameter is. The shadow (and the backup) are
    keyed by the parameter name with the activation-checkpoint prefix stripped, the name the checkpoint files use;
    a parameter of the model that has no shadow is a KeyError, not a skip.
    """

    def __init__(
        self,
        model: FSDPModule,
        decay: float,
    ):
        # Reshard BEFORE cloning. The EMA can be created mid-training (the distillation chain starts it at
        # ema_start_step), when the modules with reshard_after_forward=False are still unsharded; cloning then
        # would build plain-tensor shadows for parameters that are DTensors during training, and the next
        # update() would reach for `._local_tensor` on a plain tensor and die.
        self._reshard_all(model)
        self.decay = decay
        self.one_minus_decay = 1.0 - decay
        self.shadow_params = {}
        for name, param in model.named_parameters():
            key = clean_checkpoint_wrapped_module_prefix(name)
            if key in self.shadow_params:
                raise RuntimeError(
                    f"EMA: two parameters are both named {key!r} once the activation-checkpoint prefix is stripped"
                )
            self.shadow_params[key] = param.clone().detach().float()
        self.backup = {}

        # The only check on the invariant the update math depends on, and it raises: past a mismatch comes either
        # a crash or a silently wrong update.
        bad = []
        for name, param in model.named_parameters():
            shadow = self.shadow_params[clean_checkpoint_wrapped_module_prefix(name)]
            if isinstance(param, DTensor) != isinstance(shadow, DTensor):
                bad.append(name)
        if bad:
            raise RuntimeError(
                f"EMA shadow and model parameters disagree about sharding ({len(bad)} parameters, "
                f"first few: {bad[:5]}). update() would crash or compute the wrong value; "
                "reshard the model before constructing the EMA."
            )

    @staticmethod
    def _reshard_all(model) -> None:
        """Force every FSDP module back to its sharded form.

        This is a correctness precondition, not an optimisation: an unsharded parameter is a VIEW, and writes to
        it are dropped at the next reshard.
        """
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    @staticmethod
    def _write_into(dst_param, src, name: str) -> None:
        """Write `src` into `dst_param`, converting between sharded and full forms as needed.

        The form is decided by the DESTINATION, because that is what the write has to land in. Converting a full
        tensor down to this rank's shard is local; converting a shard up to a full tensor is a collective, which
        is safe here because every rank runs the same branch for the same parameter.
        """
        if isinstance(dst_param, DTensor):
            dst = dst_param._local_tensor
            if isinstance(src, DTensor):
                source = src._local_tensor
            else:
                source = distribute_tensor(
                    src.to(device=dst.device), dst_param.device_mesh, dst_param.placements
                )._local_tensor
            if source.shape != dst.shape:
                raise RuntimeError(
                    f"EMA: {name} local shard shape mismatch ({tuple(source.shape)} -> {tuple(dst.shape)})"
                )
            dst.copy_(source.to(device=dst.device, dtype=dst.dtype))
        else:
            source = src.full_tensor() if isinstance(src, DTensor) else src
            if source.shape != dst_param.shape:
                raise RuntimeError(
                    f"EMA: {name} shape mismatch ({tuple(source.shape)} -> {tuple(dst_param.shape)})"
                )
            dst_param.copy_(source.to(device=dst_param.device, dtype=dst_param.dtype))

    @torch.no_grad()
    def update(self, model: FSDPModule) -> None:
        """Advance the shadow one EMA step. Parameters that do not require grad are copied, not averaged."""
        # A no-op at the normal call site (right after optimizer.step(), where everything is sharded), but it
        # decouples the math from whatever sharding state the call site happens to be in.
        self._reshard_all(model)
        for name, param in model.named_parameters():
            shadow_param = self.shadow_params[clean_checkpoint_wrapped_module_prefix(name)]

            if isinstance(param, DTensor):
                local_param = param._local_tensor
                shadow_local = shadow_param._local_tensor
                if param.requires_grad:
                    shadow_local.sub_(self.one_minus_decay * (shadow_local.float() - local_param.float()))
                else:
                    shadow_local.copy_(local_param)
            else:
                if param.requires_grad:
                    shadow_param.sub_(self.one_minus_decay * (shadow_param.float() - param.float()))
                else:
                    shadow_param.copy_(param)

    @torch.no_grad()
    def copy_to(self, model: FSDPModule) -> None:
        """Write the EMA weights into the model (for sampling). Pair every call with store() / restore()."""
        self._reshard_all(model)
        for name, param in model.named_parameters():
            self._write_into(param, self.shadow_params[clean_checkpoint_wrapped_module_prefix(name)], name)

    @torch.no_grad()
    def store(self, model: FSDPModule) -> None:
        """Keep the live training weights aside so restore() can put them back after sampling."""
        self._reshard_all(model)
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[clean_checkpoint_wrapped_module_prefix(name)] = param.detach().clone()

    @torch.no_grad()
    def restore(self, model: FSDPModule) -> None:
        """Put the stored training weights back. Raises on an empty backup rather than leaving EMA weights in."""
        if not self.backup:
            raise RuntimeError(
                "EMA.restore() was called with an empty backup, which means store() never ran or has already "
                "been consumed. Returning quietly would leave the model on its EMA weights for good."
            )
        self._reshard_all(model)
        for name, param in model.named_parameters():
            self._write_into(param, self.backup[clean_checkpoint_wrapped_module_prefix(name)], name)
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """This rank's shard of every shadow parameter, to be reloaded by load_state_dict.

        This is the per-rank checkpoint pair. For the full tensors a released checkpoint holds, use
        `gather_full_state_dict`.
        """
        out = {}
        for key, value in self.shadow_params.items():
            local = value._local_tensor if isinstance(value, DTensor) else value
            out[key] = local.detach().cpu().clone()
        return out

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the shadow from per-rank shards written by `state_dict`.

        A state dict whose keys do not line up (different world size, different architecture, a changed
        prefix) is rejected. Accepting it would leave the shadow at the values `__init__` just cloned off the
        live weights -- the entire EMA history silently reset to "now" -- and the exported EMA checkpoint
        quietly worthless. Keys are matched as they are: `state_dict` writes prefix-stripped names, so a shard
        carrying raw `_checkpoint_wrapped_module` names (which this class never writes) is rejected too.
        """
        missing = []
        with torch.no_grad():
            for name, shadow in self.shadow_params.items():
                if name not in state_dict:
                    missing.append(name)
                    continue
                src = state_dict[name]
                if isinstance(shadow, DTensor):
                    shadow._local_tensor.copy_(src.to(shadow._local_tensor.device))
                else:
                    shadow.copy_(src.to(shadow.device))
        if missing:
            raise RuntimeError(
                f"EMA load_state_dict: {len(missing)}/{len(self.shadow_params)} shadow parameters are not "
                f"in the checkpoint, first few: {missing[:5]}.\n"
                "Continuing would leave the EMA at the values just cloned from the live weights, i.e. the "
                "whole EMA history discarded. Check that the checkpoint's world size, architecture and key "
                "prefixes match."
            )
        # The other direction: keys the shadow does not know mean this came from a different model.
        extra = len(state_dict) - (len(self.shadow_params) - len(missing))
        if extra > 0:
            raise RuntimeError(
                f"EMA load_state_dict: the checkpoint has {extra} extra keys "
                f"({len(state_dict)} entries vs {len(self.shadow_params)} shadow parameters), "
                "so it belongs to a different model."
            )

    def gather_full_state_dict(self) -> Dict[str, torch.Tensor]:
        """Full (unsharded, CPU) shadow tensors, for exporting one safetensors file.

        A collective: `full_tensor()` gathers across the mesh, so every rank has to call it.
        """
        state_dict = {}
        for key, value in self.shadow_params.items():
            tensor = value.full_tensor() if isinstance(value, DTensor) else value
            state_dict[key] = tensor.cpu().clone()
        return state_dict


__all__: List[str] = [
    "FSDPEMAModel",
    "clean_checkpoint_wrapped_module_prefix",
]
