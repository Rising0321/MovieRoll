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

"""Small shared helpers, and the checkpointed training-state records.

`ScalarStates` is written into `scalar_states.pkl` next to every checkpoint and read back on resume, so its
FIELD NAMES are part of the on-disk format -- renaming one silently loses that counter when an existing
checkpoint is loaded. `CycleStates` is the per-logging-window accumulator and is not checkpointed.
"""

import collections.abc
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from itertools import repeat

from loguru import logger


def _ntuple(n):
    """Create a function that converts input to n-tuple."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


# Convenience function for the one tuple size the modules need
to_2tuple = _ntuple(2)


@dataclass
class BaseStates:
    epoch: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset_epoch_based_states(self):
        pass

    def __repr__(self):
        max_key_length = max(len(k) for k in asdict(self))
        return "\n".join(
            [f"{k:>{max_key_length}}: {v}" for k, v in asdict(self).items()]
        )

    def to_dict(self):
        states = deepcopy(vars(self))
        # Convert defaultdict to dict
        for key, value in states.items():
            if isinstance(value, defaultdict):
                states[key] = dict(value)

        return states

    @classmethod
    def from_pretrained(cls, scalar_state, rank=None, world_size=None, default_rank0_ss=None):
        _scalar_state = {}
        if isinstance(scalar_state, dict):
            _scalar_state.update(scalar_state)
        elif isinstance(scalar_state, list):
            assert rank is not None and world_size is not None and default_rank0_ss is not None, (
                f"rank({rank}), world_size({world_size}), and default_rank0_ss({default_rank0_ss}) should be provided "
                f"when scalar_state is a list."
            )
            if len(scalar_state) != world_size:
                # Check if scalar_state all the same
                if all(scalar_state[0] == x for x in scalar_state):
                    default_rank0_ss = True

                if default_rank0_ss:
                    logger.info(f" len(scalar_state)={len(scalar_state)} != world_size={world_size}, Set default rank0 scalar state.")
                    _scalar_state.update(scalar_state[0])
                else:
                    _scalar_state.update(scalar_state[0])
                    for i in range(len(scalar_state)):
                        if isinstance(scalar_state[i], dict):
                            for k, v in scalar_state[i].items():
                                if isinstance(v, dict):
                                    _scalar_state[k].update(v)
                    logger.info(f"default_rank0_ss is False, Update scalar_state={_scalar_state}")
            else:
                _scalar_state.update(scalar_state[rank])
        else:
            raise ValueError(f"Unknown scalar state type: {type(scalar_state)}")

        # A checkpoint may carry fields this class does not have (older per-mask-type / per-data-type
        # counters); keep only the fields this class has.
        known = {f.name for f in fields(cls)}
        state = cls(**{key: value for key, value in _scalar_state.items() if key in known})
        # to_dict stored the counters as plain dicts; give them back their defaultdict behaviour, or the first
        # `+= n` on a key that was not yet written when the checkpoint was saved raises KeyError.
        for key, default in vars(cls()).items():
            value = getattr(state, key)
            if isinstance(default, defaultdict) and isinstance(value, dict):
                setattr(state, key, defaultdict(default.default_factory, value))
        return state


@dataclass
class ScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    """
    # - Global level
    epoch: dict = field(default_factory=lambda: defaultdict(int))
    train_steps: int = 0  # Accumulated training steps
    update_steps: int = 0  # Accumulated update steps
    current_run_update_steps: int = 0  # Update steps in current run (Reset for every resume)
    consumed_samples_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples
    epoch_consumed_samples: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples

    consumed_samples_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples per data-parallel group
    consumed_tokens_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed tokens
    consumed_computations_attn: float = 0         # Accumulated consumed computations of attention + mlp
    consumed_computations_total: float = 0        # Accumulated consumed computations of total

    lr: float = 0.0  # Current learning rate


@dataclass
class CycleStates(BaseStates):
    log_steps: int = 0
    running_loss: float = 0
    running_tokens: dict = field(default_factory=lambda: defaultdict(int))
    running_samples: dict = field(default_factory=lambda: defaultdict(int))
    # streaming per-chunk monitoring (not added to total loss)
    running_per_chunk_losses: dict = field(default_factory=lambda: defaultdict(float))
    # Paired with running_per_chunk_losses: how many micro-batches each key actually appeared in. Different
    # chunk_X keys appear at different rates, so the count is the only correct denominator for an average.
    running_per_chunk_loss_count: dict = field(default_factory=lambda: defaultdict(int))

    def reset_epoch_based_states(self):
        self.log_steps = 0
        self.running_loss = 0
        self.running_tokens = defaultdict(int)
        self.running_samples = defaultdict(int)
        self.running_per_chunk_losses = defaultdict(float)
        self.running_per_chunk_loss_count = defaultdict(int)

    def add_streaming_loss(self, per_chunk_losses_dict):
        for key, chunk_loss in per_chunk_losses_dict.items():
            self.running_per_chunk_losses[key] += chunk_loss
            self.running_per_chunk_loss_count[key] += 1
