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

import math
import torch
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    # handle old pytorch versions
    DTensor = None

# This code is modified from the GitHub repository of KellerJordan:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps = 5):
    """Newton-Schulz quintic iteration for the zeroth power (orthogonalization) of G.

    The coefficients maximize the slope at zero, so the result is not UV^T but something like US'V^T with
    S'_{ii} ~ Uniform(0.5, 1.5); that costs no model performance and needs fewer steps.
    """
    if isinstance(G, DTensor):
        device_mesh = G.device_mesh
        G = G.full_tensor()
    else:
        device_mesh = None
        
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    
    if device_mesh != None:
        return DTensor.from_local(X, device_mesh) 
    else:
        return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    `muon_params` (each >= 2-D) get the orthogonalized update, scaled by `adjust_lr_for_muon`; `adamw_params` get
    the internal AdamW instead. `momentum` / `nesterov` / `ns_steps` configure the Muon branch, `adamw_betas` /
    `adamw_eps` the AdamW branch, and `wd` is the weight decay of both.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # Moonlight ("Muon is Scalable for LLM Training") scales the learning rate by the size of the parameter
        # matrix, so that the update RMS matches AdamW's
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step; `closure` re-evaluates the model and returns the loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    # e.g. vision_in when no window of the step was scored with the image; a zero gradient keeps
                    # `full_tensor` valid
                    if isinstance(p, DTensor):
                        g = DTensor.from_local(torch.zeros_like(p.to_local()), p.device_mesh, placements=p.placements)
                    else:
                        g = torch.zeros_like(p)
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = g.bfloat16()
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
