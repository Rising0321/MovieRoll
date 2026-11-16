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

# ==============================================================================
# Modified from SiT (https://github.com/willisma/SiT)
# ==============================================================================

"""Linear interpolant between data and noise.

The linear ("interpolant coupling") plan: the forward process the training step needs --
`xt = alpha_t * x1 + sigma_t * x0` with `alpha_t = 1 - t`, `sigma_t = t` -- plus the velocity target.
"""


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim, num_frames], time vector
      x: [batch_dim, channels, num_frames, height, width], data point
    """
    assert x.ndim == 5, "Input x must be 5D"
    if len(t.size()) == 1:
        t = t.unsqueeze(-1)
    t = t.view(t.size(0), 1, t.size(1), 1, 1)
    return t


#### Coupling Plans ####


class ICPlan:
    """Linear Coupling Plan"""

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return 1 - t, -1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return t, 1

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1)
        return t, xt, ut
