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

"""The rolling-window machinery of MovieRoll, with no framework import.

Nothing in this package imports SGLang, diffusers or `movieroll`: a request's bookkeeping (chunk layout, event
boundaries, KV recompute policy), the per-frame scheduling of a window, the clean-history KV cache, the rotary
table and the sampling loop itself are plain torch. The SGLang-facing code (`sglang_movieroll.runtime`,
`sglang_movieroll.configs`) drives it. Each piece is a port of the `movieroll` module its docstring names.
"""
