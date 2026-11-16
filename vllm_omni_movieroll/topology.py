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
#
# ==============================================================================
# Follows vLLM-Omni v0.32.0 (commit 5917730a), vllm_omni/model_executor/models/hunyuan_video/pipeline.py
# ==============================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MovieRoll's stage topologies for vLLM-Omni: one diffusion stage whose final output is a video, per chain.

Registered with `vllm_omni.config.pipeline_registry.register_pipeline` (the out-of-tree route) and matched to a model
directory through `model_index.json`'s ``_class_name`` (``diffusers_class_name``), like vLLM-Omni's HunyuanVideo-1.5 topology.
Without one a plugin model falls back to the generic diffusion stage, whose output type defaults to ``image``, and the
video API (`/v1/videos`) refuses the model ("No final video output stage"). The two chains run the same pipeline class
under two architecture names, because vLLM-Omni looks up a model's serving defaults by that name (`video_defaults`).
"""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

from vllm_omni_movieroll.profiles import PIPELINE_CLASS


def _video_pipeline(model_type: str, model_arch: str) -> PipelineConfig:
    return PipelineConfig(
        model_type=model_type,
        model_arch=model_arch,
        diffusers_class_name=model_arch,
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="dit",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(),
                final_output=True,
                final_output_type="video",
                model_arch=model_arch,
            ),
        ),
    )


MOVIEROLL_PIPELINES = (
    _video_pipeline("movieroll", PIPELINE_CLASS["base"]),
    _video_pipeline("movieroll_distilled", PIPELINE_CLASS["distilled"]),
)
