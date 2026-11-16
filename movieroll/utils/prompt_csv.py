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

"""The sampling prompt CSV that `movieroll.generate` and SFT's training-time sampling both read.

One row is one sample: `index`, an integer `seed`, and either a `prompt` column or event columns `event_1`,
`event_2`, ... (taken in numeric order, each framed by `global_content_prompt` / `global_quality_prompt` when
both exist; the video is split evenly across the events). An optional `input_image` column names an image
relative to the CSV's directory; an empty cell means text-to-video.

The SFT config validator reads the same format with code of its own (`_check_sample_prompt_csv`), because it
must report a malformed row by file and line number before any weight loads.
"""

import csv
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptCsv:
    """The rows of one prompt CSV as parallel lists, plus what its header declared."""

    prompts: list
    seeds: list
    indexs: list
    csv_names: list
    event_prompts: list
    input_images: list
    event_cols: list
    has_input_image: bool


def read_prompt_csv(pmt_path):
    """Parse one prompt CSV, rows in file order."""
    parent_dir = os.path.basename(os.path.dirname(pmt_path))
    file_name = os.path.splitext(os.path.basename(pmt_path))[0]
    with open(pmt_path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fnames = reader.fieldnames or []
        _has_global = 'global_content_prompt' in fnames and 'global_quality_prompt' in fnames
        _event_cols = sorted(
            [f for f in fnames if f.startswith('event_') and f[6:].isdigit()],
            key=lambda x: int(x[6:])
        )
        _has_events = len(_event_cols) >= 1
        # Optional image-to-video column; an empty cell means the row has no input image
        _has_input_image = 'input_image' in fnames
        _prompts = []
        _seeds = []
        _indexs = []
        _csv_name = []
        _event_prompts = []
        _input_images = []
        for line in reader:
            _indexs.append(line['index'])
            _seeds.append(int(line['seed']))
            _image = (line['input_image'] or '').strip() if _has_input_image else ''
            _input_images.append(os.path.join(os.path.dirname(pmt_path), _image) if _image else None)
            _csv_name.append((parent_dir, file_name))
            if _has_events:
                g_content = line.get('global_content_prompt', '').strip() if _has_global else ''
                g_quality = line.get('global_quality_prompt', '').strip() if _has_global else ''
                composed = []
                for ec in _event_cols:
                    e_raw = line[ec].strip()
                    composed.append(' '.join(filter(None, [g_content, e_raw, g_quality])))
                _prompts.append(composed[0])
                _event_prompts.append(composed)
            else:
                _cur_prompt = line['prompt']
                _prompts.append(_cur_prompt)
                _event_prompts.append([_cur_prompt])
    return PromptCsv(
        prompts=_prompts,
        seeds=_seeds,
        indexs=_indexs,
        csv_names=_csv_name,
        event_prompts=_event_prompts,
        input_images=_input_images,
        event_cols=_event_cols,
        has_input_image=_has_input_image,
    )
