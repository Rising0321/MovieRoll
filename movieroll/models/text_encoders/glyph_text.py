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

"""The quoted text of a prompt, which the byT5 glyph encoder reads.

SFT (`train.py`), Stage-2 and inference (both through the pipeline's byT5 encoding) extract it with this one
function, so they accept the same quote characters: a prompt in curly quotes gets glyph conditioning in all
three, or in none.
"""

import re

# Quoted text picked up as glyph input: ASCII straight quotes (") and curly quotes (U+201C / U+201D).
GLYPH_QUOTE_PATTERN = r'\"(.*?)\"|“(.*?)”'


def extract_glyph_texts(prompt):
    """The quoted strings of `prompt` in order of appearance; repeats are dropped when there is more than one."""
    matches = re.findall(GLYPH_QUOTE_PATTERN, prompt)
    glyph_texts = [m[0] or m[1] for m in matches]
    return list(dict.fromkeys(glyph_texts)) if len(glyph_texts) > 1 else glyph_texts
