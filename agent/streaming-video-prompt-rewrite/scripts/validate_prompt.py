#!/usr/bin/env python3
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

"""Rule validator and pipeline helpers for streaming video prompt rewrites.

Python 3 standard library only. Messages and prompt templates come in English and Chinese,
selected with `lang="en"` / `lang="zh"` (the CLI's `--lang`); the checks themselves are the
same in both languages, and both language's wrappers and camera phrasings are recognised
whatever `lang` is set to.

CLI:
    python validate_prompt.py prompt.txt                # single / sequential output
    python validate_prompt.py response.txt --batch 3    # batch output with 3 events
    cat prompt.txt | python validate_prompt.py -
    python validate_prompt.py prompt.txt --json --no-length
    python validate_prompt.py prompt.txt --lang zh      # report the violations in Chinese

Exit code: 0 if every check passes, 1 otherwise.

Importable helpers:
    validate_rewritten_prompt, validate_batch_response, build_feedback_block,
    build_system_prompt, build_batch_user_message, build_streaming_user_message,
    rewrite_with_validation
"""

import argparse
import json
import re
import sys
from pathlib import Path

EVENT_DELIMITER = "@@@EVENT_BREAK@@@"
MAX_QUALITY_RETRIES = 3
LANGUAGES = ("en", "zh")
DEFAULT_LANGUAGE = "en"
# One rule set per language: `build_system_prompt` reads the file of the language it is asked for.
SYSTEM_PROMPT_FILES = {
    "en": Path(__file__).resolve().parent.parent / "system_prompt.md",
    "zh": Path(__file__).resolve().parent.parent / "system_prompt_CN.md",
}
DEFAULT_LIMITS = {"min_words": 150, "max_words": 300, "min_chars_zh": 150, "max_chars_zh": 350}

# The ending phrases banned by rule 2.A.3 of system_prompt.md / system_prompt_CN.md (SKILL.md and
# SKILL_CN.md repeat the list). "the end of" is banned there for the end of an interaction, a scene or a
# moment; a place ("the far end of the pier") is not ending language, so only a narrative or temporal noun
# after it counts.
TERMINAL_LANGUAGE_RE = re.compile(
    r"\b(?:finally|eventually|in the end|at last|in conclusion|conclu(?:de|des|ded|ding)|"
    r"wraps? up|comes? to (?:an? )?(?:end|close)|"
    r"settl(?:e|es|ed|ing) into (?:its |their |his |her |a |the )?final|"
    r"the end of (?:the |this |that |its |his |her |their )?(?:\w+ )?"
    r"(?:interaction|scene|moment|video|clip|story|event|sequence|day|performance|journey|"
    r"conversation|exchange|encounter|routine|process|ritual|ceremony|show|game|match)s?)\b|"
    r"(?:最后|最终|终于|至此结束|告一段落)",
    re.IGNORECASE,
)
SELF_REFERENCE_RE = re.compile(
    r"\b(?:this|the) (?:video|scene) (?:shows?|depicts?|presents?)\b|"
    r"(?:这个|本)(?:视频|场景)(?:展示|描绘|呈现)",
    re.IGNORECASE,
)
# A trajectory word only counts when it is the camera's own verb. Directly after a determiner it is an
# attribute of a noun ("the narrowing waterfall"), and before 的 it is a Chinese attribute (旋转的舞者):
# both describe the scene, not the camera.
FORBIDDEN_CAMERA_RE = re.compile(
    r"\b(?:the )?camera\s+(?:slowly\s+|steadily\s+|smoothly\s+)?"
    r"(?:pan(?:s|ning)?|zoom(?:s|ing)?|tilt(?:s|ing)?|doll(?:y|ies)|"
    r"orbit(?:s|ing)?|circl(?:es|ing)|roll(?:s|ing)?|rotat(?:es|ing)|"
    r"crane(?:s)?|ascend(?:s|ing)?|descend(?:s|ing)?|ris(?:es|ing)|"
    r"push(?:es|ing)? (?:in|forward)|pull(?:s|ing)? back|"
    r"moves? (?:left|right|upward|downward|forward|backward|closer|away))\b|"
    r"\b(?:pan|zoom|tilt|dolly|orbit|circle|roll|rotation|crane)[- ]shot\b|"
    r"\b(?:camera|shot|composition)\b[^.!?。！？]{0,40}"
    r"(?<!\bthe )(?<!\ba )(?<!\ban )(?<!\bits )(?<!\bhis )(?<!\bher )(?<!\btheir )"
    r"\b(?:refram(?:e|es|ed|ing)|widen(?:s|ed|ing)?|narrow(?:s|ed|ing)?)\b|"
    r"\bframe (?:widens?|narrows?|is reframed)\b|"
    r"(?:镜头|摄像机|相机)[^，。！？]{0,8}"
    r"(?:平移|摇摄|推进|推近|拉远|变焦|缩放|升起|上升|下降|环绕|旋转|倾斜|俯仰)(?!的)",
    re.IGNORECASE,
)
# The subject named inside a canonical camera sentence belongs to the scene, and its own motion is allowed
# ("keeping the rising balloon centered", 镜头跟随旋转木马移动), so it is blanked out before the trajectory scan.
CAMERA_SUBJECT_RES = (
    (re.compile(r"(\bkeeping\s+)[^.!?。！？]{1,80}?(?=\s+cent(?:er|r)ed\b)", re.IGNORECASE), "subject"),
    (re.compile(r"(\bcamera\s+(?:tracks?|follows?)\s+)[^,.!?。！？]{1,80}?(?=,)", re.IGNORECASE), "subject"),
    (re.compile(r"((?:镜头|摄像机|相机)(?:持续)?(?:跟随|追踪))[^，。！？]{1,30}?(?=移动)"), "主体"),
    (re.compile(r"(将|把)[^，。！？]{1,30}?(?=保持在画面)"), "主体"),
)
ALLOWED_CAMERA_RE = re.compile(
    r"\b(?:the )?camera (?:remains? (?:static|still)|is (?:stationary|fixed)|"
    r"tracks?|follows?)\b|"
    r"\b(?:static|fixed|tracking|follow) (?:camera|shot)\b|"
    r"(?:镜头|摄像机|相机)(?:保持)?(?:静止|固定|跟随|追踪)|"
    r"(?:静止|固定|跟随|追踪)镜头",
    re.IGNORECASE,
)
TRACKING_CAMERA_RE = re.compile(
    r"\bcamera (?:tracks?|follows?)\b|\b(?:tracking|follow) (?:camera|shot)\b|"
    r"(?:镜头|摄像机|相机)(?:持续)?(?:跟随|追踪)|(?:跟随|追踪)镜头",
    re.IGNORECASE,
)
CENTERING_RE = re.compile(
    r"\bcent(?:er|re)(?:ed|ing)?\b|"
    r"(?:画面|构图)(?:的)?(?:中央|中心)|(?:保持|始终|持续)居中",
    re.IGNORECASE,
)
EXIT_FRAME_RE = re.compile(
    r"\b(?:moves?|walks?|runs?|exits?|leaves?) (?:out of|off) (?:the )?frame\b|"
    r"(?:移出|走出|跑出|离开)(?:了)?画面",
    re.IGNORECASE,
)
# Both languages' wrappers are rejected whatever the message language is: a Chinese rewrite may
# just as well arrive behind an English label.
WRAPPER_PREFIXES = (
    "```", "{", "[", "Prompt:", "Rewritten Prompt:", "改写结果：", "改写后的prompt：",
)

# Violation messages, keyed by a stable code so the two languages cannot drift apart.
MESSAGES = {
    "empty": {
        "en": "Output is empty.",
        "zh": "输出为空。",
    },
    "length_zh": {
        "en": "Length is {count} Chinese characters; it must be {low}-{high} (punctuation excluded).",
        "zh": "当前长度为 {count} 个汉字，必须在 {low}-{high} 个汉字之间（不计标点）。",
    },
    "length_words": {
        "en": "Length is {count} words; it must be {low}-{high} words.",
        "zh": "当前长度为 {count} 个英文词，必须在 {low}-{high} 个词之间。",
    },
    "line_breaks": {
        "en": "Output must be one continuous paragraph with no line breaks.",
        "zh": "输出必须是一整段连续文字，不能有换行。",
    },
    "wrapper": {
        "en": "Output only the rewritten prompt, without wrappers or labels.",
        "zh": "只输出改写后的 prompt，不要加包裹符号或标签。",
    },
    "terminal": {
        "en": "Remove terminal or concluding language; keep the action open-ended.",
        "zh": "删除表示结束/收尾的措辞，保持动作开放、可持续。",
    },
    "self_reference": {
        "en": "Remove self-reference such as 'this video shows' or 'this scene depicts'.",
        "zh": "删除“这个视频展示”“this scene depicts”之类的自我指涉。",
    },
    "forbidden_camera": {
        "en": "Camera movement must be static or tracking; remove all other trajectories.",
        "zh": "镜头只能是静止或跟随，删除其他所有运镜轨迹。",
    },
    "no_camera": {
        "en": "State exactly one static or tracking/follow camera behavior.",
        "zh": "必须且只能有一句静止或跟随/追踪镜头的描述。",
    },
    "mixed_camera": {
        "en": "State only one base camera behavior; do not mix static and tracking.",
        "zh": "只能描述一种基础镜头行为，不要混用静止和跟随。",
    },
    "tracking_not_centered": {
        "en": "A tracking camera must explicitly keep the main subject centered.",
        "zh": "跟随镜头必须明确写出让主体保持在画面中央。",
    },
    "batch_count": {
        "en": "Expected {expected} prompts separated by a line containing only '{delimiter}', found {found}.",
        "zh": "期望得到 {expected} 条 prompt（以仅包含 '{delimiter}' 的行分隔），实际得到 {found} 条。",
    },
    "length_label_zh": {
        "en": "{count} Chinese characters",
        "zh": "{count} 个汉字",
    },
    "length_label_words": {
        "en": "{count} words",
        "zh": "{count} 个英文词",
    },
    "feedback_header": {
        "en": "\n\nVALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:\n",
        "zh": "\n\n上一次生成的校验反馈：\n",
    },
    "feedback_footer": {
        "en": "\nRegenerate the complete output and correct every issue above.",
        "zh": "\n请重新生成完整输出，并修正以上所有问题。",
    },
    "section_missing": {
        "en": "Section '{name}' not found in {path}",
        "zh": "在 {path} 中找不到段落 '{name}'",
    },
    "batch_event": {
        "en": "Event {index}: {event}",
        "zh": "事件 {index}：{event}",
    },
    "history_none": {
        "en": "PREVIOUS EVENTS (already finalized): None — this is the first event in the sequence.\n",
        "zh": "已定稿的历史事件：无——这是序列中的第一个事件。\n",
    },
    "history_line": {
        "en": "Event {index} (already rewritten, finalized): {prompt}",
        "zh": "事件 {index}（已改写并定稿）：{prompt}",
    },
    "history_header": {
        "en": "PREVIOUS EVENTS (already finalized, read-only context — do NOT rewrite or repeat these):\n",
        "zh": "已定稿的历史事件（只读上下文——不要改写或重复）：\n",
    },
    "new_event_header": {
        "en": "\nNEW EVENT TO REWRITE:\n",
        "zh": "\n待改写的新事件：\n",
    },
    "new_event_footer": {
        "en": "\n\nRewrite ONLY the new event above into one complete prompt, "
              "following the Multi-Event Streaming Mode rules.",
        "zh": "\n\n请仅将上面的新事件改写为一条完整的 prompt，遵循多事件流式模式的规则。",
    },
}


def _message(code, lang=DEFAULT_LANGUAGE, **fields):
    """One message in `lang`, falling back to English for an unknown language."""
    variants = MESSAGES[code]
    return variants.get(lang, variants[DEFAULT_LANGUAGE]).format(**fields)


def measure_length(text):
    """Return (unit, count): Chinese characters when Chinese dominates, otherwise words."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", text))
    if chinese_chars >= 20 and chinese_chars >= latin_words * 2:
        return "zh_chars", chinese_chars
    return "words", len(re.findall(r"\b\w+(?:[-']\w+)*\b", text))


def _mask_camera_subjects(text):
    """`text` with the subject of every canonical camera sentence replaced by a neutral word."""
    for pattern, placeholder in CAMERA_SUBJECT_RES:
        text = pattern.sub(lambda match: match.group(1) + placeholder, text)
    return text


def validate_rewritten_prompt(text, check_length=True, lang=DEFAULT_LANGUAGE, **limits):
    """Return rule violations for one rewritten prompt; an empty list means it passed."""
    limits = {**DEFAULT_LIMITS, **limits}
    stripped = text.strip() if text else ""
    if not stripped:
        return [_message("empty", lang)]

    errors = []
    if check_length:
        unit, count = measure_length(stripped)
        if unit == "zh_chars":
            low, high = limits["min_chars_zh"], limits["max_chars_zh"]
            if not low <= count <= high:
                errors.append(_message("length_zh", lang, count=count, low=low, high=high))
        else:
            low, high = limits["min_words"], limits["max_words"]
            if not low <= count <= high:
                errors.append(_message("length_words", lang, count=count, low=low, high=high))
    if "\n" in stripped or "\r" in stripped:
        errors.append(_message("line_breaks", lang))
    if stripped.startswith(WRAPPER_PREFIXES):
        errors.append(_message("wrapper", lang))
    if TERMINAL_LANGUAGE_RE.search(stripped):
        errors.append(_message("terminal", lang))
    if SELF_REFERENCE_RE.search(stripped):
        errors.append(_message("self_reference", lang))
    if FORBIDDEN_CAMERA_RE.search(_mask_camera_subjects(stripped)):
        errors.append(_message("forbidden_camera", lang))
    allowed_camera_count = len(ALLOWED_CAMERA_RE.findall(stripped))
    if allowed_camera_count == 0:
        errors.append(_message("no_camera", lang))
    elif allowed_camera_count > 1:
        errors.append(_message("mixed_camera", lang))
    if (
        TRACKING_CAMERA_RE.search(stripped)
        and CENTERING_RE.search(stripped) is None
        and EXIT_FRAME_RE.search(stripped) is None
    ):
        errors.append(_message("tracking_not_centered", lang))
    return errors


def split_batch_response(response_text):
    parts = [part.strip() for part in (response_text or "").split(EVENT_DELIMITER)]
    return [part for part in parts if part]


def validate_batch_response(
    response_text, expected_count, check_length=True, lang=DEFAULT_LANGUAGE, **limits
):
    """Split a batch response on EVENT_DELIMITER and validate every part.

    Returns (parts, errors); errors maps "structure" or "event <i>" to a list of violations
    and is empty when the whole batch passed.
    """
    parts = split_batch_response(response_text)
    errors = {}
    if len(parts) != expected_count:
        errors["structure"] = [
            _message(
                "batch_count", lang,
                expected=expected_count, delimiter=EVENT_DELIMITER, found=len(parts),
            )
        ]
    for index, part in enumerate(parts, 1):
        part_errors = validate_rewritten_prompt(part, check_length, lang=lang, **limits)
        if part_errors:
            errors[f"event {index}"] = part_errors
    return parts, errors


def build_feedback_block(errors, lang=DEFAULT_LANGUAGE):
    """Format validation errors as a block to append to the next generation request."""
    if isinstance(errors, dict):
        errors = [f"{key}: {message}" for key, messages in errors.items() for message in messages]
    return (
        _message("feedback_header", lang)
        + "\n".join(f"- {error}" for error in errors)
        + _message("feedback_footer", lang)
    )


def build_system_prompt(mode="single", lang=DEFAULT_LANGUAGE, path=None):
    """Assemble the BASE section plus one MODE section (single | batch | sequential).

    The section markers are the same in both rule files, so only the file changes with `lang`;
    an explicit `path` overrides the choice.
    """
    path = Path(path) if path is not None else SYSTEM_PROMPT_FILES.get(lang, SYSTEM_PROMPT_FILES[DEFAULT_LANGUAGE])
    text = path.read_text(encoding="utf-8")

    def section(name):
        match = re.search(rf"<!-- BEGIN {name} -->\n(.*?)<!-- END {name} -->", text, re.S)
        if not match:
            raise ValueError(_message("section_missing", lang, name=name, path=path))
        return match.group(1).strip()

    return section("BASE") + "\n\n" + section(f"MODE: {mode.upper()}")


def build_batch_user_message(events, lang=DEFAULT_LANGUAGE):
    return "\n".join(
        _message("batch_event", lang, index=index, event=event)
        for index, event in enumerate(events, 1)
    )


def build_streaming_user_message(previous_rewritten, new_event, lang=DEFAULT_LANGUAGE):
    """User message for sequential mode: finalized earlier rewrites + the new raw event."""
    if not previous_rewritten:
        history_block = _message("history_none", lang)
    else:
        lines = [
            _message("history_line", lang, index=index, prompt=prompt)
            for index, prompt in enumerate(previous_rewritten, 1)
        ]
        history_block = _message("history_header", lang) + "\n".join(lines) + "\n"
    return (
        history_block
        + _message("new_event_header", lang)
        + new_event
        + _message("new_event_footer", lang)
    )


def rewrite_with_validation(
    call_llm,
    system_prompt,
    user_message,
    expected_count=None,
    max_quality_retries=MAX_QUALITY_RETRIES,
    lang=DEFAULT_LANGUAGE,
    **limits,
):
    """Generate -> validate -> retry with feedback, for any model or API.

    call_llm(system_prompt, user_message) -> str is supplied by the caller; if the API has
    no system role, concatenate the two strings inside call_llm.
    Returns the validated prompt (str), the validated list of prompts when expected_count
    is given (batch mode), or None when every attempt failed.
    """
    feedback = ""
    for _ in range(max_quality_retries):
        result = call_llm(system_prompt, user_message + feedback)
        if not result:
            continue
        if expected_count is None:
            errors = validate_rewritten_prompt(result, lang=lang, **limits)
            if not errors:
                return result.strip()
        else:
            parts, errors = validate_batch_response(
                result, expected_count, lang=lang, **limits
            )
            if not errors:
                return parts
        feedback = build_feedback_block(errors, lang=lang)
    return None


def _length_label(text, lang=DEFAULT_LANGUAGE):
    unit, count = measure_length(text.strip())
    code = "length_label_zh" if unit == "zh_chars" else "length_label_words"
    return _message(code, lang, count=count)


def main():
    parser = argparse.ArgumentParser(description="Validate streaming video prompt rewrites.")
    parser.add_argument("input", help="File containing the model output, or '-' for stdin.")
    parser.add_argument("--batch", type=int, metavar="N", help="Validate a batch response with N events.")
    parser.add_argument("--no-length", action="store_true", help="Skip the length check.")
    parser.add_argument("--lang", choices=LANGUAGES, default=DEFAULT_LANGUAGE,
                        help="Language of the reported violations (the checks are identical).")
    parser.add_argument("--min-words", type=int, default=DEFAULT_LIMITS["min_words"])
    parser.add_argument("--max-words", type=int, default=DEFAULT_LIMITS["max_words"])
    parser.add_argument("--min-chars-zh", type=int, default=DEFAULT_LIMITS["min_chars_zh"])
    parser.add_argument("--max-chars-zh", type=int, default=DEFAULT_LIMITS["max_chars_zh"])
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    args = parser.parse_args()

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    limits = {
        "min_words": args.min_words,
        "max_words": args.max_words,
        "min_chars_zh": args.min_chars_zh,
        "max_chars_zh": args.max_chars_zh,
    }
    check_length = not args.no_length
    lang = args.lang

    if args.batch:
        parts, errors = validate_batch_response(
            text, args.batch, check_length, lang=lang, **limits
        )
        report = {
            "passed": not errors,
            "structure_errors": errors.get("structure", []),
            "events": [
                {
                    "index": index,
                    "length": _length_label(part, lang),
                    "errors": errors.get(f"event {index}", []),
                }
                for index, part in enumerate(parts, 1)
            ],
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("PASS" if report["passed"] else "FAIL")
            for error in report["structure_errors"]:
                print(f"structure: {error}")
            for event in report["events"]:
                status = "FAIL" if event["errors"] else "PASS"
                print(f"event {event['index']}: {status} ({event['length']})")
                for error in event["errors"]:
                    print(f"  - {error}")
    else:
        errors = validate_rewritten_prompt(text, check_length, lang=lang, **limits)
        report = {"passed": not errors, "length": _length_label(text, lang), "errors": errors}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"{'PASS' if report['passed'] else 'FAIL'} ({report['length']})")
            for error in errors:
                print(f"- {error}")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
