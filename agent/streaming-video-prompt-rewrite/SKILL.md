---
name: streaming-video-prompt-rewrite
description: Rewrites short user ideas into detailed, structured, single-paragraph prompts for streaming (indefinitely extending, chunk-by-chunk) text-to-video generation. Enforces a static or subject-tracking camera that keeps the main subject centered, open-ended continuable motion, natural pace, strict input fidelity, and a fixed length band; supports single-event, batch multi-event, and sequential (interactive) multi-event rewriting with cross-event consistency, including image-to-video prompts that stay consistent with the input image, plus a rule validator and a feedback-retry loop. Use when expanding or rewriting video generation prompts, preparing prompts for streaming, long, or infinite video models or for a video that starts from an input image, writing multi-event or interactive video stories, or building an LLM prompt-rewrite pipeline for video benchmarks.
---

# Streaming Video Prompt Rewrite

Turn a short idea (or an ordered series of event ideas) into a detailed, objective, single-paragraph video prompt that a streaming video model can keep extending indefinitely without reaching a dead end.

## Files

- `system_prompt.md` — the full rule set. Read it before the first rewrite in a session. It doubles as a drop-in system prompt: the `BASE` section plus one `MODE` section.
- `examples.md` — worked examples for each mode, plus quick conversions. `examples_CN.md` is localized, not a line-by-line translation: its examples and section numbers differ, and it adds a stationary-subject single-mode example.
- `SKILL_CN.md` / `system_prompt_CN.md` / `examples_CN.md` — the same rule set in Chinese; read those instead when the output language is Chinese.
- `scripts/validate_prompt.py` — rule validator and pipeline helpers, shared by both languages (Python 3 standard library only). `--lang` / `lang=` picks the language of the reported violations; the checks themselves are language-independent.

## Choose a mode

| Mode | Input | Output |
|---|---|---|
| Single | one sentence | one prompt |
| Batch multi-event | N ordered events, all known upfront | N prompts separated by a line containing only `@@@EVENT_BREAK@@@` |
| Sequential (streaming) | finalized rewrites of earlier events + one new raw event | one prompt for the new event only |

Use Sequential when events arrive interactively (the next event is written after the previous one has been generated); it never looks at future events. Use Batch only when the whole sequence is known and consistency can be planned upfront.

## Non-negotiable rules

1. **Static or tracking camera only.** Pick exactly one base behavior and state it once, verbatim:
   - Static: `The camera remains static, keeping <named subject> centered.` (Chinese output: `镜头保持静止，始终将<主体>保持在画面中央。`)
   - Tracking: `The camera tracks <named subject>, continuously keeping <named subject> centered.` (Chinese output: `镜头跟随<主体>移动，始终将<主体>保持在画面中央。`)

   Use tracking when a clear main subject moves through space, and static when it is stationary or moves only within a small central area. Convert every other requested movement (pan, zoom, push/pull, dolly, tilt, crane, rise/descend, orbit, roll) to the closer of the two: "zoom in on the runner" becomes tracking, and "orbit a stationary statue" becomes static. A static camera never reframes, widens, or narrows. Viewpoint, lens, and pace words (low angle, bird's-eye, FPV, macro, fisheye, slow motion, time-lapse) may stay when the input asks for them. Do not add "tracking shot", "static shot", or "the camera follows" anywhere else, and do not put an adverb between "camera" and "tracks" (the validator won't recognize it).
2. **Centered subject.** A clear main subject (or the active group plus core action objects) stays at or near the frame center throughout; a tracking camera keeps recentering it. Close-ups are only for stationary or small-range subjects; use medium or wide shots for subjects that travel. If there is no clear main subject, use a balanced static wide shot (`The camera remains static.`). Never invent a subject, especially not to justify tracking.
3. **Open-ended action.** Describe ongoing, repeating, or bounded motion that stays coherent if generation continues. Never use ending language: finally, eventually, in the end, at last, concluding, concludes, to conclude, in conclusion, the end of [the interaction/scene/moment], wraps up, comes to an end/close, settles into its final state; 最后, 最终, 终于, 至此结束, 告一段落. After the opening sequence (Initially / Then / Next / As), switch to Throughout / Meanwhile / All the while / As this continues.
4. **Natural pace by default.** Unless the input names a speed, motion is real-world pace. Do not stack intensifiers (rapidly, frantically, relentlessly, explosively, in a blur); express continuity with steady, sustained, rhythmic, continuous. Summarize crowds and repeated small elements broadly instead of giving each its own motion. With a static camera and nothing else moving, the subject must still visibly move (breathing, blinking, swaying, gesturing).
5. **Strict fidelity.** Keep every entity, attribute, exact quantity, event order, requested style, and on-screen text. Never invent a protagonist or major object; never drop a named entity or group; cluster ensembles ("a cluster of spice merchants"). Plausible descriptive detail on existing entities is fine.
6. **Style defaults to cinematic realism**, even for fantastical subjects, unless the input explicitly names or implies another style. The style tag appears once, at the very end, in 3–10 words (Chinese output: about 5–15 characters; realistic default: `Cinematic photography realistic style.`).
7. **Language.** Chinese or English input gives same-language output; any other language gives English. On-screen text stays in the input language. Well-known franchise characters whose original name is English keep their official English name instead of a local translation.
8. **Format.** Output only the prompt: one continuous paragraph, no line breaks, no Markdown, no labels, no self-reference ("this video shows"). Length: English 150–300 words; Chinese 150–350 Chinese characters, punctuation excluded. Budget about 45% subjects, 25% background, 30% camera, lighting, composition, style, and mood. Change the band only if the target model's caption distribution differs.

## Prompt structure (all in one paragraph)

1. Summary sentence naming every main subject and the setting.
2. Subject appearance: age, ethnicity, hair, clothing, colors, textures, distinguishing features.
3. Action in present tense: Initially → Then → Next, then the sustained state (Throughout / Meanwhile).
4. Environment, background, and secondary elements; every sound has a visible source.
5. Re-anchor the main subject's unchanging traits once more.
6. Technical block: canonical camera sentence, shot size and angle, lighting, color, composition, mood.
7. Style tag last.

## Multi-event rules (Batch and Sequential)

- **One camera setup for the whole story.** Behavior (static or tracking), viewpoint, angle, and shot size are chosen at event 1 and reused. Switch between static and tracking only when a later raw event clearly requires it. A static camera keeps its exact framing, and new subjects enter the established frame. A tracking camera keeps recentering the whole active group without dropping a co-subject. The only exception to centering is an event whose raw input explicitly says a subject leaves the frame, and only during that exit; never invent exits.
- **Consistent identities and setting.** Recurring entities are re-described with identical core traits; the location stays the same unless the input moves it.
- **No cross-event pronouns.** The first mention of an entity in each event's prompt uses its full established noun phrase. "It glides toward the diver" becomes "The large gray reef shark with a notched dorsal fin glides toward the scuba diver…". Pronouns are fine after that within the same event.
- **Continuity.** Each event starts where the previous one left off: no resets, teleporting, or changes to time of day, weather, or ongoing states.
- **Full length per event.** The length band applies to every event on its own; watch for later events drifting short.
- **Sequential only.** Never rewrite or repeat earlier events; output only the new event's prompt.

## Image-to-video (the video starts from an input image)

The image is the first frame. It fixes the subject, its appearance, the setting, the lighting, and the composition; the prompt describes that frame and how it continues.

- **Describe the image, then the motion.** The summary sentence, subject appearance, and environment describe what the image shows; the action starts from the pose and situation in the image and continues as open-ended motion.
- **Never contradict the image or add to it.** No subject, object, or setting the image does not contain, and no different clothing, colors, time of day, weather, or location. Where the user's text and the image disagree about what is visible, follow the image. A subject the text explicitly asks for but the image lacks enters the established frame during the action; it is not in the opening.
- **Shot, light, and style follow the image.** Shot size, angle, and lighting in the technical block match it, and the style tag names its look (a watercolor stays a watercolor) instead of the realistic default.
- **Seen or not seen.** If you can see the image, describe it first-hand. If you cannot, keep to what the user says about it and invent no visual details (clothing, colors, background, light); spend the length on the motion and on what the user did say.
- **The camera rule is unchanged:** static or tracking, one canonical sentence, the image's main subject centered.
- **Multi-event:** the image is the first frame of event 1; later events continue from it under the multi-event rules.

## Workflow

```
- [ ] 1. Identify the mode and output language; for multi-event, note the established camera setup and entity descriptions
- [ ] 2. Extract subjects, exact quantities, actions, setting, requested style/viewpoint/speed, on-screen text; with an input image, what it shows
- [ ] 3. Choose the camera: static or tracking (from the subject's movement), plus viewpoint, angle, and shot size
- [ ] 4. Choose the style (default cinematic realism) and lighting
- [ ] 5. Draft following the prompt structure
- [ ] 6. Validate, fix, and regenerate until clean (at most 3 attempts)
- [ ] 7. Output only the final prompt(s)
```

## Validation and retry

When a shell is available, run the validator on the draft:

```bash
python scripts/validate_prompt.py prompt.txt                 # single or sequential output
python scripts/validate_prompt.py response.txt --batch 3     # batch output, 3 events expected
echo "<prompt>" | python scripts/validate_prompt.py -
```

Exit code 0 means pass. It checks: non-empty, length band, single paragraph, no wrapper or label, no ending language, no self-reference, no camera trajectory other than static or tracking and no static reframing (a motion word that describes the subject, as in "keeping the narrowing waterfall centered", is not a camera move), exactly one camera statement (static and tracking never mixed), a tracking camera that explicitly keeps the subject centered (unless the text describes a frame exit), and (in batch mode) exactly N parts. Options: `--lang {en,zh}`, `--no-length`, `--min-words/--max-words/--min-chars-zh/--max-chars-zh`, `--json`.

Without a shell, apply the same checks by hand. Either way, also check what the script cannot see: fidelity (no invented or dropped entities, exact counts), static vs. tracking matching the subject's movement, the named subject in the camera sentence, identity re-anchoring, the style tag at the end, natural pace, visible subject motion under the static camera, cross-event consistency and pronouns, and, for image-to-video, consistency with the input image.

On failure, regenerate the complete prompt with this block appended to the request:

```text
VALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:
- <violation 1>
- <violation 2>
Regenerate the complete output and correct every issue above.
```

Stop after 3 quality attempts and report the failure rather than returning a prompt that still violates the rules.

## Building an LLM pipeline

- **System prompt:** `BASE` plus one of `MODE: SINGLE`, `MODE: BATCH`, or `MODE: SEQUENTIAL` from `system_prompt.md`. If the API has no system role, prepend it to the user message.
- **User message:** Single mode sends the raw sentence. Batch mode sends `Event 1: ...` lines, one per event. Sequential mode uses this template:

```text
PREVIOUS EVENTS (already finalized, read-only context — do NOT rewrite or repeat these):
Event 1 (already rewritten, finalized): <rewritten prompt 1>
Event 2 (already rewritten, finalized): <rewritten prompt 2>

NEW EVENT TO REWRITE:
<raw new event>

Rewrite ONLY the new event above into one complete prompt, following the Multi-Event Streaming Mode rules.
```

  For the first event, replace the history block with `PREVIOUS EVENTS (already finalized): None — this is the first event in the sequence.`
- **Image-to-video:** the image rules are part of `BASE`. Attach the image to the user message when the model accepts images; otherwise add what the user says about the image to the raw sentence.
- **Sequential loop:** rewrite event k with the validated rewrites of events 1..k-1 as history, append the result to the history, and never pass future raw events.
- **Resuming:** re-validate cached outputs and regenerate only the ones that fail.
- **Python helpers** (run from `scripts/`, or add it to `sys.path`):

```python
from validate_prompt import (
    build_system_prompt, build_streaming_user_message, build_batch_user_message,
    rewrite_with_validation,
)

system = build_system_prompt("sequential")
history = []
for event in events:
    prompt = rewrite_with_validation(call_llm, system, build_streaming_user_message(history, event))
    if prompt is None:
        raise RuntimeError(f"Rewrite failed validation: {event}")
    history.append(prompt)
```

  `call_llm(system_prompt, user_message) -> str` wraps whatever model or API is in use. For batch mode, pass `expected_count=N` to get back a list of prompts.

## Writing a MovieRoll prompt CSV

`movieroll.generate` reads one row per sample: `index`, an integer `seed`, and either a `prompt` column (single event) or `event_1`, `event_2`, … holding the N prompts of a batch or sequential run in order (see [`example/multi_event_base.csv`](../../example/multi_event_base.csv)). The video is split evenly across the events; there are no timestamp columns. Optional columns: `global_content_prompt` and `global_quality_prompt`, used only when both exist, are joined around every event as `global_content event global_quality`; the rewritten prompts are already complete, so they can be left out. `input_image` is a path relative to the CSV's directory; an empty cell means text-to-video, and one run must be all image-to-video or all text-to-video. Write the file with a CSV writer, since the prompts contain commas.
