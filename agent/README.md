# Agent skills

Skills a coding agent can load to read and operate this repository. Each one is a folder with `SKILL.md`
(English) and `SKILL_CN.md` (Chinese); the `references/` files inside are English only. For Claude Code, copy a
folder into `.claude/skills/`.

供编程 agent 加载、用来读懂和操作本仓库的 skill。每个 skill 一个文件夹，内含 `SKILL.md`（英文）与
`SKILL_CN.md`（中文）；文件夹内 `references/` 下的参考资料只有英文一份。在 Claude Code 里，把文件夹复制到
`.claude/skills/` 下即可。

| Skill | What it covers | 内容 |
|---|---|---|
| [`streaming-video-prompt-rewrite`](./streaming-video-prompt-rewrite) | Rewriting a short idea, or a list of events, into the long detailed prompts MovieRoll works best with, for text-to-video and image-to-video. Includes a rule validator. | 把一句想法或一串事件改写成 MovieRoll 擅长的长 prompt，文生视频和图生视频都适用，附规则校验脚本 |
| [`movieroll-run`](./movieroll-run) | Running inference and any of the four training stages: which config fields, which launch command, which pre-flight checks. | 跑推理和四个训练阶段：改哪些配置、用哪条命令、先做哪些检查 |
| [`movieroll-port`](./movieroll-port) | Porting MovieRoll's streaming logic, image-to-video included, into another code base whose structure is not known in advance. | 把 MovieRoll 的流式逻辑（含图生视频）迁移到结构未知的其他代码库 |
| [`movieroll-sglang`](./movieroll-sglang) | Running MovieRoll through the SGLang Diffusion adapter in `sglang_movieroll/`. | 通过 `sglang_movieroll/` 的 SGLang 适配器运行 |
| [`movieroll-vllm-omni`](./movieroll-vllm-omni) | Running MovieRoll through the vLLM-Omni adapter in `vllm_omni_movieroll/`. | 通过 `vllm_omni_movieroll/` 的 vLLM-Omni 适配器运行 |

The prompt-rewrite skill is also usable without an agent: its `system_prompt.md` / `system_prompt_CN.md`, the base
section plus the section of one mode (or `build_system_prompt(mode)` in
`streaming-video-prompt-rewrite/scripts/validate_prompt.py`), works as a system prompt for any LLM, and the same
script checks rewritten prompts in either language.

prompt 改写 skill 不依赖 agent 也能用：`system_prompt.md` / `system_prompt_CN.md` 取基础部分加上某一种模式的部分
（或用 `streaming-video-prompt-rewrite/scripts/validate_prompt.py` 里的 `build_system_prompt(mode)` 拼好），就能作为
任意大模型的 system prompt，同一个脚本也能校验两种语言改写出的 prompt。
