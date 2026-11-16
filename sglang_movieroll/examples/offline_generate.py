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

"""Generate one MovieRoll video through sglang Diffusion's Python entry point (`DiffGenerator`).

    python -m sglang_movieroll.examples.offline_generate --model ./movieroll-distilled \
        --prompt "A red fox trots across a snowy clearing at dawn." --num-frames 81 --output-dir out

``--num-gpus N`` with ``--ulysses-degree`` / ``--ring-degree`` / ``--tp-size`` / ``--enable-cfg-parallel``
spreads the video over several GPUs; ``--attention-backend`` and ``--enable-torch-compile`` pick sglang's
own optimisations; ``--event`` (repeated) makes a multi-event video; ``--image`` makes it
image-to-video. The mp4 is written to ``--output-dir``; ``--dump-dir`` also saves the decoded video tensor
``[1, 3, T, H, W]`` in [0, 1] as ``<output file name>.pt`` (what the comparisons with MovieRoll read).

The registration call is at module level on purpose: sglang's workers are spawned and re-import this file (as
``__mp_main__``), so they register MovieRoll in their own process too.
"""

import argparse
import os
import time

import sglang_movieroll

sglang_movieroll.register()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="a directory built by `sglang_movieroll.tools.assemble_model_dir`",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--prompt-file", default=None, help="read the prompt from a text file"
    )
    parser.add_argument(
        "--event", action="append", default=None, help="repeat for a multi-event video"
    )
    parser.add_argument(
        "--event-file",
        action="append",
        default=None,
        help="an event prompt read from a file",
    )
    parser.add_argument("--negative-prompt", default=None, help="base chain only")
    parser.add_argument(
        "--image", default=None, help="input image of an image-to-video run"
    )
    parser.add_argument("--num-frames", type=int, default=721)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=930)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--ulysses-degree", type=int, default=None)
    parser.add_argument("--ring-degree", type=int, default=None)
    parser.add_argument("--sp-degree", type=int, default=None)
    parser.add_argument("--enable-cfg-parallel", action="store_true")
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="sglang's attention backend: fa (default on Hopper), sage_attn, torch_sdpa",
    )
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--use-fsdp-inference", action="store_true")
    parser.add_argument("--dit-layerwise-offload", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--perf-dump-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-file-name", default="movieroll")
    parser.add_argument(
        "--dump-dir", default=None, help="also save the decoded video tensor here"
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="port of sglang's process group (give each job its own when several start at once)",
    )
    parser.add_argument(
        "--keep-offload",
        action="store_true",
        help="keep sglang's default CPU offload of the DiT / text encoders / VAE",
    )
    args = parser.parse_args()
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            args.prompt = handle.read().strip()
    if args.event_file:
        args.event = (args.event or []) + [
            open(path, encoding="utf-8").read().strip() for path in args.event_file
        ]
    if not args.prompt and not args.event:
        parser.error("give --prompt, --prompt-file, --event or --event-file")

    from sglang.multimodal_gen import DiffGenerator

    from sglang_movieroll.rolling.request import build_prompt

    prompt = build_prompt(args.event) if args.event else args.prompt
    kwargs = dict(model_path=args.model, num_gpus=args.num_gpus)
    for name in (
        "tp_size",
        "ulysses_degree",
        "ring_degree",
        "sp_degree",
        "attention_backend",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    for name in (
        "enable_cfg_parallel",
        "use_fsdp_inference",
        "dit_layerwise_offload",
        "warmup",
        "enable_torch_compile",
    ):
        if getattr(args, name):
            kwargs[name] = True
    if args.master_port:
        kwargs.update(master_port=args.master_port, scheduler_port=args.master_port + 1)
    if not args.keep_offload:
        kwargs.update(
            dit_cpu_offload=False, text_encoder_cpu_offload=False, vae_cpu_offload=False
        )
    generator = DiffGenerator.from_pretrained(**kwargs)
    request = dict(
        prompt=prompt,
        num_frames=args.num_frames,
        seed=args.seed,
        output_path=args.output_dir,
        output_file_name=args.output_file_name,
        save_output=True,
        # the decoded tensor comes back to this process only when asked for
        return_file_paths_only=args.dump_dir is None,
    )
    if args.profile:
        request["profile"] = True
    if args.perf_dump_path:
        request["perf_dump_path"] = args.perf_dump_path
    for name in ("height", "width", "negative_prompt"):
        if getattr(args, name) is not None:
            request[name] = getattr(args, name)
    if args.image:
        request["image_path"] = args.image
    start = time.time()
    try:
        result = generator.generate(sampling_params_kwargs=request)
    finally:
        generator.shutdown()
    if result is None:
        raise SystemExit("generation failed: see the error above")
    print(f"generated in {time.time() - start:.1f} s -> {result.output_file_path}")
    if args.dump_dir:
        import torch

        os.makedirs(args.dump_dir, exist_ok=True)
        video = (
            torch.as_tensor(result.samples).float().cpu().unsqueeze(0)
        )  # [1, 3, T, H, W] in [0, 1]
        torch.save(video, os.path.join(args.dump_dir, f"{args.output_file_name}.pt"))


if __name__ == "__main__":
    main()
