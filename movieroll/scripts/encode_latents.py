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

"""Encode raw video clips into the pre-encoded latents SFT trains on.

`--src` holds a `manifest.jsonl` whose lines are SFT manifest lines plus a `video`: the clip's path, relative
to `--src` or absolute. `--dst` becomes an SFT data root (see `movieroll/train.py`): the same manifest without
the `video` field, and `latents/<id>.pt` per line. Each clip is read with `read_video_clip` -- its first
`--num_frames` frames at `--fps`, scaled to cover `--height` x `--width` and centre-cropped -- and encoded
with the trainer's own `encode_pixel_values`, the latent sample drawn from a generator seeded with `--seed`
plus the line index.

Every process takes the lines rank, rank + world_size, ... and talks to no other, so one GPU or a torchrun
of any size works. A latent is written under a temporary name and renamed when complete, so `--skip_existing`
resumes a run that stopped halfway. Run it as a module from the repository root:

    python -m movieroll.scripts.encode_latents --src ./data/raw_clips --dst ./data/my_sft_set
    torchrun --nproc_per_node 8 -m movieroll.scripts.encode_latents --src ./data/raw_clips --dst ./data/my_sft_set

The defaults of `--height`, `--width`, `--num_frames`, `--fps` and the VAE are those of the shipped SFT
configs; change them together with the config you train with.
"""

import argparse
import json
import os
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from movieroll.models.autoencoders.vae_loader import load_vae
from movieroll.trainer.model_inputs import encode_pixel_values


def read_video_clip(path, num_frames: int, height: int, width: int, fps: float) -> torch.Tensor:
    """The first `num_frames` frames of a video at `fps`, scaled to cover `height` x `width` and centre-cropped.

    Returns float32 `pixel_values` [3, num_frames, height, width] in [-1, 1]. ffmpeg does the frame-rate
    conversion, the scaling and the crop.
    """
    video_filter = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    reader = imageio_ffmpeg.read_frames(str(path), output_params=["-vf", video_filter])
    frames = []
    try:
        next(reader)  # the metadata dict comes first
        for frame in reader:
            frames.append(np.frombuffer(frame, dtype=np.uint8))
            if len(frames) == num_frames:
                break
    finally:
        reader.close()
    if len(frames) < num_frames:
        raise ValueError(f"{path}: {len(frames)} frames at {fps} fps, but a clip needs {num_frames}")
    video = torch.from_numpy(np.stack(frames)).view(num_frames, height, width, 3)
    return video.permute(3, 0, 1, 2).float() / 127.5 - 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Encode the video clips of a manifest into SFT latents")
    parser.add_argument("--src", type=str, required=True, help="directory whose manifest.jsonl names a video per line")
    parser.add_argument("--dst", type=str, required=True, help="the SFT data root to write")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=241, help="frames per clip, 16k+1 (SFT sample_n_frames)")
    parser.add_argument("--fps", type=float, default=24, help="the frame rate clips are read at")
    parser.add_argument("--vae_path", type=str, default="ckpts/vae")
    parser.add_argument("--vae_precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0, help="line i draws its latent sample with seed + i")
    parser.add_argument("--skip_existing", action="store_true", help="keep the latents a previous run wrote")
    return parser.parse_args(argv)


def read_manifest(src: Path):
    """The lines of `src/manifest.jsonl`; each needs a string `id` and a string `video`."""
    with open(src / "manifest.jsonl", encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record.get("id"), str) or not (isinstance(record.get("video"), str) and record["video"]):
            raise ValueError(f"{src / 'manifest.jsonl'} line {line_number}: needs a string `id` and `video`")
    return records


def main(argv=None):
    args = parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        raise ValueError("--dst must differ from --src: its manifest would replace the one that names the videos")
    records = read_manifest(src)
    if not torch.cuda.is_available():
        raise RuntimeError("encoding needs a GPU")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    (dst / "latents").mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(dst / "manifest.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                sft_record = {key: value for key, value in record.items() if key != "video"}
                handle.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

    vae, _, _, _ = load_vae(args.vae_precision, args.vae_path, device=device, streaming=False)
    mine = range(rank, len(records), world_size)
    start = time.time()
    for done, index in enumerate(mine, start=1):
        record = records[index]
        out_path = dst / "latents" / f"{record['id']}.pt"
        if args.skip_existing and out_path.is_file():
            continue
        pixel_values = read_video_clip(src / record["video"], args.num_frames, args.height, args.width, args.fps)
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        latents = encode_pixel_values(pixel_values.unsqueeze(0).to(device), vae, generator)
        partial = out_path.with_name(out_path.name + ".partial")
        torch.save(latents[0].float().cpu(), partial)
        os.replace(partial, out_path)
        print(f"[rank {rank}] {done}/{len(mine)} {record['id']} {tuple(latents.shape[1:])} "
              f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
