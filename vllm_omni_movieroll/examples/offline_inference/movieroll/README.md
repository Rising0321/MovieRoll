# MovieRoll offline inference

1. Assemble a model directory once per MovieRoll model folder with `download_movieroll.py`, inside the vLLM-Omni
   environment. `--model` takes the folder (`config.json` + `model.safetensors`), as `generate_movieroll.py --model`
   does; the other components default to the main README's `ckpts/` layout (see the script's docstring).
2. Generate with the offline `Omni` API (package README), or with vLLM-Omni's shared example scripts `text_to_video.py`
   and `image_to_video.py`, unmodified. Those come from `examples/offline_inference/` of a vLLM-Omni v0.32.0 source
   checkout (they are not in the wheel); start them through `run_example.py`, which loads vLLM-Omni's plugins before
   the script looks up MovieRoll's defaults. On several GPUs source `../../movieroll_env.sh` first: it sets
   `OMP_NUM_THREADS` and vLLM-Omni's per-request time limit `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` (600 s by default) unless
   they are set (package README, Parallelism and PS-VAE).

```bash
D=vllm_omni_movieroll/examples/offline_inference/movieroll
E=vllm-omni/examples/offline_inference           # a vLLM-Omni v0.32.0 source checkout
python $D/download_movieroll.py --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python $D/download_movieroll.py --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python $D/download_movieroll.py --model ckpts/movieroll/base --out ./movieroll-base

. vllm_omni_movieroll/examples/movieroll_env.sh       # several GPUs
python $D/run_example.py $E/text_to_video/text_to_video.py --model ./movieroll-distilled --num-frames 81 --seed 930 \
    --prompt "A red fox trots across a snowy clearing at dawn." --output fox.mp4
python $D/run_example.py $E/text_to_video/text_to_video.py --model ./movieroll-base --num-frames 161 \
    --prompt "event one" --extra-body '{"events": ["event one", "event two", "event three"]}' --output events.mp4
python $D/run_example.py $E/image_to_video/image_to_video.py --model ./movieroll-distilled-i2v --image first_frame.png \
    --num-frames 81 --prompt "..." --output i2v.mp4
```

Through `run_example.py` the scripts take MovieRoll's defaults for whatever you leave out: 480 x 864, 721 frames,
24 fps, 50 steps for Base and the five-step schedule for Distilled. `--guidance-scale` is ignored, with a warning
unless it is 1: whether Base runs classifier-free guidance is the model directory's `use_cfg`, and its strength is
AnnealCFG's (`cfg_min` / `cfg_max`, also set in the directory). `--negative-prompt` likewise only counts where guidance runs;
Distilled, or Base with `use_cfg: false`, ignores it with a warning. The input image is handed to
MovieRoll as it is and scaled to cover the video, then centre-cropped, also when you pass `--height` / `--width`.
Started directly, the scripts cannot see the plugin yet when they choose their defaults, and fall back to their Wan2.2
presets. From Python, MovieRoll's own inputs ride in `OmniDiffusionSamplingParams.extra_args` (`events`,
`video_length`, `duration`); see the package README.
