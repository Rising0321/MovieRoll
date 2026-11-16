# MovieRoll 训练指南

[English](./TRAINING.md) · [返回 README](../README_CN.md)

本文包含训练数据准备与各阶段训练步骤。请先按 [README 中的环境和权重准备说明](../README_CN.md#setup) 完成设置。以下命令均在仓库根目录执行。

## 训练初始化权重

除 README 中的 VAE、文本编码器和所选 MovieRoll 模型外，SFT 初始化及 DMD 打分模型还需要 HunyuanVideo-1.5 的 480p transformer 权重：

```bash
hf download tencent/HunyuanVideo-1.5 --include 'transformer/480p_t2v/*' --local-dir ./ckpts
```

## 📦 训练数据准备

SFT 训练读取的是预先编码好的 latent：一个 `manifest.jsonl`，加上每个片段对应的一个 latent 文件。你只需要把视频切成
片段、写好 manifest，剩下的交给 `encode_latents`：

```
raw_clips/                                        movieroll_sft_stage1/
├── manifest.jsonl（带 "video" 字段）       ──▶   ├── manifest.jsonl（同样的内容，去掉 "video"）
└── clips/0001.mp4 ...     encode_latents         └── latents/0001.pt ...
```

**第一步：列出片段。** `manifest.jsonl` 的每一行描述一个片段：

```json
{"id": "0001", "video": "clips/0001.mp4", "text": "A chef flips a pancake in a sunlit kitchen."}
{"id": "0002", "video": "clips/0002.mp4", "text": "A day at the beach.", "duration_sec": 10.1, "events": [{"text": "Kids build a sandcastle.", "start_sec": 0.0, "end_sec": 5.0}, {"text": "A wave washes it away.", "start_sec": 5.0, "end_sec": 10.1}]}
```

| 字段 | 含义 |
|---|---|
| `id` | 片段名，对应的 latent 保存为 `latents/<id>.pt` |
| `video` | 片段路径，可以相对于 `raw_clips/`，也可以是绝对路径 |
| `text` | caption；空字符串用于训练无条件分支 |
| `events` | 可选，多事件片段的 `{text, start_sec, end_sec}` 列表 |
| `duration_sec` | 片段时长（秒），有 `events` 时必须提供。事件时间按占 `duration_sec` 的比例映射到被编码的前 241 帧上，所以片段更长时，`duration_sec` 和各事件时间都要按这前约 10 秒来填，或者先把片段裁到这个长度 |

**第二步：编码。** 脚本可以跑在任意数量的 GPU 上：

```bash
torchrun --nproc_per_node=8 -m movieroll.scripts.encode_latents \
  --src ./data/raw_clips --dst ./data/movieroll_sft_stage1
```

每个片段取前 241 帧（24 fps，略长于 10 秒），缩放到能覆盖 480×640 后居中裁剪，再用和训练器完全相同的 VAE 代码编码。
在 24 fps 下不足 241 帧的片段会报错；正好 10.0 秒的片段只有 240 帧，所以请把片段裁得稍长一些，例如 10.1 秒。
输出的 latent 是形状为 `[32, 61, 30, 40]` 的 float32 张量，已经乘过 VAE 的 scaling factor。
默认参数和仓库自带的 SFT 配置一致；任务中断后，加上 `--skip_existing` 重新运行即可接着往下跑。

蒸馏阶段不需要视频，只需要 prompt：

| 阶段 | CSV 列 |
|---|---|
| ODE 初始化（`data.prompt_path`） | `prompt`，可选 `seed`；也可以用每行一条 prompt 的 `.txt` 文件 |
| CuLT-DMD 第一阶段 / 第二阶段 | `prompt` / `event_1`、`event_2` |
| SFT 训练中采样（`sample_prompt_csv`） | `index`、`seed`，以及 `prompt` 或 `event_1..event_N` |
| CuLT-DMD 训练中采样（`sampling.test_event_csv`） | `event_1..event_N`（至少要有 `event_1`，不读取 `prompt` 列）；ACS 还需要 `input_image` |
| 批量推理（`--prompt-csv`） | `index`、`seed`，以及 `prompt` 或 `event_1..event_N`，可选 `input_image` |

## 🎓 训练

MovieRoll 的训练分为四个阶段：

```
HunyuanVideo-1.5 480p T2V
  └─① 流式 SFT，单事件 ──▶ MovieRoll-Base
       └─② 流式 SFT，多事件 ──▶ 多事件 Base
            └─③ ODE 初始化 ──▶ 少步 student
                 ├─④ CuLT-DMD ──▶ MovieRoll-Distilled（文生视频）
                 └─④ CuLT-DMD + ACS ──▶ MovieRoll-Distilled（图生视频）
```

配置文件名里的阶段编号与上图不同：① 和 ② 是 `sft/stage1_single_event.yaml` 与 `sft/stage2_multi_event.yaml`，
③ 是 `stage2_ode.yaml`，④ 是 `stage3_dmd_rolling.yaml`（图生视频为 `stage3_dmd_rolling_acs.yaml`）。

每个阶段在 `movieroll/scripts/launch/` 下都有对应的启动脚本：先在该阶段的配置文件里改好下面列出的字段，再运行脚本；
所有路径都会在加载权重之前检查一遍。配置文件路径和启动拓扑通过环境变量指定：

```bash
export NPROC=8                      # 每台机器的卡数
export NNODES=1                     # 多机训练：机器数量
export NODE_RANK=0                  # 多机训练：本机的序号
export MASTER_ADDR=127.0.0.1        # 多机训练：0 号机器的地址
```

每个启动脚本默认读取对应阶段的配置。需要更换配置时，将 `CONFIG` 仅设置在该次启动命令前，例如 `CONFIG=movieroll/configs/sft/stage2_multi_event.yaml bash movieroll/scripts/launch/train_sft.sh`；不要将某一阶段的 `CONFIG` 全局导出后用于其他阶段。

### ①② 流式 SFT

配置文件：第 ① 阶段是 `movieroll/configs/sft/stage1_single_event.yaml`，第 ② 阶段是
`movieroll/configs/sft/stage2_multi_event.yaml`。最常需要修改的字段：

```yaml
dit_weight: ckpts/transformer/480p_t2v/diffusion_pytorch_model.safetensors   # 初始权重；② 用 ckpts/movieroll/base/model.safetensors
data_root: ./data/movieroll_sft_stage1      # encode_latents 的输出目录（--dst）
output_dir: ./outputs/movieroll_sft_stage1  # checkpoint 和采样结果
height: 480                                 # 分辨率和帧数，必须与 latent 一致
width: 640
sample_n_frames: 241
sp_size: 2                                  # 序列并行组大小（② 为 4）
fsdp_sharding_strategy: hybrid              # 只有一个分片组时（如不足 8 张卡）用 full
fsdp_sub_world_size: 8                      # 每个 FSDP 分片组的卡数；必须整除总卡数（发布的训练用 16）
learning_rate: 2.0e-5                       # ② 为 1.0e-5
max_train_steps: 1000000
checkpointing_steps: 2000
streaming_chunk_size: 4                     # 每个 chunk 的 latent 帧数
streaming_window_size: 5                    # 每个滚动窗口的 chunk 数
streaming_t2v_prob: 0.8                     # 文生视频样本的比例，其余样本训练图生视频
streaming_caption_type_probs: {event: 0.0, structure: 1.0}   # ② 用事件 caption 训练：{event: 0.9, structure: 0.1}
streaming_context_noise_sigma_max: 0.15     # 给干净历史帧加噪的最大强度
sample_interval: 2000                       # 每 N 步采样一次（② 为 10000）；设为 0 关闭采样
sample_prompt_csv:                          # 采样用的 prompt
  - ./data/prompts/movieroll_bench_streaming.csv
```

启动：

```bash
bash movieroll/scripts/launch/train_sft.sh                                                      # 第 ① 阶段
CONFIG=movieroll/configs/sft/stage2_multi_event.yaml bash movieroll/scripts/launch/train_sft.sh   # 第 ② 阶段
```

> 💡 想先检查环境是否配置正确，可以在配置里设 `dry_run: true` 和 `max_train_steps: 2`，它只跑两步、不做实际训练。

### ③ ODE 初始化

配置文件：`movieroll/configs/stage2_ode.yaml`。先由 teacher 生成去噪轨迹，再让少步 student 学着沿这些轨迹去噪。

```yaml
teacher_ckpt: ./outputs/movieroll_sft_stage2/checkpoints/step-6000/ema_diffusion_pytorch_model.safetensors  # ② 的 EMA 权重
student_ckpt: ./outputs/movieroll_sft_stage2/checkpoints/step-6000/ema_diffusion_pytorch_model.safetensors  # student 从 teacher 初始化
data:
  prompt_path: ./data/ode_init/prompts.csv   # teacher 生成轨迹所用的 prompt
ode:
  trajectory_cache_dir: ./data/ode_init/rolling_ode_traj_cache   # 预计算写入、训练读取的轨迹目录
  denoising_step_list: [1000, 800, 600, 400, 200]               # student 的 5 步
video:
  num_frames: 241
  resolution: [480, 864]
logdir: ./outputs/movieroll_stage2_ode
max_steps: 1000
save_interval: 100
```

启动：

```bash
bash movieroll/scripts/launch/precompute_ode.sh    # teacher 生成轨迹
bash movieroll/scripts/launch/train_ode.sh         # student 学习这些轨迹
```

轨迹预计算的多机方式不同：每台机器各跑一个任务，设置 `SHARD_ID=<本机序号>` 和 `NUM_SHARDS=<机器数量>`，所有任务
写入同一个 `trajectory_cache_dir`。

### ④ CuLT-DMD 蒸馏

配置文件：`movieroll/configs/stage3_dmd_rolling.yaml`。student 在长 self-rollout 上学习，梯度在整条 rollout 上累积。
训练分两个阶段：先是 241 帧的单事件视频，再是 721 帧、包含两个事件的视频。

```yaml
generator_ckpt: ./outputs/movieroll_stage2_ode/<run>/checkpoints/step-1000/student.safetensors  # ③ 得到的 student
real_score_ckpt: ckpts/transformer/480p_t2v/diffusion_pytorch_model.safetensors   # teacher 打分模型
fake_score_ckpt: ckpts/transformer/480p_t2v/diffusion_pytorch_model.safetensors   # critic，从同一份权重初始化
dmd_phases:
  short_single_event:                       # 第一阶段
    max_steps: 600
    video_length: 241
    csv: ./data/movieroll/stage3_phase1_prompts.csv          # 列：prompt
  long_dual_event:                          # 第二阶段
    max_steps: 600
    video_length: 721
    csv: ./data/movieroll/stage3_phase2_event_captions.csv   # 列：event_1、event_2
dmd:
  real_guidance_scale: 4.0                  # teacher 打分时的 CFG
  context_noise: 0.15                       # rollout 时给干净历史帧加的噪声
sampling:
  test_event_csv: ./data/movieroll/stage3_test_events.csv    # 验证用的 prompt
fsdp:
  shard_world_size: 8                       # 每个 FSDP 分片组的卡数；不能超过总卡数（发布的训练用 32）
sp_size: 4
logdir: ./outputs/movieroll_stage3_dmd
save_interval: 100
ema_start_step: 200                         # 从这一步起，checkpoint 里会多一个 generator_ema.safetensors
```

启动：

```bash
bash movieroll/scripts/launch/train_dmd.sh
```

要用自己训练出的权重推理，新建一个模型目录，放入 `movieroll/configs/models/distilled/t2v/config.json`，再把
`<logdir>/<exp_name>_<时间戳>/checkpoints/step-<N>/generator_ema.safetensors` 改名为 `model.safetensors` 放进去，
然后传给 `--model`。

#### 🖼️ 图生视频蒸馏（ACS）

配置文件：`movieroll/configs/stage3_dmd_rolling_acs.yaml`，在上面的配置基础上做了这些改动：

```yaml
dmd_phases:
  long_dual_event:
    max_steps: 900                 # 第二阶段更长，总共 1500 步
dmd:
  score_mode: acs                  # 从锚帧开始的那个窗口以图像为条件打分
  sampling_context_noise: 0.0      # 训练中的采样不加 context noise
siglip_path: ckpts/siglip          # FLUX.1-Redux-dev 的 SigLIP 图像编码器
video:
  task_sampling:
    t2v_prob: 0.0                  # 只训练图生视频
sampling:
  test_event_csv: ./data/movieroll/stage3_test_events_i2v.csv   # 需要带 input_image 列
fsdp:
  shard_world_size: 8              # 必须等于总卡数
```

和第 ④ 阶段一样填好 `generator_ckpt` 和各个 CSV 路径，再下载 SigLIP 编码器（FLUX.1-Redux-dev 是 gated 仓库，需要先在
Hugging Face 上同意许可协议），然后启动训练：

```bash
hf download black-forest-labs/FLUX.1-Redux-dev --include 'image_encoder/*' 'feature_extractor/*' \
  --local-dir ./ckpts/siglip
CONFIG=movieroll/configs/stage3_dmd_rolling_acs.yaml bash movieroll/scripts/launch/train_dmd.sh
```

发布的图生视频模型是 `ckpts/movieroll/distilled/i2v`，推理时加上 `--input-image` 即可。用自己训练的权重时，按第 ④
阶段的方法建模型目录，config 换成 `movieroll/configs/models/distilled/i2v/config.json`。
