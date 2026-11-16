[Read in English](./README.md)

<img src="../assets/movieroll-logo.png" alt="MovieRoll: And... Action!" width="800">

### × vLLM-Omni

`vllm_omni_movieroll` 是一个 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 插件，把 MovieRoll 的两个模型
做成原生的 diffusion pipeline `MovieRollPipeline`，单卡、多卡都能跑：Distilled（5 步，不需要 CFG）和 Base（50 步，
配合 AnnealCFG）。它通过 vLLM-Omni 的插件入口完成注册，不改动 vLLM-Omni 的任何代码，运行时也不会 import `movieroll`
包。运行时沿用 vLLM-Omni 自己的默认配置：区域化的 `torch.compile`、vLLM 的并行层，以及平台默认的 attention 后端。

## ✨ 亮点

- **vLLM-Omni 的各种用法都支持**：离线 `Omni` API、vLLM-Omni 自带的示例脚本，以及 `vllm serve --omni` 提供的
  `/v1/videos` 接口（同步、异步，生成途中也能取消）。
- **支持文生视频、图生视频和多事件视频**，请求规则与 `generate_movieroll.py` 保持一致。
- **支持 vLLM-Omni 的并行方式**：张量并行、Ulysses、ring、AllGather-KV、CFG 并行（Base）、HSDP 和 CPU offload，
  另外还能用 PS-VAE 把 VAE 解码分摊到多张卡上。
- **速度快。** 8 卡生成一分钟的 Distilled 视频，去噪只需 113 秒（开启 SageAttention 后 78 秒），原仓库需要 150 秒。

## 🛠️ 安装

vLLM-Omni 需要单独的虚拟环境，因为它锁定的 torch 和 transformers 版本与 MovieRoll 冲突。我们使用 vLLM-Omni 0.32.0、
vLLM 0.32.0、torch 2.13.0（CUDA 12.9）和 Python 3.12。脚本还会以可编辑模式安装本插件，把它注册到 vLLM-Omni：

```bash
bash vllm_omni_movieroll/tools/build_venv.sh ./venv-vllm-omni python3.12
source ./venv-vllm-omni/bin/activate
```

如果想用 SageAttention 加速采样，可以再执行下面的命令。脚本会编译一个验证过的 commit（`v2.2.0` 版本的 Hopper kernel
在 H20 上有问题），默认只编 Hopper（sm90）的 kernel，并自动检查编译结果；别的显卡请先设好 `TORCH_CUDA_ARCH_LIST`：

```bash
bash vllm_omni_movieroll/tools/build_sageattention.sh ./venv-vllm-omni
```

## 🧱 模型目录

vLLM-Omni 读取 diffusers 格式的模型目录，所以发布的每个 MovieRoll 模型目录都要先转成这种格式，在 vLLM-Omni 的虚拟环境
里转一次即可。传参方式和 `generate_movieroll.py --model` 一样，直接给目录：里面的 `config.json` 决定用哪条链（Base 还是
Distilled）以及采样设置，`model.safetensors` 提供权重。其余公共组件默认从 [主 README](../README_CN.md#setup) 的
`ckpts/` 结构里找，放在别处的话用 `--vae`、`--text-encoder` 等参数指定。transformer 和 Qwen2.5-VL 以软链接的方式放进
目录（加 `--copy` 则会复制），glyph 编码器和 VAE 会转换一次格式。

```bash
D=vllm_omni_movieroll/examples/offline_inference/movieroll
python $D/download_movieroll.py --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python $D/download_movieroll.py --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python $D/download_movieroll.py --model ckpts/movieroll/base --out ./movieroll-base
```

图生视频目录和文生视频目录除了权重不同，还有一项设置不同：缓存的历史帧不加 context noise。转出来的目录会保留这项设置，
写在 `model_index.json` 的 `movieroll_overrides` 里；同一个文件里的类名标明了是哪条链：Base 是 `MovieRollPipeline`，
Distilled 是 `MovieRollDistilledPipeline`。如果手上只有一个单独的权重文件，也可以改用
`--dit <文件>.safetensors --sampling-mode base|distilled`，这时采样设置就是这条链发布时的默认值。

要不要做 classifier-free guidance 由模型决定，请求说了不算。Base 用 AnnealCFG 做 guidance，每一帧的强度在 `cfg_min` 和
`cfg_max` 之间变化；Distilled 没有 guidance。想关掉 Base 的 guidance，或者调整这个范围，就在转换之前把 `config.json` 里的
`use_cfg` 设为 `false`、改 `cfg_min` / `cfg_max`（已经转好的目录改 `model_index.json` 的 `movieroll_overrides`，也可以写在
部署 YAML 的 `model_config.movieroll_overrides` 里）。请求里的 `guidance_scale` 不起作用。

## 🚀 生成视频

用离线 Python API 生成：

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="./movieroll-distilled")
out = omni.generate(
    {"prompt": "A red fox trots across a snowy clearing at dawn."},
    OmniDiffusionSamplingParams(num_frames=721, seed=930),
)
```

| 想要 | 设置 |
|---|---|
| 分几幕讲故事 | `extra_args={"events": ["第一幕...", "第二幕..."]}`，视频会平均分给各幕。幕数多于 chunk 数、有幕分不到 chunk 时，Base 会丢掉这一幕并给出警告，Distilled 则直接报错 |
| 图生视频 | 在 prompt 字典里加 `"multi_modal_data": {"image": pil_image}`（HTTP 请求用 `input_reference`），最好配 `./movieroll-distilled-i2v`。不设 `height` / `width` 时视频是 480×864，图片会等比缩放到盖满画面再居中裁剪 |
| 换长度或分辨率 | `num_frames`（`16k+1`，默认 721），`height` / `width`（16 的倍数，默认 480×864） |
| Base 的采样参数 | `num_inference_steps`（5 的倍数，默认 50）、`negative_prompt` |
| SageAttention | 设置环境变量 `DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN` |

两种模型都不看请求里的 `guidance_scale`：只要它不是 vLLM-Omni 的默认值 1，就会被忽略并给出警告。做不做 guidance 看
目录里的 `use_cfg`，强度由 AnnealCFG 决定。Distilled 始终用固定的 5 步采样，传入别的 `num_inference_steps` 会被忽略并
给出警告。不做 guidance 的模型（Distilled，或者设了 `use_cfg: false` 的 Base）没有无条件分支，收到的 `negative_prompt`
用不上，同样会被忽略并给出警告。

vLLM-Omni 的公共示例脚本 `text_to_video.py` 和 `image_to_video.py` 不用改也能跑。它们不在 vLLM-Omni 的 wheel 里，要从
vLLM-Omni v0.32.0 源码的 `examples/offline_inference/` 下取。请通过 `run_example.py` 来启动：它先加载 vLLM-Omni 的插件，
再原样运行脚本，后面的参数都交给脚本自己：

```bash
python vllm_omni_movieroll/examples/offline_inference/movieroll/run_example.py \
  vllm-omni/examples/offline_inference/image_to_video/image_to_video.py \
  --model ./movieroll-distilled-i2v --image first_frame.jpg --prompt "..." --num-frames 81 --output eagle.mp4
```

插件会向 vLLM-Omni 声明 MovieRoll 的默认值（480×864、721 帧、24 fps；Base 50 步，Distilled 用它固定的 5 步；不声明
guidance scale，所以脚本也不会发送），并声明输入图片由 MovieRoll 自己处理尺寸。示例脚本是在创建引擎之前查这两样的，
而 vLLM-Omni 到创建引擎时才加载插件，所以直接启动脚本时两样都查不到，脚本会退回它给 Wan2.2 准备的预设：文生视频是
1280×720、40 步、guidance 4.0；图生视频是 16 fps、50 步、guidance 5.0、按图片定尺寸，而且只要传了 `--height` /
`--width`，图片就会被直接拉伸到这个尺寸。

### 服务部署

```bash
MODEL=./movieroll-distilled bash vllm_omni_movieroll/examples/online_serving/movieroll/run_server.sh
curl -s http://127.0.0.1:8091/v1/videos/sync -F prompt="A red fox trots across a snowy clearing at dawn." \
  -F num_frames=81 -F seed=930 -o fox.mp4
```

请求不带 `size` 时视频是 864x480（宽 × 高），不带 `fps` 时是 24 fps；图生视频请求的 `input_reference` 会原样交给
MovieRoll，由它缩放到盖满画面再居中裁剪。多个事件以 JSON 放在 `extra_params` 表单字段里，例如
`-F 'extra_params={"events": ["...", "..."]}'`。请求不带 `seed` 时 vLLM-Omni 会随机取一个，想复现同一个视频就要传
`seed`。同一目录下的 `run_curl.sh` 会提交一个三事件的异步任务，轮询状态并下载视频。`DELETE /v1/videos/<id>` 可以取消任务，
任务会在下一个窗口处停下。要用 SageAttention，在启动服务的命令后加上 `--diffusion-attention-backend SAGE_ATTN`。

## ⚡ 并行与 PS-VAE

`deploy/` 目录里准备好了常用布局的配置：`movieroll_usp2.yaml`、`_usp4`、`_ring2`、`_usp2_ring2`、`_tp2`、`_hsdp2`
和 `_base_cfg2_usp2`。也可以直接传参数：`--ulysses-degree`、`--ring-degree` 或 `--allgather-degree`、
`--tensor-parallel-size`、`--cfg-parallel-size 2`（仅限开着 guidance 的 Base）、`--use-hsdp --hsdp-shard-size` 以及
`--vae-patch-parallel-size`。16 个 attention head 必须能被张量并行数与 Ulysses 并行数的乘积整除。

**PS-VAE。** 把 `--vae-patch-parallel-size N` 设成卡数，就会启用 MovieRoll 的并行流式解码：每张卡解码一部分行，最后
再汇总。它不做空间分块，结果和单卡解码一致，只有浮点舍入上的差别。vLLM-Omni 仍会打印 “VAE patch parallelism is NOT
enabled for MovieRollPipeline; ignoring”（或 `MovieRollDistilledPipeline`），这句话说的是 vLLM-Omni 自己的分块解码，
MovieRoll 并不使用，可以忽略。

**环境变量。** 多卡运行之前，以及任何服务部署之前（deploy 目录下的 YAML，包括单卡那份，都用多进程 executor），请先
source `vllm_omni_movieroll/examples/movieroll_env.sh`（`run_server.sh` 会自动执行）。它会设置两个变量：

- `OMP_NUM_THREADS=16`。不设置的话，vLLM 会把它设成机器的 CPU 总数，逐帧的视频后处理会因此变慢，在负载较高的机器上
  可能多花一分钟以上。
- `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400`。不设置的话，vLLM-Omni 会中止任何超过 600 秒的请求，而一分钟的 Base 视频即使
  用 8 卡也要 30 分钟左右。如果请求仍然会超出这个限制，MovieRoll 会提前返回 400，并指出需要的数值，服务本身照常运行。

在 H20 上用文生视频、81 帧实测（首个请求）：

| 模型 | 布局 | 耗时 |
|---|---|---|
| Distilled | 单卡 / Ulysses 2 / Ulysses 4 / TP 2 × Ulysses 2 | 63 秒 / 46 秒 / 35 秒 / 31 秒 |
| Base | 单卡 / CFG 2 / CFG 2 × Ulysses 2 | 775 秒 / 412 秒 / 229 秒 |

PS-VAE 能降低解码的显存峰值：单卡 16.7 GB，两卡每卡 8.5 GB，四卡每卡 4.6 GB。

## 🎞️ 一分钟视频

在 H20-3e 上生成 60 秒视频（1441 帧，24 fps，480×864），下表只统计滚动去噪循环的耗时：

| 模型 | 布局 | 本插件 | 原仓库 |
|---|---|---|---|
| Distilled | 单卡 | 711 秒 | 769 秒 |
| Distilled | Ulysses 8 + PS-VAE 8 | 113 秒 | 150 秒 |
| Distilled | 同上，SageAttention | 78 秒 | – |
| Base | CFG 2 × Ulysses 4 + PS-VAE 8 | 1830 秒 | 2224 秒 |
| Base | 同上，SageAttention | 1192 秒 | – |

整整一分钟里画面始终清晰。多卡运行和部署服务时别忘了 source `movieroll_env.sh`。

## ✅ 功能支持

**支持，并已在发布权重上验证：** Distilled 和 Base；文生视频、图生视频和多事件；`torch.compile` 和
`--enforce-eager`；张量并行 2、4；Ulysses 2、4、8；ring 2；AllGather-KV 2；CFG 并行 2（Base）；HSDP 2、4；PS-VAE；
模型级和逐层 CPU offload；`FLASH_ATTN`、`TORCH_SDPA`、`SAGE_ATTN` 三种 attention 后端；721 帧和 1441 帧的视频；离线
`Omni`、公共示例脚本和 `vllm serve --omni`（同步、异步、取消）；warm-up 和 pipeline profiler。

下列功能会在加载权重之前直接报错，并说明原因：

| 不支持 | 原因 |
|---|---|
| FP8（`--quantization fp8`） | Distilled 生成的视频和 `generate_movieroll.py` 的相比 PSNR 只有 15.0 dB，已经是明显不同的视频，而且没有提速 |
| 流水线并行 | 一个视频是一串严格顺序执行的窗口计算，各个 stage 只会互相等待 |
| Cache-DiT、TeaCache | 滚动窗口每次调用处理的窗口都不同，时间步也是逐帧的，没有可以复用的计算 |
| 逐步执行、流式输出 | 需要把 pipeline 拆成逐步调用的方法，当前版本不提供 |
| LoRA | 目前没有 MovieRoll 的 LoRA |
| `SAGE_ATTN_3` | vLLM-Omni 只允许在 Blackwell GPU 上用 |
| VAE 分块与切片 | 分块之间会做混合，视频会因此改变；流式解码本身已经能控制显存 |
| 非 strict 的 Ulysses（`ulysses_mode`）、ring 与 allgather 同时用、`--vae-patch-parallel-size` 既不是 1 也不是卡数、和 checkpoint 不一致的 `flow_shift` | 都会改变或破坏滚动计算 |
| 分布式逐层 offload、专家并行、`text_encoder_tp_size`、paged diffusion KV、量化的 attention K/V、`max_num_seqs > 1` | 未经验证，或不适用于 MovieRoll（没有 MoE 层；一个请求对应一个视频） |

vLLM-Omni 的交互式流播放（`/v1/realtime/video` 加 `streaming_buffer_seconds`）同样不能用。这是 vLLM-Omni 服务端的
功能，不是 MovieRoll 的设置；因为 `MovieRollPipeline` 不是可交互的分块 pipeline，这类请求由 vLLM-Omni 自己拒绝。

## 📜 许可证

本目录遵循 MovieRoll 的 [许可证](../LICENSE)。改编自 Apache-2.0 项目（vLLM-Omni、diffusers、ring attention）的代码
保留了原有声明，并列在 [NOTICE](./NOTICE) 中。
