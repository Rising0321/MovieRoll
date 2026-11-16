[Read in English](./README.md)

<img src="../assets/movieroll-logo.png" alt="MovieRoll: And... Action!" width="800">

### × SGLang Diffusion

`sglang_movieroll` 把 MovieRoll 的两个模型做成了原生的 [SGLang Diffusion](https://github.com/sgl-project/sglang)
（`sglang.multimodal_gen`）pipeline，单卡、多卡都能跑：Distilled（5 步，不需要 CFG）和 Base（50 步，配合 AnnealCFG）。
它没有改动 SGLang 的任何代码，只是把自己的 pipeline 注册进 SGLang，然后启动 SGLang 自己的 CLI、server 或
`DiffGenerator`。运行时也不会 import `movieroll` 包，所以不需要编译 FlashAttention。

## ✨ 亮点

- **SGLang 的各种用法都支持**：离线 Python API、`sglang` 命令行，以及 HTTP 服务（`/v1/videos`）。
- **支持文生视频、图生视频和多事件视频**，请求规则与 `generate_movieroll.py` 保持一致。
- **支持 SGLang 的并行方式**：Ulysses、ring、混合 USP、张量并行、CFG 并行（Base）和 FSDP；VAE 解码通过 PS-VAE
  （`--vae-sp`）分摊到多张卡上。
- **速度快。** 8 卡生成一分钟的 Distilled 视频，去噪只需 136 秒（开启 SageAttention 后 99 秒），原仓库需要 150 秒。

## 🛠️ 安装

SGLang 需要单独的虚拟环境，因为它锁定的 torch 和 transformers 版本与 MovieRoll 冲突。我们使用 SGLang 0.5.10.post1
（最后一个面向 CUDA 12 的版本）和 Python 3.12。在仓库根目录下执行：

```bash
bash sglang_movieroll/tools/build_venv.sh ./venv-sglang python3.12
source ./venv-sglang/bin/activate
export PYTHONPATH=$PWD        # 让 Python 找得到 sglang_movieroll
```

如果想用 SageAttention 加速采样，可以再执行下面两条命令。脚本会编译一个验证过的 commit（`v2.2.0` 版本的 Hopper kernel
在 H20 上有问题），默认只编 Hopper（sm90）的 kernel，别的显卡请先设好 `TORCH_CUDA_ARCH_LIST`。检查脚本会输出 `OK` 或
`BROKEN`：

```bash
bash sglang_movieroll/tools/build_sageattention.sh ./venv-sglang
python -m sglang_movieroll.tools.check_sageattention
```

## 🧱 模型目录

SGLang 读取 diffusers 格式的模型目录，所以发布的每个 MovieRoll 模型目录都要先转成这种格式，转一次即可。传参方式和
`generate_movieroll.py --model` 一样，直接给目录：里面的 `config.json` 决定用哪条链（Base 还是 Distilled）以及采样设置，
`model.safetensors` 提供权重。其余公共组件默认从 [主 README](../README_CN.md#setup) 的 `ckpts/` 结构里找，放在别处
的话用 `--vae-path`、`--text-encoder-path` 等参数指定。大文件默认以软链接的方式放进目录，加上 `--copy` 则会复制。

```bash
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/base --out ./movieroll-base
```

图生视频目录和文生视频目录除了权重不同，还有一项设置不同：缓存的历史帧不加 context noise。转出来的目录会保留这项设置
（写在 `model_index.json` 的 `movieroll_overrides` 里），加载模型时日志里会打印实际使用的 context noise。如果手上只有一个
单独的权重文件，也可以改用 `--dit-weight <文件>.safetensors --sampling-mode base|distilled`，这时采样设置就是这条链发布时
的默认值。

要不要做 classifier-free guidance 由模型决定，请求说了不算。Base 用 AnnealCFG 做 guidance，每一帧的强度在 `cfg_min` 和
`cfg_max` 之间变化；Distilled 没有 guidance。想关掉 Base 的 guidance，或者调整这个范围，就在转换之前把 `config.json` 里的
`use_cfg` 设为 `false`、改 `cfg_min` / `cfg_max`（已经转好的目录则改 `model_index.json` 的 `movieroll_overrides`）。
请求里的 `guidance_scale` 不起作用。

## 🚀 生成视频

单卡生成：

```bash
python -m sglang_movieroll.examples.offline_generate --model ./movieroll-distilled \
  --prompt "A red fox trots across a snowy clearing at dawn." --num-frames 81 --output-dir out
```

常用选项：

| 想要 | 加上 |
|---|---|
| 分几幕讲故事 | `--event "第一幕..." --event "第二幕..."`，视频会平均分给各幕 |
| 图生视频 | `--image ./first_frame.jpg`，最好配 `--model ./movieroll-distilled-i2v`。不设 `--height` / `--width` 时视频是 480×864，图片会等比缩放到盖满画面再居中裁剪 |
| 用更多卡 | `--num-gpus 4 --ulysses-degree 4` |
| 两张卡跑 Base | `--model ./movieroll-base --num-gpus 2 --enable-cfg-parallel`（正负两个分支各占一张卡；Distilled 会被拒绝） |
| 更快的 attention | `--attention-backend sage_attn` |
| 单卡生成长视频 | `--keep-offload`（保留 SGLang 的 CPU offload，给 VAE 解码腾出显存） |

如果要自己写启动脚本，请在脚本的顶层调用 `sglang_movieroll.register()`。SGLang 的 worker 是用 spawn 方式启动的，
会重新 import 这个脚本，所以每个 worker 里都要执行一次注册。

### 服务部署

`python -m sglang_movieroll.cli` 就是注册好 MovieRoll 的 `sglang` 命令，`generate` 和 `serve` 都可以直接用：

```bash
python -m sglang_movieroll.cli serve --model-path ./movieroll-distilled --port 30000 --num-gpus 2
curl 127.0.0.1:30000/v1/videos -H 'Content-Type: application/json' \
  -d '{"prompt": "A red fox trots across a snowy clearing at dawn.", "num_frames": 81}'
# 返回 {"id": ...}；轮询 GET /v1/videos/<id> 直到状态变成 completed，再用 GET /v1/videos/<id>/content 下载视频
```

`width` 和 `height` 可以不传，默认 480×864。图生视频请求通过 `input_reference` 传图：可以用 multipart 上传文件，也可以在
JSON 里写图片路径。

通过 HTTP 发请求时有两点要注意：

- **一定要传 `num_frames`**（满足 `16k+1` 的帧数）。接口会把时长换算成 `fps × 秒数`，结果通常不是 MovieRoll 支持的
  长度，这样的请求会返回 400，并给出最接近的合法帧数。
- **多事件视频的事件要写在 prompt 里**，格式是 JSON：
  `{"events": ["第一幕...", "第二幕..."]}`。SGLang 没有其他途径能把模型专属的字段传给 worker。

## ⚡ 并行

| 参数 | 作用 |
|---|---|
| `--ulysses-degree U` | 把窗口里的 token 分到多张卡上，在 attention 前后与 head 维互换 |
| `--ring-degree R` | key 沿环形依次传递，再合并各段的 attention 结果（SGLang 只允许配 `fa` 或 `sage_attn` 后端） |
| `--ulysses-degree U --ring-degree R` | 两者同时使用（SGLang 的混合 USP） |
| `--tp-size T` | 切分 attention head 和前馈层，`T × U` 必须能整除 16 个 head |
| `--enable-cfg-parallel` | 仅限 Base：正负两个分支在不同的卡上同时计算 |
| `--use-fsdp-inference` | 对 transformer block 分片，每个 block 计算时再聚合；不能和 `--tp-size` 同时用 |
| `--vae-sp` | PS-VAE：把 VAE 解码分摊到序列并行组里的各张卡上；有序列并行组时默认开启。只有 `sglang_movieroll.cli` 命令接受这个参数，离线示例没有这个选项，始终保持开启 |

在 H20 上用 Distilled 文生视频、81 帧实测的多卡结果：

| 布局 | 卡数 | 耗时 | 加速比 |
|---|---|---|---|
| 单卡 | 1 | 62.0 秒 | 1.00× |
| `--ulysses-degree 2` | 2 | 36.4 秒 | 1.70× |
| `--ulysses-degree 4` | 4 | 29.4 秒 | 2.11× |
| `--tp-size 2 --ulysses-degree 2` | 4 | 29.2 秒 | 2.12× |
| `--use-fsdp-inference` | 2 | 38.7 秒 | 1.60× |
| `--enable-cfg-parallel`，Base | 2 | 412.1 秒（单卡 796.2 秒） | 1.93× |

PS-VAE 还能降低解码的显存峰值：单卡 56.3 GiB，两卡每卡 30.3 GiB，四卡每卡 17.3 GiB。长视频正是靠它才放得下。

## 🎞️ 一分钟视频

在 H20-3e 上生成 60 秒视频（1441 帧，24 fps，480×864），下表只统计滚动去噪循环的耗时：

| 模型 | 布局 | 本适配 | 原仓库 |
|---|---|---|---|
| Distilled | 单卡 | 736 秒 | 769 秒 |
| Distilled | `--ulysses-degree 8` | 136 秒 | 150 秒 |
| Distilled | `--ulysses-degree 8`，SageAttention | 99 秒 | – |
| Base | `--enable-cfg-parallel --ulysses-degree 4`（8 卡） | 1882 秒 | 2224 秒 |
| Base | 同上，SageAttention | 1198 秒 | – |

整整一分钟里画面始终清晰。单卡生成一分钟的 Distilled 视频需要加 `--keep-offload`。

**100 分钟的限制。** SGLang 0.5.10.post1 会放弃运行超过 100 分钟的请求。单卡生成一分钟的 Base 视频大约要 4 个小时，
所以请用上表中的 8 卡布局。适配层不会等到超时才失败：生成两个 chunk 后，它会估算总耗时，如果来不及完成，就直接拒绝
请求，并给出估算结果。

## ✅ 支持与不支持的功能

SGLang 提供、但 MovieRoll 无法支持的功能，都会在启动时或构造请求时明确拒绝，并说明原因，不会在视频生成到一半时才报错。
只有三个请求字段是打印警告后忽略，而不是拒绝：`generator_device`（SGLang 的视频接口每个请求都会带上它，而每条链都在自己
固定的设备上抽噪声）；不等于 SGLang 默认值 1 的 `guidance_scale`（做不做 guidance 看模型的 `use_cfg`，强度由 AnnealCFG
决定）；以及不做 guidance 的模型收到的 negative prompt（Distilled，或者设了 `use_cfg: false` 的 Base），这时它用不上。

**支持：** 单卡；Ulysses、ring、混合 USP、张量并行、CFG 并行（Base）、FSDP / HSDP 和 `--vae-sp`；CPU offload 和逐层
DiT offload；warmup；性能分析和 `--perf-dump-path`；`fa`、`sage_attn`、`torch_sdpa` 三种 attention 后端（SGLang 只允许
`fa` 和 `sage_attn` 配 `--ring-degree`）；`--enable-torch-compile`（可以用，但会更慢：KV cache 每一轮都在变长，导致反复重新编译）；
通过 `/v1/videos` 部署服务。

| 不支持 | 原因 |
|---|---|
| `--vae-tiling` | 空间分块之间会做混合，视频会因此改变；`--vae-sp` 分摊解码时不做空间分块 |
| `--vae-slicing` | 它是把一个 batch 拆开解码，而 MovieRoll 一个请求只有一条视频，本来就是按 chunk 逐段解码的 |
| Cache-DiT、TeaCache | 它们在同一个 latent 的相邻步之间复用残差，而滚动窗口每一步处理的 chunk 都不一样 |
| `sage_attn_3` | 它的 FP4 kernel 只能在 Blackwell GPU 上运行 |
| 稀疏 attention 后端 | 有损，而且假定整段序列做 self-attention，而 MovieRoll 要 attend 到 cache 前缀 |
| Distilled，或设了 `use_cfg: false` 的 Base 目录，配 `--enable-cfg-parallel` | 没有第二个 guidance 分支，另一张卡无事可做 |
| `--transformer-weights-path`、`--enable-svdquant`（FP8、NVFP4、ModelSlim、SVDQuant） | 没有量化版的 MovieRoll 权重，既没有可加载的，也没有可对照验证的 |
| `--dit-precision` 不是 `bf16` | 两条链都是用 bf16 的 transformer 评测的，MovieRoll 从未在别的精度下运行过 |
| LoRA | 目前没有 MovieRoll 的 LoRA，无从验证 |
| `comfyui_mode`（由 SGLang 的 ComfyUI 集成设置） | 它每一步只驱动一个 latent，滚动窗口没有这样的结构 |
| `num_outputs_per_prompt > 1`、`true_cfg_scale`、`guidance_rescale`、`cfg_normalization`、`boundary_ratio`、`return_trajectory_*` | 不属于 MovieRoll 的采样器 |

**SGLang 0.5.10.post1 本身不提供：** 数据并行、流水线并行、取消正在生成的视频，以及通过 HTTP 查询进度（服务端会在
日志里记录每个完成的 chunk）。MovieRoll 只能跑在 SGLang 的原生后端上：`--backend diffusers` 没有可以加载的 MovieRoll
pipeline。

## 📜 许可证

本目录遵循 MovieRoll 的 [许可证](../LICENSE)。改编自 Apache-2.0 项目的代码保留了原有声明，并列在 [NOTICE](./NOTICE) 中。
