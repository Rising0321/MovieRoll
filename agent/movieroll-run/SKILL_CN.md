---
name: movieroll-run-zh
description: 在本仓库里跑 MovieRoll —— 推理(base 50 步链或 distilled 少步链、文生视频或图生视频、单事件或多事件)和四个训练阶段中的任意一个(单事件 streaming SFT、多事件 SFT、ODE 初始化、CuLT-DMD 蒸馏,含 acs 图生视频配方)。能把"生成一段 30 秒的视频"、"用我自己的数据微调"、"把蒸馏续上"、"我这个任务为什么被拒了"这类需求,对应到该改哪些配置字段、该用哪条启动命令、该先做哪些前置检查,依据是本仓库真实的入口代码。凡是涉及用 MovieRoll 生成视频、启动或恢复某个训练阶段、选帧数或并行拓扑、准备 SFT latent 或蒸馏用的 prompt CSV、把训练产出的 checkpoint 装成可推理的模型目录,以及排查被拒绝/卡住/视频比要求短的运行,都用本 skill。
---

# 跑 MovieRoll

英文版是 `SKILL.md`;`references/` 下的参考资料只有英文一份。

## 仓库里有什么

| 路径 | 作用 |
|---|---|
| `generate_movieroll.py` → `movieroll/generate.py` | 推理,两条链共用 |
| `train_movieroll.py` → `movieroll/train.py` | streaming SFT,阶段 ① ② |
| `distill_movieroll.py` → `movieroll/distill.py` | `--mode stage2` 是 ③ ODE,`--mode stage3` 是 ④ CuLT-DMD |
| `movieroll/scripts/launch/*.sh` | 每个训练阶段的启动脚本 |
| `movieroll/configs/` | `sft/`、`stage2_ode.yaml`、`stage3_dmd_rolling{,_acs}.yaml`、`models/` |
| `movieroll/scripts/encode_latents.py` | 原始片段 → SFT 训练用的预编码 latent |

推理任务前先读 `references/inference.md`,训练任务前先读 `references/training.md`。那两份里是逐字段的契约、
必须成立的不变量和报错对照表;本文件讲的是流程。

## 第一步 —— 先判断是哪类需求

先想清楚是下面哪一类,并在回答里说出来:

- **生成视频** → 推理。用哪条链?默认 `distilled`;用户要最好画质或要 50 步的行为时用 `base`。文生视频还是
  图生视频?一个 prompt 还是多个事件?
- **训练** → 哪个阶段?"用我的视频微调"是 ①(有多事件 caption 时再加 ②);"要跑得快 / 少步"是先 ③ 再 ④;
  "i2v 蒸馏"是 ④ 的 `score_mode: acs`。
- **准备数据** → SFT 用 `encode_latents`,③ ④ 用 prompt CSV。
- **打包结果** → 把训练 checkpoint 装成 `--model` 能读的模型目录。
- **排查问题** → 先去对应参考资料的报错对照表里找那句报错,多数拒绝都是算术问题,报错文本本身已经给出了改法。

如果需求模糊到会影响命令本身(用哪条链、哪个阶段、几张卡、多少帧),就问。训练阶段不要猜:各阶段互相消费
上一阶段的 checkpoint,跑错一个要白等几个小时。

## 第二步 —— 动手之前先把算术算清楚

绝大多数跑不起来都是被校验器拦在下面这几条上。先在纸上算一遍,并在回答里写出算出来的数。

```
chunk_size C = 4,  window_size W = 5          (发布版的几何)

latent_length = (video_length - 1) // 4 + 1
要求:  latent_length % C == 1   ⇒   video_length = 16n + 1
合法:  17、81、241(10 秒)、721(30 秒)、1441(60 秒)   @ 24 fps
distilled 和所有 i2v 运行还要求 latent_length >= 1 + C  ⇒  至少 17 帧

height、width:16 的正整数倍
sp_size:要整除 16(注意力头数)⇒ 1、2、4、8、16;且 world_size % sp_size == 0
推理用 FSDP 时(--shard-world):world_size % shard_world == 0 且 shard_world % sp_size == 0
           (--shard-world 默认是 8:不到 8 张卡时要调小,或者用 --fsdp-sharding-strategy none)
训练用 hybrid 分片时(分片组 = ①② 的 fsdp_sub_world_size / ③④ 的 fsdp.shard_world_size):
           replicas = world_size // 分片组 >= 2,且 world_size % (replicas * sp_size) == 0
           (是向下取整的除法,所以分片组不必整除 world_size;
            分片组等于 world_size 时按 full 分片跑)
训练阶段 ④:至少 2 个 rank;score_mode 为 acs 时只能有一个分片组
训练阶段 ④:K = (latent_frames - 1) // C // W     (phase 1 是 3,phase 2 是 9)
```

然后确认路径都在:`ckpts/vae`、文本编码器、glyph 的两个 JSON、起步用的 checkpoint、数据根目录。每个入口都会
在加载权重前校验这些,但自己先看一眼比来回跑一趟快。

## 第三步 —— 跑

**推理。** 按 `references/inference.md` 拼命令。多卡用 `torchrun`,单卡用 `python` 加 `--sp-size 1`:

```bash
torchrun --nproc_per_node=8 generate_movieroll.py \
  --model ckpts/movieroll/distilled/t2v --sp-size 8 \
  --prompt "..." --output-dir ./outputs
```

图生视频用图生视频的模型目录,再给出第一帧(`--input-image`,CSV 批量时用 `input_image` 列),细节见
`references/inference.md` 的 "Image-to-video" 一节:

```bash
torchrun --nproc_per_node=8 generate_movieroll.py \
  --model ckpts/movieroll/distilled/i2v --sp-size 8 --input-image first_frame.jpg \
  --prompt "..." --output-dir ./outputs
```

**训练。** 改该阶段的 YAML,然后跑它的脚本。设置写在 YAML 里,不要写在命令行上:

```bash
CONFIG=movieroll/configs/sft/stage1_single_event.yaml bash movieroll/scripts/launch/train_sft.sh
bash movieroll/scripts/launch/precompute_ode.sh    # ③ 第一步
bash movieroll/scripts/launch/train_ode.sh         # ③ 第二步
bash movieroll/scripts/launch/train_dmd.sh         # ④
CONFIG=movieroll/configs/stage3_dmd_rolling_acs.yaml bash movieroll/scripts/launch/train_dmd.sh   # ④ 图生视频(ACS)
```

正式长跑之前先做一次便宜的试跑:阶段 ① 支持 `dry_run: true` 配 `max_train_steps: 2`,会把数据管线和拓扑都走
一遍但不训练。③ ④ 的配置错误几秒内就暴露(校验在 NCCL 初始化之前),所以启动后读完第一屏输出本身就是检查。

## 第四步 —— 如实汇报

说清楚实际发生了什么:跑了什么命令、产物落在哪(路径规则在 `references/inference.md` 里)、训练的话跑到第几步、
写了哪个 checkpoint。被拒绝就把那句报错原文贴出来。如果只校验了配置没真训,就说只校验了。没看到文件生成出来,
就不要说视频已经生成好了。

## 硬规则

1. **`movieroll/configs/models/*/config.json` 和 `movieroll/commons/profiles.py` 里的采样参数不是"可调默认值"**,
   它们是发布 checkpoint 评测时用的确切取值。只有用户要求时才改,并且要说明改了什么。
2. **任何东西都不能依赖 rank。** 所有会影响形状的抽样都在 sp 组 rank 0 上抽完再广播。加一个依赖 rank 的分支、
   或者在某个 rank 上多抽一次随机数,都不会报错 —— 它会让集合通信错位然后卡死。给训练或 pipeline 加代码时同样适用。
3. **改几何就作废产物。** 改了 `chunk_size`、`window_size`、帧数或分辨率,SFT 的 latent 要重新编码,ODE 的
   precompute 要重新跑(而且换一个新的 cache 目录)。
4. **③ ④ 的恢复必须是同一套拓扑**:同样的卡数、同样的 `sp_size`、同样的 FSDP 布局。要换拓扑就别 resume,
   改成从导出的 `safetensors` 重新起跑。多机跑 ④ 时,`logdir` 还必须放在每个节点都能读到的存储上:两个
   phase 交接时,每个 rank 都要把刚写的 checkpoint 读回来。
5. **context noise、KV recompute 和 AnnealCFG 别关。** 长视频能撑住靠的就是它们;为了"省事"把它们关掉会改变
   输出质量,而且关掉 KV recompute 迟早会超出 RoPE 网格。
6. **不要臆造参数名或配置键。** 所有校验器都会按名字拒绝未知键。觉得缺了什么,去读 argparse 或配置 dataclass,
   不要猜名字。
7. 本仓库不带测试。用户要"跑个测试"时,能用的手段就是校验器、`dry_run`,以及一次短的真实运行。

## 工作循环

```
- [ ] 1. 判断是推理还是哪个训练阶段,并说出来
- [ ] 2. 读对应的那份参考资料
- [ ] 3. 把算术(帧数、几何、拓扑)算一遍,确认路径都在
- [ ] 4. 长跑之前先把命令或配置改动给用户看
- [ ] 5. 启动,读第一屏输出看有没有被校验器拒掉
- [ ] 6. 汇报命令、产物和真实结果
```
