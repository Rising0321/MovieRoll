---
name: movieroll-sglang-zh
description: 通过 `sglang_movieroll/` 里的 SGLang Diffusion 适配器跑 MovieRoll —— 搭它钉选版本的虚拟环境、把发布的 checkpoint 组装成 diffusers 风格的模型目录、离线生成或起 HTTP 服务、选多卡布局(张量并行、Ulysses、ring、CFG 并行、FSDP、PS-VAE),以及和同一个 venv 里的原生 MovieRoll 对拍。凡是有人要在 SGLang 下跑 MovieRoll、在那边碰到注册或模型目录的问题、要选或要排查 SGLang 的并行布局、撞上 100 分钟的客户端超时或请求被拒,或者问适配器的输出和原仓库差多少,都用本 skill。
---

# 在 SGLang Diffusion 上跑 MovieRoll

英文版是 `SKILL.md`;`references/` 下的参考资料只有英文一份。

| 文件 | 什么时候读 |
|---|---|
| `references/setup.md` | 每台机器一次:venv、SageAttention、从 MovieRoll 模型目录组装 SGLang 模型目录、采样参数从哪来 |
| `references/run.md` | 每次运行:注册、离线与服务入口、帧数算术、多卡、框架限制、与原生对拍、报错对照表 |

适配器在 `sglang_movieroll/`(它自己的 README 是 `sglang_movieroll/README_CN.md`)。它是按框架原生写法做的:
用 SGLang 自己的 loader、层和并行方案,而不是把 MovieRoll 的 pipeline 包一层;运行时**不** import `movieroll` 包。

## 先记住两件事

这里绝大部分时间都是浪费在这两条上:

1. **SGLang 要用它自己的 venv。** 它钉的 torch 和 transformers 和 MovieRoll 的冲突。绝对不要装进你跑原生
   MovieRoll 的那个环境里 —— 而且对拍时只能和**同一个 venv 里**的原生比:torch 版本不同会改变 bf16 在 CPU 上的
   `randn` 随机流,也就改变了初始噪声。
2. **`register()` 必须写在启动脚本的模块顶层。** SGLang 用 spawn 起 GPU worker,每个 worker 是全新解释器并会重新
   import 启动脚本,所以注册写在 `main()` 里会让 worker 找不到 pipeline。`cli.py` 和
   `examples/offline_generate.py` 就是正确的写法。

## 流程

1. **先判断需求类型。** 这台机器第一次跑 → 读 `references/setup.md`;已经搭好了 → 直接看 `references/run.md`;
   报错 → 先去报错对照表里找那句话,多数拒绝都是故意的,报错文本本身就给了改法。
2. **确认当前环境是适配器那个 venv**,并且 `PYTHONPATH` 里有仓库根目录。
3. **确认模型目录存在、而且是用户要的那条链。** 一个目录对应一条链,请求里换不了 base / distilled。每个目录都是由一个
   MovieRoll 模型目录转出来的(`--model ckpts/movieroll/base`、`distilled/t2v` 或 `distilled/i2v`);图生视频要用
   `distilled/i2v` 转出来的那个,它不加 context noise。
4. **把帧数算清楚**:`num_frames = 16k + 1`(17、81、241、721、1441)。把你选的数写出来。
5. **按 `references/run.md` 的表选布局**,并复核:`tp_size x ulysses_degree` 要整除 16;CFG 并行只能用在开着
   guidance(`use_cfg`)的 base 目录上;FSDP 和 TP 互斥。
6. **长任务启动前先估时间。** 适配器会把预计超过 SGLang 100 分钟客户端超时的请求**提前拒掉** —— 那是设计,
   不是 bug,解法是加卡或分段生成,不是重试。
7. **汇报**命令、产物落在哪、真实结果。

## 硬规则

1. **`rolling/profiles.py` 里的采样参数是发布时评测用的确切取值,不是可调默认值。** 模型目录的 `config.json` 只允许在
   MovieRoll 自己的 stage 读的那几项上与之不同(context noise、KV recompute 策略、`use_cfg`,以及 AnnealCFG 的范围
   `cfg_min` / `cfg_max`),转换工具会把这几项写进 `model_index.json`。Base 的 guidance 是模型的设置:开不开看
   `use_cfg`,强度由 AnnealCFG 决定,请求里的 `guidance_scale` 不起作用。其余的只在用户要求时改,并说明改了什么。
2. **不要为了让适配器跑起来去改 `movieroll/`。** 是适配器去适配,包本身保持原样。
3. **`registry.py` 里的注册顺序不能动。** 先组件类,再跑 SGLang 自己的 pipeline 发现,最后才加 MovieRoll 的
   pipeline。顺序反了会静默地让所有内置 pipeline 都没注册上。
4. **优先用 SGLang 自己的效率优先设置**(它的层、它的 attention、它的 loader)。不要为了多拿几个 dB 去写模拟原生
   MovieRoll 舍入行为的代码。
5. **不要绕过拒绝。** `references/run.md` 里列的那些拒绝,要么是因为该功能对滚动窗口模型本身就不对(LoRA、
   teacache、batch > 1、VAE tiling……),要么是这个版本里坏的。
6. **`--enable-torch-compile` 在这里不是白捡的收益** —— KV cache 每轮都在变长,会反复触发重新编译。
7. 这次发布不带测试;能用的检查就是客户端的请求校验、一次短的真实运行,以及和同 venv 原生的对拍。

## 工作循环

```
- [ ] 1. 是搭环境还是跑?读对应那份参考资料
- [ ] 2. 确认 venv、PYTHONPATH、模型目录
- [ ] 3. 帧数 = 16k+1;并行度要整除 16 个头
- [ ] 4. 对着 100 分钟上限估一下耗时
- [ ] 5. 启动,读第一屏输出看有没有被拒
- [ ] 6. 汇报命令、产物、结果
```
