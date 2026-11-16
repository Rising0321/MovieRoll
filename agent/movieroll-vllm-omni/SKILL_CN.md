---
name: movieroll-vllm-omni-zh
description: 通过 `vllm_omni_movieroll/` 里的 vLLM-Omni 适配器跑 MovieRoll —— 搭它钉选版本的虚拟环境、组装模型目录、设好引擎依赖的那两个环境变量、离线运行或起 `/v1/videos` 服务、选 deploy YAML 与并行布局(张量并行、Ulysses、ring、CFG 并行、HSDP、PS-VAE)、安全地覆盖采样参数,以及和同一个 venv 里的原生 MovieRoll 对拍。凡是有人要在 vLLM-Omni 下跑 MovieRoll、在那边碰到插件没被发现或请求超时的问题、要选或排查 deploy 配置、收到 compute-budget 的 400,或者问适配器输出和原仓库差多少,都用本 skill。
---

# 在 vLLM-Omni 上跑 MovieRoll

英文版是 `SKILL.md`;`references/` 下的参考资料只有英文一份。

| 文件 | 什么时候读 |
|---|---|
| `references/setup.md` | 每台机器一次:venv 以及它那些版本钉选为什么反常、SageAttention、模型目录、`movieroll_overrides`、那两个环境变量 |
| `references/run.md` | 每次运行:插件发现、离线与服务入口、请求字段、八个 deploy YAML、并行及其拒绝规则、框架限制、与原生对拍、报错对照表 |

适配器在 `vllm_omni_movieroll/`(它自己的 README 是 `vllm_omni_movieroll/README_CN.md`)。它是按框架原生写法做的
—— 用 vLLM 的层、vLLM-Omni 的 loader 和并行方案 —— 并以插件方式注册自己。

## 最容易浪费时间的三件事

1. **插件发现靠的是已安装包的元数据。** 只有在那个 venv 里跑过
   `pip install --no-deps -e vllm_omni_movieroll`,entry point 才存在。把目录丢进 `PYTHONPATH` 没有任何作用,
   症状是"模型架构未知",或者 `/v1/videos` 以 "No final video output stage" 拒绝。
2. **`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` 必须在引擎启动之前 export。** 它默认 600 秒,超时会把**所有** worker 都停掉,
   不只是这一条请求。先 `source vllm_omni_movieroll/examples/movieroll_env.sh`;事后再设没用,因为 vLLM-Omni 的
   executor 只在本进程第一个引擎启动时读一次(MovieRoll 的 compute-budget 拦截在 pipeline 构造时另读一次)。
   同一个文件还定了 `OMP_NUM_THREADS`,不设的话后处理会慢得离谱。
3. **vLLM-Omni 要用它自己的 venv**,而且那些版本钉选都是有原因的(cu129 的 torch、torchcodec 0.16、
   llguidance 1.6)。和原生对拍时只能比**同一个 venv 里**的原生 —— torch 版本不同会改变 bf16 在 CPU 上的随机流,
   也就改变了初始噪声。

## 流程

1. **先判断需求类型。** 这台机器第一次跑 → `references/setup.md`;已经搭好 → `references/run.md`;
   报错 → 去报错对照表里找,几乎每条拒绝都是故意的,并且自己写了改法。
2. **确认 venv、可编辑安装和环境变量。** 光是 `import vllm_omni_movieroll` 成功还不够(走 `PYTHONPATH` 也能
   import):在那个 venv 里 `importlib.metadata.entry_points(group="vllm_omni.general_plugins")` 要列出
   `movieroll`;超时变量要在**将要启动引擎的那个进程**的环境里。
3. **确认模型目录和链。** 一个目录一条链,请求里换不了。每个目录都是由一个 MovieRoll 模型目录转出来的
   (`download_movieroll.py --model ckpts/movieroll/base`、`distilled/t2v` 或 `distilled/i2v`);图生视频要用
   `distilled/i2v` 转出来的那个,它不加 context noise。
4. **把帧数算清楚**:`16k+1`(17、81、241、721、1441),distilled 和 i2v 至少 17 帧。不给尺寸时视频是 480x864,
   图生视频也一样。vLLM-Omni 的公共示例脚本要通过 `run_example.py` 启动;直接启动的话,脚本看不到插件声明的默认值,
   会用 Wan2.2 的预设。
5. **选一个 deploy YAML**(表在 `references/run.md`),或者自己拼布局并复核:
   `tensor_parallel_size x ulysses_degree` 要整除 16;ring 和 allgather 互斥;CFG 并行只能用在开着 guidance
   (`use_cfg`)的 base 目录上;
   PS-VAE 需要 `vae_patch_parallel_size == world_size`。**出厂的八个 YAML 都没开 PS-VAE**,要用得自己加。
6. **长请求要预料到 compute-budget 的拦截**:跑几分钟后给一个 400,并告诉你该把变量设成多少。那是设计好的行为,
   解法是提高超时、减帧数或加卡。
7. **汇报**命令、产物落在哪、真实结果。

## 硬规则

1. **采样参数是发布时评测用的取值。** 模型目录自带的差异(图生视频目录的 context noise)在转换时就写进了
   `model_index.json` 的 `movieroll_overrides`。其余的只在用户要求时改,只能走 `movieroll_overrides`(有白名单,写在
   `model_index.json` 或 deploy YAML 里,**绝不在请求里**),并说明改了什么。Base 的 guidance 也属于这类设置:
   开不开看 `use_cfg`,强度由 AnnealCFG 的 `cfg_min` / `cfg_max` 决定,请求里的 `guidance_scale` 不起作用;
   不做 guidance 的模型(distilled,或 `use_cfg: false`)收到的 `negative_prompt` 也一样,两者都只打一条警告。把
   `kv_recompute_threshold` 调高可能让窗口超出 120 帧的位置网格。
2. **不要为了让适配器跑起来去改 `movieroll/`。**
3. **优先用框架自己的效率优先设置。** 不要为了多拿几个 dB 去写模拟原生舍入行为的代码。
4. **不要绕过拒绝。** `references/run.md` 里每一条,要么对滚动窗口模型本身就不对(batch > 1、LoRA、TeaCache、
   流水并行、VAE tiling),要么是实测更差(FP8 只有 15 dB)。
5. **注意输出值域。** pipeline 交给 vLLM-Omni 的是 `[-1, 1]`;`Omni` 在 `output_type="pt"` 时返回的已经换成了
   `[0, 1]`,和原生 `--output-type pt` 一样。算任何指标前先确认两边的值域。
6. 这次发布不带测试;能用的检查就是请求校验、一次短的真实运行,以及和同 venv 原生的对拍。

## 工作循环

```
- [ ] 1. 是搭环境还是跑?读对应那份参考资料
- [ ] 2. 确认 venv、可编辑安装、movieroll_env.sh、模型目录
- [ ] 3. 帧数 = 16k+1;并行度整除 16 个头;PS-VAE 等于 world size
- [ ] 4. 启动,读第一屏输出看有没有被拒或收到 budget 400
- [ ] 5. 汇报命令、产物、结果
```
