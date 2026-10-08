# embeddinggemma2-zh

`google/embeddinggemma-2`（2026-10-06 发布的 740M 多模态嵌入模型）的**中文评测与微调**。

> **中文圈第一个 C-MTEB 全量实测。** 发布后 48 小时内，声明
> `base_model=google/embeddinggemma-2` 的 60 个衍生仓库里**零中文、零 C-MTEB 数字**
> （GitHub 搜 `embeddinggemma + C-MTEB` = 0 仓库，`+ chinese` = 0 仓库），本项目补这个空白。
>
> 权重与模型卡：<https://huggingface.co/Weidows/embeddinggemma-2-zh>

## 结果（`MTEB(cmn, v1)`，31 个中文任务，全量跑完）

| 模型 | 参数量 | 总体 | 备注 |
|---|---|---|---|
| **`google/embeddinggemma-2`**（只载文本骨干） | **271M** | **0.5815** | 本次实测，173.8 min |
| `BAAI/bge-m3` | 568M | **0.6068** | 本机同 harness 实跑，84.9 min |
| `Qwen/Qwen3-Embedding-0.6B` | 595M | **0.6748** | `mteb/results` 公开数 |

- **用 bge-m3 一半的参数量拿到它约 96% 的中文成绩**，且 7 个单项反超 bge-m3。
- **真短板**：医疗领域（CmedqaRetrieval −0.092、MedicalRetrieval −0.109、CMedQAv1/2-reranking ≈ −0.11）
  与中文 STS 的 AFQMC / ATEC / BQ（落后 bge-m3 0.10~0.12）。
- **强项**：通用检索（T2 0.7618 / MMarco 0.7825 / Du 0.7637 / Covid 0.7721）、新闻聚类（反超 bge-m3 +0.07）。
- 两个"看起来是短板其实不是"的陷阱也写进报告了：PAWSX（bge-m3 也只有 0.1566）、聚类
  （我们 0.4925 **高于** bge-m3 0.4615）。
- 完整 31 任务明细 + 逐任务 delta → [`report/cmteb-embeddinggemma2-zh.md`](report/cmteb-embeddinggemma2-zh.md)
  · [`results/cmteb_comparison.md`](results/cmteb_comparison.md)

## 微调结论：**零和**（0.5815 → 0.5851，+0.0037）

针对上面 6 个短板做了定向 LoRA 微调，**净收益实质上为零** —— 涨的和跌的互相抵消。
把这个结论和**全部**回归数据如实公开，比发布一个只报喜的权重更有用。

**做法**：LoRA r=32 / α=64 / 9 个 target（含 `embedding_projection` 与 24 层 `per_layer_projection`），
可训练 **10.92M（4.03%）**；数据 **273,549 对**（中文 STS 156k 官方 train split + cMedQA-V2.0 医疗 120k，
两侧均已去污染到**查询/文档零重叠**）；1 epoch / bs 64 / lr 2e-4 / bf16，**85.2 min**，
train_loss 0.9009 → 0.2512，峰值显存 9.85 GB。

| 任务类型 | 微调前 | 微调后 | Δ |
|---|---|---|---|
| Retrieval | 0.6179 | 0.5893 | **−0.0287** |
| Reranking | 0.5664 | 0.6219 | **+0.0555** |
| PairClassification | 0.6732 | 0.6249 | **−0.0483** |
| Clustering | 0.4925 | 0.4962 | +0.0037 |
| STS | 0.4441 | 0.4853 | **+0.0412** |
| Classification | 0.6948 | 0.6933 | −0.0014 |
| **总体** | **0.5815** | **0.5851** | **+0.0037** |

**涨**：BQ +0.1665（反超 bge-m3）、CMedQAv1-reranking +0.1353、CMedQAv2-reranking +0.1317（均反超）、
CmedqaRetrieval +0.0910（追平 bge-m3）、AFQMC +0.0902、ATEC +0.0897、IFlyTek +0.0646、MedicalRetrieval +0.0391

**跌**：CovidRetrieval −0.1288、MMarcoRetrieval −0.0829、EcomRetrieval −0.0508、Cmnli −0.0494、
Ocnli −0.0471、MMarcoReranking −0.0423、DuRetrieval −0.0393、MultilingualSentiment −0.0387、
JDReview −0.0377、QBQTC −0.0366、T2Retrieval −0.0326

**诊断**：① **不是灾难性遗忘**（不在靶单里、本来也不弱的 LCQMC 只动 **−0.0002**）；
② **是数据覆盖缺口** —— 训练集只有「通用中文 STS 短句 + 医疗 QA」，**没有 NLI、没有通用检索的
(query, passage) 对**，于是练过的域涨、没练过的域跌（PairClassification 是纯副作用，
通用检索是域偏移：医疗检索涨 +0.09/+0.04 而通用检索跌 −0.03~−0.13）；③ 训练 `max_seq_length=192`
而评测用 2048，长文属分布外。

→ 缺口可补：**Ocnli/Cmnli 有官方 train split**（补 NLI 正负对）+ 引通用中文 (query, passage) 对 +
训练长度提到 512。下一步见 [ROADMAP.md](ROADMAP.md) 的 2.4b。

> ⚠️ **可比性**：AFQMC/ATEC/BQ/LCQMC 是用**它们自己的 train split** 训的（有监督 in-domain 微调），
> 这些分数**不能**与榜单上的零样本条目对比；医疗三项是 cMedQA-V2.0 → V1.0 的跨数据集迁移。

进度与下一步见 [ROADMAP.md](ROADMAP.md)。

## 环境

- Python 3.12（uv 管理）、torch 2.11.0+cu128（RTX 5070 Ti 是 sm_120，必须 cu128+）、
  transformers 5.19.0、sentence-transformers 6.1.0、mteb 2.24.0、torchvision 0.26.0+cu128、pillow、peft。
- 模型权重在 `models/embeddinggemma-2/`（基座）与 `models/embeddinggemma-2-zh-lora/`（微调产物），均未入库。

```bash
uv sync                 # 装依赖
uv run python src/sanity_check.py    # 冒烟 + 吞吐测量
```

## 网络（重要）

本机 `HF_ENDPOINT=https://hf-mirror.com` 会让 `hf download` 报
`Local entry not found. [SSL: UNEXPECTED_EOF_WHILE_READING]`。
可用组合是**直连 + 本地代理**（git config 里有 `http://127.0.0.1:7890`）：

```bash
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
```

后续所有需要联网的脚本都带上同样的前缀。

## 跑评测

```bash
# 全量 C-MTEB（MTEB(cmn, v1)，31 个中文任务）
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python src/eval_mteb.py --benchmark "MTEB(cmn, v1)" --tag base --batch-size 64

# 冒烟（几个小任务）
... src/eval_mteb.py --tasks Ocnli LCQMC AFQMC --tag smoke --batch-size 64

# 中断续跑
... --overwrite only-missing

# 微调后（8 个靶任务快评 / 全量）
./src/phase2/eval_finetuned.sh
./src/phase2/eval_finetuned.sh --full
```

结果写到 `results/mteb_<tag>_<时间戳>.{json,md}`。

**显存纪律**：`--vram-fraction`（默认 **0.7**）用 `torch.cuda.set_per_process_memory_fraction`
给进程上硬上限，宁可自己 OOM 退出也不拖垮整机；配合 `expandable_segments:True` 减少碎片。
注意 nvidia-smi 看到的是 **reserved**（caching allocator 会涨到上限），不是真实需求 ——
bs=64 与 bs=128 的 reserved 都是同一个 cap，但 `torch.cuda.max_memory_allocated()` 才反映真实峰值
（训练实测 9.85 GB / 上限 11.95 GB）。

## 已知坑（踩过的）

| 坑 | 现象 | 处理 |
|---|---|---|
| **整串媒体 URL 让评测崩掉** | 语料里出现「整条就是媒体 URL」的文档时，ST 把它当媒体去解码 → `ImportError: 要解码 YouTube 视频需要先装 yt_dlp`。MMarcoRetrieval 语料第 **98275** 条就是裸 YouTube 链接；`https://example.com/clip.mp4` 同样崩；`https://www.baidu.com` 正常；只有**整串**是媒体 URL 才触发 | `src/eval_mteb.py::pin_text_modality` —— pin 到纯 `text` modality（默认开启，`--no-text-pin` 可关）。定位脚本：`src/find_video_trigger.py` |
| **默认路径慢 4 倍** | chat-template/多模态 processor 路径 92 docs/s；pin 后 376 docs/s | 同上。已验证 200 条中文文本上两条路径 embedding 一致（`np.allclose(atol=1e-4) == True`），所以是白拿的加速 |
| mteb 未注册 embeddinggemma-2 | `mteb.get_model("google/embeddinggemma-2-text")` 不存在（mteb 2.24.0 只有 300m） | 用 `SentenceTransformerEncoderWrapper` 直接包本地模型；模型自带 `config_sentence_transformers.json` 的 task→prompt 映射，wrapper 会自动读取 |
| 缺依赖 | `EmbeddingGemma2Processor requires the PIL library` / `No module named 'torchvision'` | `pillow` + `torchvision`（torchvision 必须和 torch 同源 cu128 index） |
| `max_seq_length` 溢出 | ST 从 `max_position_embeddings`(262144) 解析出 ~1e60，等于不截断 | 显式设 `st.max_seq_length = 2048`（`--max-seq-length`）。**用户侧同样要设**，微调后的权重也是这个默认值 |
| 长文档 OOM | 中文检索语料里有 6 万字符的离群长文档，动态 padding 下会 CUDA OOM | `--vram-fraction 0.7` + `expandable_segments:True`；必要时降 `--batch-size` |
| prompt 键被忽略 | mteb 警告 `FactChecking / CodeRetrieval / SentenceSimilarity` 不是合法 task type 键并忽略 | 无害：mteb 真正会用到的 `Retrieval`/`Retrieval-query`/`Retrieval-document`/`Classification`/`Clustering`/`STS`/`Reranking`/`PairClassification` 都合法且生效 |
| fp16 | 模型卡：激活范围超 fp16，会静默 NaN/劣化 | 脚本一律 bf16（GPU）/ fp32（CPU） |
| **PEFT 撞视觉塔** | `ValueError: Target module Gemma4ClippableLinear(...) is not supported by PEFT` —— 该模块只在**视觉塔**里（768 宽） | 加 LoRA / 重载 / 评测一律带 `config_kwargs={"vision_config": None, "audio_config": None}`；不带就会连 1.5GB 的编解码器一起实例化 |
| **LoRA 打包写不出可用模型** | ST 用 peft 的 `inject_adapter_in_model` **原地注入** → `auto_model` 不是 PeftModel（没有 `merge_and_unload()`），`save_pretrained()` **只写 adapter**（无 config.json、无权重），而单独 `mod.merge()` 之后 wrapper 仍在树里（实测 193 个） | `src/phase2/train_lora.py::merge_and_unwrap()`：遍历 → `merge()` → 用 `get_base_layer()` 替换父模块的孩子；再用 `config.save_pretrained()` + `safetensors.save_file()` 手写（transformers 的 `save_pretrained` 检测到残留 peft 状态会走 adapter 分支并崩在 `active_adapters` UnboundLocalError） |
| 噪声文档被当训练正例 | STS train split 里低分对若不过阈值会污染对比学习 | `build_train_data.py` 按 label 尺度自适应阈值（0/1 → 0.6，0-5 → 3.0）只取正对 |
| **数据泄漏假通过** | 第一版泄漏检查报「重叠 0」，实为在比**空字符串** —— `cMedQA-V2.0` 的 `input` 列全空、真问题在 `instruction` 列 | `pick_col` 按非空比例选列；修正后实测**评测查询 100% 落在训练集内**（7,592 行命中），遂改为**查询级 + 文档级双重去污染** |

## 结构

```
src/sanity_check.py        加载/维度/prompt 前缀/MRL 截断/吞吐 自检
src/eval_mteb.py           mteb 评测（默认 MTEB(cmn, v1)），含 text-modality pin
src/make_comparison.py     由原始 JSON 生成 base/bgem3/微调 三方对比表
src/make_report.py         生成中文评测报告
src/publish_hf.py          发布卡片 + 权重 + 原始分数到 HF（含回读校验）
src/phase2/inspect_sources.py   训练源盘点
src/phase2/build_train_data.py  构造训练对（含双重去污染）
src/phase2/check_leakage.py     泄漏检查（含 STS train/eval 拆分重叠）
src/phase2/train_lora.py        LoRA 训练 + merge/unwrap + 打包 + 重载校验
src/phase2/eval_finetuned.sh    微调后评测（弱项快评 / --full 全量）
src/find_video_trigger.py       二分扫描语料，定位触发媒体解析的具体文档
src/throughput_test.py          真实语料上的 batch size / 吞吐 / 显存标定
results/                  原始分数（json + md）+ 三份参考结果
report/                   中文评测报告
logs/                     运行日志
models/                   本地权重（gitignore）
```
