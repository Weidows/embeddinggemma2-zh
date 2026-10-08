---
license: apache-2.0
language:
- zh
library_name: sentence-transformers
base_model: google/embeddinggemma-2
pipeline_tag: feature-extraction
tags:
- embeddinggemma
- embedding
- chinese
- c-mteb
- mteb
- sentence-transformers
---

# embeddinggemma-2-zh

`google/embeddinggemma-2`（2026-10-06 发布，740M 多模态嵌入模型）的**中文能力实测**与**中文微调**工作仓。

本仓库包含两样东西：

| | 内容 | 状态 |
|---|---|---|
| ① | `google/embeddinggemma-2` 文本骨干的 **C-MTEB 全量中文实测**（31 任务，同机同 harness 对照 bge-m3） | ✅ 结论明确，见下 |
| ② | 在其上做的**中文 LoRA 微调权重**（本仓库根目录的 `model.safetensors`） | ⚠️ **实测结论是「零和」**，用前请务必读完「中文微调」一节 |

> 微调**不是**本仓库的卖点，**结论**才是。下面把涨的和跌的全部列出，不做选择性披露。
> 代码与完整原始结果：<https://github.com/Weidows/embeddinggemma2-zh>

## 为什么做这个

`google/embeddinggemma-2` 的官方评测覆盖 MTEB（多语言 v2）、MIEB、MAEB，
但**发布后 48 小时内没有任何中文基准数字**：声明 `base_model=google/embeddinggemma-2`
的 60 个衍生仓库里零中文；GitHub 搜 `embeddinggemma + C-MTEB` 与 `+ chinese` 均为 0 仓库；
中文圈内容全部是转载/新闻。（核实时间：2026-10-08）

## 实测结果 — C-MTEB（`MTEB(cmn, v1)`，31 个中文任务）

只加载**文本骨干**（`config_kwargs={"vision_config": None, "audio_config": None}`，271.0M 参数），
bf16，截断 2048，batch 16。

| 模型 | 参数量 | 总体 | 来源 |
|---|---|---|---|
| **`google/embeddinggemma-2`**（文本骨干） | **271M** | **0.5815** | 本仓实测，单卡 RTX 5070 Ti，173.8 min |
| `BAAI/bge-m3` | 568M | **0.6068** | 同 harness 同机实跑，84.9 min |
| `Qwen/Qwen3-Embedding-0.6B` | 595M | **0.6748** | `mteb/results` 公开条目 |

**用 bge-m3 一半的参数量，拿到它约 96% 的中文成绩**；7 个单项反超 bge-m3。

### 按任务类型

| 任务类型 | embeddinggemma-2 | bge-m3 | Qwen3-Embedding-0.6B |
|---|---|---|---|
| Retrieval | 0.6179 | 0.6546 | 0.7103 |
| Reranking | 0.5664 | 0.6280 | 0.6258 |
| PairClassification | 0.6732 | 0.6846 | 0.7642 |
| Clustering | **0.4925** | 0.4615 | 0.6874 |
| STS | 0.4441 | 0.4994 | 0.5461 |
| Classification | 0.6948 | 0.7129 | 0.7149 |

### 明确的短板（对 bge-m3 落后约 0.10）

| 任务 | 本模型 | bge-m3 | 差 |
|---|---|---|---|
| AFQMC | 0.2783 | 0.4000 | −0.1218 |
| ATEC | 0.3447 | 0.4647 | −0.1200 |
| CMedQAv1-reranking | 0.6650 | 0.7769 | −0.1120 |
| MedicalRetrieval | 0.4335 | 0.5427 | −0.1092 |
| CMedQAv2-reranking | 0.6882 | 0.7909 | −0.1026 |
| BQ | 0.4731 | 0.5739 | −0.1008 |

→ 收敛为两块：**医疗领域（检索 + reranking）** 与 **中文 STS（AFQMC / ATEC / BQ）**。

### 两点需要澄清（容易被误读）

- **PAWSX 0.1497 不是它的短板**：bge-m3 也只有 0.1566，这是对抗性改写任务本身难。
- **中文聚类并不弱**：本模型 0.4925 **高于** bge-m3 0.4615（ThuNews 两项各反超约 0.07）；
  只是不及 Qwen3-Embedding-0.6B。

## 中文微调（LoRA r=32）— 结论：**零和**

针对上面 6 个短板做的定向微调。**结果是净收益 +0.0037，实质为零**，涨的和跌的几乎抵消。

### 方法

| 项 | 值 |
|---|---|
| 基座 | `google/embeddinggemma-2` **文本骨干**（271.0M） |
| LoRA | r=32 / α=64 / dropout 0.05 / 9 个 target（`q,k,v,o,gate,up,down` + `embedding_projection` + 24 层 `per_layer_projection`），**可训练 10.92M（4.03%）** |
| 训练数据 | **273,549 对** = 中文 STS **156k**（AFQMC/ATEC/BQ/LCQMC/PAWSX/STSB 的 **train split**）+ 医疗 **120k**（cMedQA-V2.0 的 question→answer） |
| 去污染 | STS 侧剔除与评测 pair 完全相同的 91 条；医疗侧剔除命中评测查询的 7,592 行 + 命中评测语料的 8,776 行 → **查询/文档重叠均为 0** |
| 超参 | 1 epoch / bs 64 / lr 2e-4 / `max_seq_length` 192 / bf16 / **禁用 fp16** |
| 成本 | **85.2 min**，train_loss 0.9009 → 0.2512，eval_loss 0.2099，峰值显存 9.85 GB |

### 结果（31 任务，微调前 → 微调后）

| 任务类型 | 微调前 | 微调后 | Δ |
|---|---|---|---|
| Retrieval | 0.6179 | 0.5893 | **−0.0287** |
| Reranking | 0.5664 | 0.6219 | **+0.0555** |
| PairClassification | 0.6732 | 0.6249 | **−0.0483** |
| Clustering | 0.4925 | 0.4962 | +0.0037 |
| STS | 0.4441 | 0.4853 | **+0.0412** |
| Classification | 0.6948 | 0.6933 | −0.0014 |
| **总体** | **0.5815** | **0.5851** | **+0.0037** |

**涨的**

| 任务 | 微调前 | 微调后 | Δ | vs bge-m3 |
|---|---|---|---|---|
| BQ | 0.4731 | 0.6396 | +0.1665 | **+0.0657 反超** |
| CMedQAv1-reranking | 0.6650 | 0.8003 | +0.1353 | **+0.0234 反超** |
| CMedQAv2-reranking | 0.6882 | 0.8200 | +0.1317 | **+0.0291 反超** |
| CmedqaRetrieval | 0.2451 | 0.3361 | +0.0910 | −0.0013 追平 |
| AFQMC | 0.2783 | 0.3685 | +0.0902 | −0.0315 |
| ATEC | 0.3447 | 0.4344 | +0.0897 | −0.0303 |
| IFlyTek | 0.4302 | 0.4948 | +0.0646 | **+0.0180 反超** |
| MedicalRetrieval | 0.4335 | 0.4726 | +0.0391 | −0.0701 |

**跌的**

| 任务 | 微调前 | 微调后 | Δ |
|---|---|---|---|
| CovidRetrieval | 0.7721 | 0.6433 | **−0.1288** |
| MMarcoRetrieval | 0.7825 | 0.6997 | **−0.0829** |
| EcomRetrieval | 0.6040 | 0.5532 | −0.0508 |
| Cmnli | 0.7167 | 0.6672 | −0.0494 |
| Ocnli | 0.6297 | 0.5826 | −0.0471 |
| MMarcoReranking | 0.2511 | 0.2088 | −0.0423 |
| DuRetrieval | 0.7637 | 0.7243 | −0.0393 |
| MultilingualSentiment | 0.7255 | 0.6868 | −0.0387 |
| JDReview | 0.7964 | 0.7587 | −0.0377 |
| QBQTC | 0.3372 | 0.3006 | −0.0366 |
| T2Retrieval | 0.7618 | 0.7292 | −0.0326 |

### 为什么会零和（诊断）

1. **不是灾难性遗忘**：不在靶单里、本来也不弱的 LCQMC 只动了 **−0.0002**。通用能力没有雪崩。
2. **是数据覆盖缺口**：训练集只有「通用中文 STS 短句 + 医疗 QA」，**没有 NLI 数据、没有通用检索的 (query, passage) 对**。
   结果就是**练过的域涨、没练过的域跌** —— PairClassification（Ocnli/Cmnli）是纯副作用，
   通用检索（Covid/MMarco/Du/Ecom/T2）是域偏移（医疗检索涨 +0.09/+0.04，通用检索跌 −0.03~−0.13）。
3. **长文属分布外**：训练时 `max_seq_length=192`，而评测用 2048。

→ 想把这轮微调做成正收益，缺口的补法是明确的：补 **Ocnli/Cmnli 的 train split**（NLI 正负对，
这两个数据集本身就有 train split）和 **通用中文 (query, passage) 对**，并把训练长度提到 512。
本轮没有继续迭代 —— **把这个"零和"结论和完整回归数据如实公开**，比发布一个只报喜的权重更有用。

### ⚠️ 可比性声明（重要）

| 任务 | 训练数据来源 | 能否与零样本模型比 |
|---|---|---|
| AFQMC / ATEC / BQ / LCQMC | **它们自己的官方 train split** | ❌ **不能**。这是有监督 in-domain 微调，不是零样本成绩，不要拿去和榜单上的零样本条目对比 |
| 医疗三项（MedicalRetrieval / CmedqaRetrieval / CMedQAv1&2-reranking） | cMedQA-**V2.0**（与评测所用的 cMedQA-V1.0 不同版本，已去污染） | ✅ 跨数据集迁移 |

## 用法

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Weidows/embeddinggemma-2-zh")   # 271M 文本骨干
model.max_seq_length = 2048   # 建议显式设置（见下）
emb = model.encode(["蚂蚁借呗怎么关闭", "花呗额度怎么提升"], normalize_embeddings=True)
```

> ⚠️ 建议显式设置 `max_seq_length`：该模型的 config 会让 sentence-transformers 解析出一个溢出的
> 默认值（约 `1e30`，等于**完全不截断**）。这与微调无关，原版 `google/embeddinggemma-2` 同样如此。

**注意**：本权重是**纯文本骨干**版（`config.json` 里 `vision_config` / `audio_config` 均为 `null`），
不含原版 740M 的图像/音频塔，因此不支持多模态输入。

如果你想改从源码仓库 `google/embeddinggemma-2` 手动加载配置，而不是用本仓库的 `config.json`：

```python
# 必须显式关掉编解码器：否则会实例化视觉塔（768 宽），
# 其 Gemma4ClippableLinear 会与 peft 冲突，也会白白加载 1.5GB 权重
SentenceTransformer("google/embeddinggemma-2",
                    config_kwargs={"vision_config": None, "audio_config": None})
```

## 实测中发现的集成缺陷（值得上游注意）

用 sentence-transformers 直接 `encode()` 时，**若某条输入"整串就是一个媒体 URL"**，
默认路径会把它当作媒体去解码：

| 输入 | 结果 |
|---|---|
| `http://www.youtube.com/ehowatHomeChannel` | ❌ `ImportError: 要解码 YouTube 视频需要先装 yt_dlp` |
| `https://example.com/clip.mp4` | ❌ libtorchcodec 报错 |
| `https://www.baidu.com` | ✅ 正常 |
| `看这段 https://example.com/a.mp4` | ✅ 正常（只有整串是媒体 URL 才触发） |

网页抓取的中文语料里这类文档很常见（`mteb/MMarcoRetrieval` 语料第 **98275** 条就是裸 YouTube 链接），
一次全量评测会因此崩在语料编码中途。把 sentence-transformers pin 到纯 `text` modality 可绕开，
**且 embedding 数值不变**（200 条中文文本上 `np.allclose(atol=1e-4) == True`），
吞吐同时提升约 4 倍（92 → 376 docs/s）。

## 复现

```bash
git clone https://github.com/Weidows/embeddinggemma2-zh && cd embeddinggemma2-zh
uv sync
./src/phase2/eval_finetuned.sh          # 微调后权重跑 8 个靶任务（约 29 min）
./src/phase2/eval_finetuned.sh --full   # 全量 31 任务（约 174 min）
```

本机需要直连 + 本地代理，**不能用 HF 镜像**（镜像会让 `hf download` 报
`Local entry not found. [SSL: UNEXPECTED_EOF_WHILE_READING]`）。

## 引用 / 数据出处

- 本模型评测：`mteb` 2.24.0，benchmark `MTEB(cmn, v1)`，逐任务原始分数在同仓 `results/`，
  本仓库 `eval/` 下也放了一份（含微调前后的原始 JSON）
- bge-m3 与 Qwen3-Embedding-0.6B 的对照：前者本机同设置实跑，后者取自 `mteb/results` 公开条目

## 局限

- 截断 2048（模型支持 8192）：受显存约束（文本塔的 `per_layer_projection_norm` 会把
  `[batch, seq, 24 层, 512 维]` 上采样到 fp32）。C-MTEB 语料 p99 ≈ 1.3k token，影响有限，
  但**医疗长文档任务可能被低估**。
- 只评测了文本，图像/音频（MIEB / MAEB）未涉及。
- 对照模型均为 2048 截断，因此与它们在公开榜单（各自默认长度）上的成绩不可直接对比。
- **微调权重仅 1 epoch、无早停、无超参搜索**，且如上所述训练长度 192 < 评测长度 2048；
  它在靶任务上的提升是真实的，但通用检索上的退化同样是真实的。
