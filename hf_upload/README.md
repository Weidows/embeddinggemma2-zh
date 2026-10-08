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

> ⚠️ **本仓库目前不含微调权重**。先发布评测结果，微调完成后会补上权重。
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
# 见 README 的「跑评测」一节的命令（本机需直连 + 代理，不能用 HF 镜像）
```

## 引用 / 数据出处

- 本模型评测：`mteb` 2.24.0，benchmark `MTEB(cmn, v1)`，逐任务原始分数在同仓 `results/`
- bge-m3 与 Qwen3-Embedding-0.6B 的对照：前者本机同设置实跑，后者取自 `mteb/results` 公开条目

## 局限

- 截断 2048（模型支持 8192）：受显存约束（文本塔的 `per_layer_projection_norm` 会把
  `[batch, seq, 24 层, 512 维]` 上采样到 fp32）。C-MTEB 语料 p99 ≈ 1.3k token，影响有限，
  但**医疗长文档任务可能被低估**。
- 只评测了文本，图像/音频（MIEB / MAEB）未涉及。
- 对照模型均为 2048 截断，因此与它们在公开榜单（各自默认长度）上的成绩不可直接对比。
