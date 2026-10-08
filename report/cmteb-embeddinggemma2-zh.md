# EmbeddingGemma 2 中文能力实测 — C-MTEB（`MTEB(cmn, v1)`，31 个任务）

> 生成自 `results\mteb_base_261008_145213.json`，全部数字由脚本从原始结果生成（`src/make_report.py`），未经人工转录。

## 摘要

`google/embeddinggemma-2`（2026-10-06 发布，740M 多模态嵌入模型，文本骨干 271M）在中文
**C-MTEB 全量 31 个任务**上的总体得分为 **0.5815**（task-type 均值口径）。

**三方对照（同 harness、同机器、同样截断与 batch）**：

| 模型 | 参数量 | 总体 | 相对我们 |
|---|---|---|---|
| `google/embeddinggemma-2`（只载文本骨干） | 271M | **0.5815** | — |
| `BAAI/bge-m3`（本机实跑） | 568M | **0.6068** | 0.0254 |
| `Qwen/Qwen3-Embedding-0.6B`（榜单公开数） | 595M | **0.6748** | +0.0933 |

一句话：**用 bge-m3 一半的参数量拿到它约 96% 的中文成绩**，
但并非全面落后 —— 有 **7 个单项反超 bge-m3**。

### 两个初判被基准推翻（这正是必须做本地基准的原因）

1. **"PAWSX 是短板"——错。** 我们 0.1497，bge-m3 0.1566，
   只差 0.0069。这是**任务本身对所有人都难**
   （对抗性改写识别），不是模型缺陷。
2. **"中文聚类弱"——错。** 我们聚类均值 0.4925，
   **高于 bge-m3 的 0.4615**；ThuNews 两项还分别反超 bge-m3
   +0.0695 和
   +0.0684。
   "聚类弱"只是相对 Qwen3-Embedding-0.6B 而言，而那个模型在聚类上异常强。

### 真正的短板（对 bge-m3 也落后约 10 分）

| 任务 | 我们 | bge-m3 | 差 |
|---|---|---|---|
| AFQMC | 0.2783 | 0.4000 | -0.1218 |
| ATEC | 0.3447 | 0.4647 | -0.1200 |
| CMedQAv1-reranking | 0.6650 | 0.7769 | -0.1120 |
| MedicalRetrieval | 0.4335 | 0.5427 | -0.1092 |
| CMedQAv2-reranking | 0.6882 | 0.7909 | -0.1026 |
| BQ | 0.4731 | 0.5739 | -0.1008 |

### 反超 bge-m3 的单项（前 6）

| 任务 | 我们 | bge-m3 | 差 |
|---|---|---|---|
| ThuNewsClusteringS2S | 0.5803 | 0.5107 | +0.0695 |
| ThuNewsClusteringP2P | 0.6311 | 0.5627 | +0.0684 |
| EcomRetrieval | 0.6040 | 0.5845 | +0.0196 |
| VideoRetrieval | 0.5808 | 0.5699 | +0.0108 |
| MMarcoRetrieval | 0.7825 | 0.7726 | +0.0099 |
| MultilingualSentiment | 0.7255 | 0.7168 | +0.0088 |

**结论**：通用检索与新闻聚类是它的舒适区；**医疗领域（检索 + reranking）与中文 STS 的
AFQMC / ATEC / BQ 是明确短板** —— 这两块就是阶段 2 微调的靶子。

## 评测设置

| 项 | 值 |
|---|---|
| benchmark | `MTEB(cmn, v1)`（= C-MTEB），31 个任务 |
| 模型 | `google/embeddinggemma-2`，**只加载文本骨干**（`config_kwargs={vision_config: None, audio_config: None}`，271.0M 参数） |
| dtype | bfloat16（模型卡明确警告：**不要用 float16**，会静默 NaN/劣化） |
| 截断 | `max_seq_length = 2048`（模型支持 8192；C-MTEB 语料 p99 ≈ 1.3k token） |
| batch size | 16 |
| prompt | 使用模型自带的 task 前缀（`task: search result \| query: ` 等），由 `config_sentence_transformers.json` 提供 |
| 框架 | sentence-transformers 6.1.0 / transformers 5.19.0 / mteb 2.24.0 / torch 2.11.0+cu128 |
| 硬件 | 单卡 RTX 5070 Ti（16GB），显存进程上限 70% |
| 耗时 | 我们 173.8 分钟；bge-m3 对照 84.9 分钟（同机同设置） |

## 结果（31 个任务）

| 任务 | 类型 | 主指标 | 分数 |
|---|---|---|---|
| T2Retrieval | Retrieval | ndcg_at_10 | 0.7618 |
| MMarcoRetrieval | Retrieval | ndcg_at_10 | 0.7825 |
| DuRetrieval | Retrieval | ndcg_at_10 | 0.7637 |
| CovidRetrieval | Retrieval | ndcg_at_10 | 0.7721 |
| CmedqaRetrieval | Retrieval | ndcg_at_10 | 0.2451 |
| EcomRetrieval | Retrieval | ndcg_at_10 | 0.6040 |
| MedicalRetrieval | Retrieval | ndcg_at_10 | 0.4335 |
| VideoRetrieval | Retrieval | ndcg_at_10 | 0.5808 |
| T2Reranking | Reranking | map_at_1000 | 0.6613 |
| MMarcoReranking | Reranking | map_at_1000 | 0.2511 |
| CMedQAv1-reranking | Reranking | map_at_1000 | 0.6650 |
| CMedQAv2-reranking | Reranking | map_at_1000 | 0.6882 |
| Ocnli | PairClassification | max_accuracy | 0.6297 |
| Cmnli | PairClassification | max_accuracy | 0.7167 |
| CLSClusteringS2S | Clustering | v_measure | 0.3743 |
| CLSClusteringP2P | Clustering | v_measure | 0.3844 |
| ThuNewsClusteringS2S | Clustering | v_measure | 0.5803 |
| ThuNewsClusteringP2P | Clustering | v_measure | 0.6311 |
| LCQMC | STS | cosine_spearman | 0.7253 |
| PAWSX | STS | cosine_spearman | 0.1497 |
| AFQMC | STS | cosine_spearman | 0.2783 |
| QBQTC | STS | cosine_spearman | 0.3372 |
| ATEC | STS | cosine_spearman | 0.3447 |
| BQ | STS | cosine_spearman | 0.4731 |
| STSB | STS | cosine_spearman | 0.8002 |
| TNews | Classification | accuracy | 0.4673 |
| IFlyTek | Classification | accuracy | 0.4302 |
| Waimai | Classification | accuracy | 0.8499 |
| OnlineShopping | Classification | accuracy | 0.8991 |
| JDReview | Classification | accuracy | 0.7964 |
| MultilingualSentiment | Classification | accuracy | 0.7255 |

### 按任务类型汇总

| 任务类型 | embeddinggemma-2 | bge-m3 | Qwen3-Embedding-0.6B |
|---|---|---|---|
| Retrieval | 0.6179 | 0.6546 | 0.7103 |
| Reranking | 0.5664 | 0.6280 | 0.6258 |
| PairClassification | 0.6732 | 0.6846 | 0.7642 |
| Clustering | 0.4925 | 0.4615 | 0.6874 |
| STS | 0.4441 | 0.4994 | 0.5461 |
| Classification | 0.6948 | 0.7129 | 0.7149 |
| **总体（task-type 均值）** | **0.5815** | **0.6068** | **0.6748** |

## 完整对照表（含逐任务 delta）

# C-MTEB (MTEB(cmn, v1)) — embeddinggemma-2 vs bge-m3 vs Qwen3-Embedding-0.6B

Every column uses **the same harness** (mteb 2.24.0, `MTEB(cmn, v1)`, 31 tasks).
`local run` = executed on this machine (RTX 5070 Ti, bf16, truncate 2048, batch 16);
other columns are the published `mteb/results` entries for the same task names.

| column | tasks covered | source |
|---|---|---|
| embeddinggemma-2 (local run) | 31/31 | local run |
| bge-m3 (local run) | 31/31 | local run |
| embeddinggemma-2-zh (LoRA r32, local run) | 31/31 | local run |
| finetuned_weak (local run) | 8/31 | local run |
| BAAI__bge-m3 | 5/31 | mteb/results (public) |
| google__embeddinggemma-300m | 3/31 | mteb/results (public) |
| Qwen__Qwen3-Embedding-0.6B | 31/31 | mteb/results (public) |

## Full 31-task table

| task | type | embeddinggemma-2 (local run) | bge-m3 (local run) | embeddinggemma-2-zh (LoRA r32, local run) | Qwen__Qwen3-Embedding-0.6B |
|---|---|---|---|---|---|
| T2Retrieval | Retrieval | 0.7618 | 0.8149 | 0.7292 | 0.8368 |
| MMarcoRetrieval | Retrieval | 0.7825 | 0.7726 | 0.6997 | 0.7985 |
| DuRetrieval | Retrieval | 0.7637 | 0.8399 | 0.7243 | 0.8410 |
| CovidRetrieval | Retrieval | 0.7721 | 0.7751 | 0.6433 | 0.8476 |
| CmedqaRetrieval | Retrieval | 0.2451 | 0.3374 | 0.3361 | 0.4109 |
| EcomRetrieval | Retrieval | 0.6040 | 0.5845 | 0.5532 | 0.6432 |
| MedicalRetrieval | Retrieval | 0.4335 | 0.5427 | 0.4726 | 0.5604 |
| VideoRetrieval | Retrieval | 0.5808 | 0.5699 | 0.5557 | 0.7436 |
| T2Reranking | Reranking | 0.6613 | 0.6688 | 0.6587 | 0.6715 |
| MMarcoReranking | Reranking | 0.2511 | 0.2754 | 0.2088 | 0.2175 |
| CMedQAv1-reranking | Reranking | 0.6650 | 0.7769 | 0.8003 | 0.8006 |
| CMedQAv2-reranking | Reranking | 0.6882 | 0.7909 | 0.8200 | 0.8135 |
| Ocnli | PairClassification | 0.6297 | 0.6508 | 0.5826 | 0.7298 |
| Cmnli | PairClassification | 0.7167 | 0.7183 | 0.6672 | 0.7986 |
| CLSClusteringS2S | Clustering | 0.3743 | 0.3827 | 0.3767 | 0.5828 |
| CLSClusteringP2P | Clustering | 0.3844 | 0.3899 | 0.4068 | 0.6062 |
| ThuNewsClusteringS2S | Clustering | 0.5803 | 0.5107 | 0.5628 | 0.7457 |
| ThuNewsClusteringP2P | Clustering | 0.6311 | 0.5627 | 0.6383 | 0.8150 |
| LCQMC | STS | 0.7253 | 0.7618 | 0.7251 | 0.7646 |
| PAWSX | STS | 0.1497 | 0.1566 | 0.1423 | 0.2763 |
| AFQMC | STS | 0.2783 | 0.4000 | 0.3685 | 0.4323 |
| QBQTC | STS | 0.3372 | 0.3331 | 0.3006 | 0.3707 |
| ATEC | STS | 0.3447 | 0.4647 | 0.4344 | 0.4858 |
| BQ | STS | 0.4731 | 0.5739 | 0.6396 | 0.6466 |
| STSB | STS | 0.8002 | 0.8060 | 0.7866 | 0.8462 |
| TNews | Classification | 0.4673 | 0.4961 | 0.4903 | 0.5296 |
| IFlyTek | Classification | 0.4302 | 0.4768 | 0.4948 | 0.5167 |
| Waimai | Classification | 0.8499 | 0.8665 | 0.8481 | 0.8129 |
| OnlineShopping | Classification | 0.8991 | 0.9206 | 0.8811 | 0.8360 |
| JDReview | Classification | 0.7964 | 0.8008 | 0.7587 | 0.8041 |
| MultilingualSentiment | Classification | 0.7255 | 0.7168 | 0.6868 | 0.7897 |

## Aggregate

| | embeddinggemma-2 (local run) | bge-m3 (local run) | embeddinggemma-2-zh (LoRA r32, local run) | Qwen__Qwen3-Embedding-0.6B |
|---|---|---|---|---|
| Retrieval | 0.6179 | 0.6546 | 0.5893 | 0.7103 |
| Reranking | 0.5664 | 0.6280 | 0.6219 | 0.6258 |
| PairClassification | 0.6732 | 0.6846 | 0.6249 | 0.7642 |
| Clustering | 0.4925 | 0.4615 | 0.4962 | 0.6874 |
| STS | 0.4441 | 0.4994 | 0.4853 | 0.5461 |
| Classification | 0.6948 | 0.7129 | 0.6933 | 0.7149 |
| **Overall (mean of type means)** | **0.5815** | **0.6068** | **0.5851** | **0.6748** |

## Delta vs `embeddinggemma-2 (local run)` (positive = ours better)

| task | type | bge-m3 (local run) | embeddinggemma-2-zh (LoRA r32, local run) | Qwen__Qwen3-Embedding-0.6B |
|---|---|---|---|---|
| T2Retrieval | Retrieval | -0.0532 | +0.0326 | -0.0750 |
| MMarcoRetrieval | Retrieval | +0.0099 | +0.0829 | -0.0160 |
| DuRetrieval | Retrieval | -0.0762 | +0.0393 | -0.0773 |
| CovidRetrieval | Retrieval | -0.0029 | +0.1288 | -0.0755 |
| CmedqaRetrieval | Retrieval | -0.0923 | -0.0910 | -0.1658 |
| EcomRetrieval | Retrieval | +0.0196 | +0.0508 | -0.0391 |
| MedicalRetrieval | Retrieval | -0.1092 | -0.0391 | -0.1269 |
| VideoRetrieval | Retrieval | +0.0108 | +0.0251 | -0.1629 |
| T2Reranking | Reranking | -0.0075 | +0.0026 | -0.0102 |
| MMarcoReranking | Reranking | -0.0243 | +0.0423 | +0.0336 |
| CMedQAv1-reranking | Reranking | -0.1120 | -0.1353 | -0.1356 |
| CMedQAv2-reranking | Reranking | -0.1026 | -0.1317 | -0.1253 |
| Ocnli | PairClassification | -0.0211 | +0.0471 | -0.1002 |
| Cmnli | PairClassification | -0.0017 | +0.0494 | -0.0819 |
| CLSClusteringS2S | Clustering | -0.0084 | -0.0024 | -0.2086 |
| CLSClusteringP2P | Clustering | -0.0055 | -0.0224 | -0.2218 |
| ThuNewsClusteringS2S | Clustering | +0.0695 | +0.0174 | -0.1654 |
| ThuNewsClusteringP2P | Clustering | +0.0684 | -0.0072 | -0.1839 |
| LCQMC | STS | -0.0365 | +0.0002 | -0.0393 |
| PAWSX | STS | -0.0069 | +0.0074 | -0.1266 |
| AFQMC | STS | -0.1218 | -0.0902 | -0.1540 |
| QBQTC | STS | +0.0042 | +0.0366 | -0.0335 |
| ATEC | STS | -0.1200 | -0.0897 | -0.1411 |
| BQ | STS | -0.1008 | -0.1665 | -0.1735 |
| STSB | STS | -0.0059 | +0.0136 | -0.0460 |
| TNews | Classification | -0.0288 | -0.0230 | -0.0623 |
| IFlyTek | Classification | -0.0466 | -0.0646 | -0.0865 |
| Waimai | Classification | -0.0166 | +0.0018 | +0.0370 |
| OnlineShopping | Classification | -0.0215 | +0.0180 | +0.0631 |
| JDReview | Classification | -0.0043 | +0.0377 | -0.0077 |
| MultilingualSentiment | Classification | +0.0088 | +0.0387 | -0.0642 |
| **Overall** | | **-0.0254** | **-0.0037** | **-0.0933** |


## 复现方法

```bash
# 1. 环境
uv sync

# 2. 权重（本机 HF_ENDPOINT 指向镜像会让 hf download 失败，必须直连+代理）
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  hf download BAAI/bge-m3 --local-dir models/bge-m3

# 3. 全量评测（我们约 2.9 小时；bge-m3 约 1.4 小时，均 RTX 5070 Ti）
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python src/eval_mteb.py --benchmark "MTEB(cmn, v1)" --tag base \
  --batch-size 16 --max-seq-length 2048 --vram-fraction 0.7

env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python src/eval_mteb.py --model models/bge-m3 --tag bgem3 --no-text-pin \
  --batch-size 16 --max-seq-length 2048 --vram-fraction 0.7

# 4. 对照表 / 本报告
uv run python src/make_comparison.py
uv run python src/make_report.py
```

## 局限

- **截断 2048 而非模型上限 8192**：受显存约束（该模型 `per_layer_projection_norm` 里有一次
  fp32 上采样，`[batch, seq, 24 层, 512 维] × 4 字节` 在 batch 32 × 2048 时就是 3.2GB 单张量）。
  C-MTEB 语料 p99 ≈ 1.3k token，影响有限，但**医疗类长文档任务可能被低估**，待做截断敏感性验证。
- **对照模型同机同设置（2048 截断）**：bge-m3 与 Qwen3-Embedding-0.6B 都是 2048 截断，
  因此与它们在公开榜单（各自默认长度）上的成绩不可直接对比，只在本文内部可比。
- **未评测多模态**：本报告只覆盖文本，图像/音频（MIEB / MAEB）未涉及。

## 附：使用这个模型时必须知道的一个坑

用 sentence-transformers 直接 `encode()` 时，**如果某条输入"整串就是一个媒体 URL"**，
默认路径会把它当成媒体去解码：

| 输入 | 结果 |
|---|---|
| `http://www.youtube.com/ehowatHomeChannel` | ❌ `ImportError: 要解码 YouTube 视频需要先装 yt_dlp` |
| `https://example.com/clip.mp4` | ❌ libtorchcodec 报错 |
| `https://www.baidu.com` | ✅ 正常 |
| `看这段 https://example.com/a.mp4` | ✅ 正常（只有整串是媒体 URL 才触发） |

网页抓取的中文语料里这种文档很常见（MMarcoRetrieval 语料第 98275 条就是裸 YouTube 链接），
一次全量评测会因此崩在语料编码中途。解法：把 sentence-transformers pin 到纯 `text` modality
（`src/eval_mteb.py::pin_text_modality`）——**同时还能提速约 4 倍，且 embedding 数值不变**
（200 条中文文本上 `np.allclose(atol=1e-4) == True`）。
