# embeddinggemma2-zh

`google/embeddinggemma-2`（2026-10-06 发布的 740M 多模态嵌入模型）的**中文评测与微调**。

> **中文圈第一个 C-MTEB 全量实测。** 发布后 48 小时内，声明
> `base_model=google/embeddinggemma-2` 的 60 个衍生仓库里**零中文、零 C-MTEB 数字**
> （GitHub 搜 `embeddinggemma + C-MTEB` = 0 仓库，`+ chinese` = 0 仓库），本项目补这个空白。

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

进度与下一步见 [ROADMAP.md](ROADMAP.md)。

## 环境

- Python 3.12（uv 管理）、torch 2.11.0+cu128（RTX 5070 Ti 是 sm_120，必须 cu128+）、
  transformers 5.19.0、sentence-transformers 6.1.0、mteb 2.24.0、torchvision 0.26.0+cu128、pillow。
- 模型权重在 `models/embeddinggemma-2/`（本地目录，未入库）。

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
```

结果写到 `results/mteb_<tag>_<时间戳>.{json,md}`。

**显存纪律**：`--vram-fraction`（默认 **0.7**）用 `torch.cuda.set_per_process_memory_fraction`
给进程上硬上限，宁可自己 OOM 退出也不拖垮整机；配合 `expandable_segments:True` 减少碎片。

## 已知坑（踩过的）

| 坑 | 现象 | 处理 |
|---|---|---|
| **整串媒体 URL 让评测崩掉** | 语料里出现「整条就是媒体 URL」的文档时，ST 把它当媒体去解码 → `ImportError: 要解码 YouTube 视频需要先装 yt_dlp`。MMarcoRetrieval 语料第 **98275** 条就是裸 YouTube 链接；`https://example.com/clip.mp4` 同样崩；`https://www.baidu.com` 正常；只有**整串**是媒体 URL 才触发 | `src/eval_mteb.py::pin_text_modality` —— pin 到纯 `text` modality（默认开启，`--no-text-pin` 可关）。定位脚本：`src/find_video_trigger.py` |
| **默认路径慢 4 倍** | chat-template/多模态 processor 路径 92 docs/s；pin 后 376 docs/s | 同上。已验证 200 条中文文本上两条路径 embedding 一致（`np.allclose(atol=1e-4) == True`），所以是白拿的加速 |
| mteb 未注册 embeddinggemma-2 | `mteb.get_model("google/embeddinggemma-2-text")` 不存在（mteb 2.24.0 只有 300m） | 用 `SentenceTransformerEncoderWrapper` 直接包本地模型；模型自带 `config_sentence_transformers.json` 的 task→prompt 映射，wrapper 会自动读取 |
| 缺依赖 | `EmbeddingGemma2Processor requires the PIL library` / `No module named 'torchvision'` | `pillow` + `torchvision`（torchvision 必须和 torch 同源 cu128 index） |
| `max_seq_length` 溢出 | ST 从 `max_position_embeddings`(262144) 解析出 ~1e60，等于不截断 | 显式设 `st.max_seq_length = 2048`（`--max-seq-length`） |
| 长文档 OOM | 中文检索语料里有 6 万字符的离群长文档，动态 padding 下会 CUDA OOM | `--vram-fraction 0.7` + `expandable_segments:True`；必要时降 `--batch-size` |
| prompt 键被忽略 | mteb 警告 `FactChecking / CodeRetrieval / SentenceSimilarity` 不是合法 task type 键并忽略 | 无害：mteb 真正会用到的 `Retrieval`/`Retrieval-query`/`Retrieval-document`/`Classification`/`Clustering`/`STS`/`Reranking`/`PairClassification` 都合法且生效 |
| fp16 | 模型卡：激活范围超 fp16，会静默 NaN/劣化 | 脚本一律 bf16（GPU）/ fp32（CPU） |

## 结构

```
src/sanity_check.py       加载/维度/prompt 前缀/MRL 截断/吞吐 自检
src/eval_mteb.py          mteb 评测（默认 MTEB(cmn, v1)），含 text-modality pin
src/find_video_trigger.py 二分扫描语料，定位触发媒体解析的具体文档
src/throughput_test.py    真实语料上的 batch size / 吞吐 / 显存标定
results/                  原始分数（json + md）
logs/                     运行日志
models/                   本地权重（gitignore）
```
